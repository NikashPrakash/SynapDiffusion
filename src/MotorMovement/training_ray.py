#training_ray.py
from src.models import *
import torch#, pdb
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
import os
import tempfile
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    FullOptimStateDictConfig,
    StateDictType,
    ShardingStrategy
)
from ray import train
from ray.train.torch import get_device, prepare_model, prepare_optimizer, TorchCheckpoint
from src.common import config

__dirname__ = "/src/MotorMovement"
class Trainer:
    def setup(self, config: dict):
        """
        config = {
            'dataset_size': 2743, 'model_class': <MEGDecoder>, 'batch_size': 32, 
            'hparams': {'architecture': 'var-cnn', 'n_classes': 2},
            'lr': 0.0007512123221743854, 'lr_factor': 0.1, 'lr_patience': 2, 
            'params': {'dropout': 0.5, 'filter_length': 7, 'n_ls': 32, 
            'nonlin_hid': <ReLU>, 'nonlin_in': <Identity>, 'nonlin_out': <Identity>,
            'pooling': 2, 'stride': 1},
            'patience': 8, 
            'parallel_setup': {'sharding_strategy': <ShardingStrategy.NO_SHARD: 3>, 'mixed_precision':torch.half..., 
            'weight_decay': 0.006964722130250569,
            'regulaizer': <function main.<locals>.<lambda> at 0x14c234049fc0>
        }
        """
        self.rank = int(os.environ.get("RANK",0))
        self.patience = config['patience']
        self.curr_count_to_patience = 0
        
        self.batch_size = config['batch_size'] // train.get_context().get_world_size()
        # total_steps_cyc = config['dataset_size'] // config['batch_size']

        self.regulaizer = config.get('regulaizer', None)
        self.criterion = CrossEntropyLoss()
        
        if issubclass(config['model_class'], MEGDecoder):
            self.model = MEGDecoder(config['hparams'], config['params'])
        if issubclass(config['model_class'], D_MI_WaveletCNN):
            self.model = D_MI_WaveletCNN()
            
        if config['parallel_setup']['sharding_strategy'] != 'CPU':
            self.model = prepare_model(self.model, parallel_strategy='fsdp', parallel_strategy_kwargs=config['parallel_setup'])
            self.model = torch.nn.utils.convert_conv2d_weight_memory_format(self.model, torch.channels_last)
            self.optimizer = optim.Adam(self.model.parameters(), config['lr'], betas=(0.9,0.95), weight_decay=config['weight_decay'], fused=True)
            self.optimizer = prepare_optimizer(self.optimizer)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), config['lr'], betas=(0.9,0.95), weight_decay=config['weight_decay'])
            
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=config['lr_factor'], patience=config['lr_patience'], min_lr=5e-9)
        # self.cycler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=75e-4, total_steps=total_steps_cyc)
        self.min_val_loss = 4e6
        self.best_epoch = 0
        self.metrics = []
        #self.y_pred, self.y_true = torch.tensor([]).cuda(), torch.tensor([]).cuda()
        self.y_pred, self.y_true = [], []

    def train(self, config):
        self.setup(config)
        self.epoch = 0
        while self.curr_count_to_patience < self.patience:
            if self.regulaizer:
                self.train_chunk(train.get_dataset_shard('train'), self.regulaizer)
            else:
                self.train_chunk(train.get_dataset_shard('train'))
            
            stats = self.evaluate_chunk()
            self.metrics.append(stats)

            self.epoch += 1
            
            self.early_stopping()
            self.save_checkpoint()
            if self.curr_count_to_patience == self.patience:
                self.best_epoch = self.epoch - self.patience
                

    def train_chunk(self, loader, secondary_reg = lambda x, y: x):
        self.model.train()
        running_loss = 0.0
        #check iter output/form
        for i, batch in enumerate(loader.iter_torch_batches(batch_size=self.batch_size, prefetch_batches=10)):
            x = batch['data']
            y = batch['labels']
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss = secondary_reg(loss, self.model.parameters())
            loss.backward()
            self.optimizer.step()
            # self.cycler.step()
            running_loss += (loss.detach().item() - running_loss)/(i+1)
        return running_loss
    
    def decode(self, preds, top):
        if top == 1:
            out = torch.argmax(preds.softmax(dim=1), dim=1)
        else:
            out = torch.argsort(preds, dim=1)[:, -(top-1):]
        return out.float()
    
    def get_metrics(self, loader):
        y_true, y_pred = torch.tensor([]), torch.tensor([])
        running_acc = 0
        running_loss = 0.0
        #check iter output/form
        for i, batch in enumerate(loader.iter_torch_batches(batch_size=self.batch_size, prefetch_batches=10)):
            x = batch['data']
            y = batch['labels']
            output = self.model(x) 
            predicted = self.decode(output, 1)
            running_loss += (self.criterion(output, y).detach().item() - running_loss)/(i+1)
            y = y.argmax(1)
            y_pred = torch.cat((y_pred, predicted.cpu()), dim=0)
            y_true = torch.cat((y_true, y.cpu()), dim=0)
            running_acc += (predicted == y).sum().item() #only for top-1 acc, # for i in range(y.shape[0]): top-5 acc correct += (y[i] in predicted[i])
        running_acc = running_acc/len(y_true)
        return running_acc, running_loss, y_pred, y_true

    def evaluate_chunk(self):
        """Evaluate the `model` on the train and validation set."""
        self.model.eval()
        with torch.inference_mode():
            train_acc, train_loss, _, _ = self.get_metrics(train.get_dataset_shard('train'))
            val_acc, val_loss, _, _ = self.get_metrics(train.get_dataset_shard('val'))
        
        self.scheduler.step(val_loss)
        
        stats = dict(train_loss=train_loss,train_accuracy=train_acc,val_loss=val_loss,val_accuracy=val_acc)
        return stats
    
    def early_stopping(self):
        """Calculate new patience and validation loss.

        Increment curr_patience by one if new loss is not less than global_min_loss
        Otherwise, update global_min_loss with the current val loss, and reset curr_count_to_patience to 0
        Increment curr_patience by one if new loss is not less than global_min_loss
        Otherwise, update global_min_loss with the current val loss, and reset curr_count_to_patience to 0

        Returns: new values of curr_patience and global_min_loss
        """
        if abs(self.metrics[-1]['val_loss'] - self.min_val_loss) <= 1e-4:
            self.curr_count_to_patience += 1
        else:
            self.min_val_loss = self.metrics[-1]['val_loss']
            self.curr_count_to_patience = 0
            if self.rank==0:
                print(f"New Val Loss: {self.min_val_loss}")

    def save_checkpoint(self):
        """Save a checkpoint file to `checkpoint_dir`."""
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = TorchCheckpoint.from_directory(temp_checkpoint_dir)
            if isinstance(self.model, FSDP) and self.model.sharding_strategy != ShardingStrategy.NO_SHARD:
                rank = train.get_context().get_world_rank()
                torch.save((self.model.module.state_dict(),self.optimizer.state_dict(), self.scheduler.state_dict()),# self.cycler.state_dict()),
                           os.path.join(temp_checkpoint_dir, f"model-rank={rank}.pt"),
                           )
                train.report(metrics={'val_loss':self.min_val_loss,'mean_accuracy':self.metrics[-1]['val_accuracy']}, checkpoint=checkpoint)
            else:
                try:
                    if train.get_context().get_world_rank() == 0:
                        torch.save((self.model.module.state_dict(),self.optimizer.state_dict(), self.scheduler.state_dict()),# self.cycler.state_dict()),
                            os.path.join(temp_checkpoint_dir, "model.pt"))
                        train.report(metrics={'val_loss':self.min_val_loss,'mean_accuracy':self.metrics[-1]['val_accuracy']}, checkpoint=checkpoint)
                except Exception as e:
                    print("no session active / distributed process")
                    torch.save((self.model.state_dict(),self.optimizer.state_dict(), self.scheduler.state_dict()),# self.cycler.state_dict()),
                            os.path.join(temp_checkpoint_dir, "model.pt"))
                    train.report(metrics={'val_loss':self.min_val_loss,'mean_accuracy':self.metrics[-1]['val_accuracy']}, checkpoint=checkpoint)
                

    def restore_model(self, best_or_restart, checkpoint):
        """Restore model from checkpoint if it exists."""
        
        filename = os.path.join(checkpoint.to_directory(), "model.pt")
        
        print(f"Loading from checkpoint {filename}.")

        model_dict, opt_dict, scheduler_dict = torch.load(filename) #, cycler_dict

        try:
            self.model.module.load_state_dict(model_dict)
            if best_or_restart == "restart":
                self.optimizer.load_state_dict(opt_dict)
                self.scheduler.load_state_dict(scheduler_dict)
                # self.cycler.load_state_dict(cycler_dict)
                # train.report(self.metrics, checkpoint=checkpoint)
            # print(f"=> Successfully restored checkpoint (trained for {checkpoint['epoch']} epochs)")
        except Exception as e:
            print(f"=> Checkpoint not successfully restored, Exception: {e}")
            raise


def clear_checkpoint(checkpoint_dir):
    """Remove checkpoints in `checkpoint_dir`."""
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith((".pt",'.pkl'))]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")