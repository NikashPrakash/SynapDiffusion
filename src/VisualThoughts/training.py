#training.py
import torch, pdb
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    FullOptimStateDictConfig,
    StateDictType
)
import os, tqdm

class Trainer:
    def __init__(self, args, model: torch.nn.Module, tr_loader: DataLoader, val_loader: DataLoader, criterion,
                 optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau):
        self.rank = int(os.environ.get("RANK",0))
        self.patience = args[0]
        self.curr_count_to_patience = args[1]
        self.checkpoint_path = args[2]
        
        self.tr_loader = tr_loader
        self.val_loader = val_loader
        
        self.regulaizer = args[5]
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.scheduler = scheduler
        
        self.min_loss = args[3]
        self.best_epoch = args[4]
        self.metrics = []
        self.y_pred, self.y_true = torch.tensor([]), torch.tensor([])

    def train_chunk(self, inner_pbar: None, secondary_reg = lambda x, y: x):
        self.model.train()
        fsdp_loss = torch.zeros(2)#.cuda()
        for x, y in self.tr_loader:
            self.optimizer.zero_grad()
            # pdb.set_trace()
            # print(x, y)
            # y = y.to(torch.cuda.current_device())
            pred = self.model(x)
            # print(pred)
            loss = self.criterion(pred, y)
            loss = secondary_reg(loss, self.model.parameters())
            loss.backward()
            self.optimizer.step()
            fsdp_loss[0] += loss.item()
            fsdp_loss[1] += x.shape[0]
            if self.rank == 0:
                inner_pbar.update(1)
        if dist.is_initialized():
            dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
        tr_loss = fsdp_loss[0]/fsdp_loss[1]
        
        if self.rank == 0:
            inner_pbar.close()
            print(f"Train Epoch: \t{self.epoch}, Loss: \t{tr_loss:.4f}")
    
    def decode(self, preds, top):
        if top == 1:
            out = torch.argmax(preds.softmax(dim=1), dim=1)
        else:
            out = torch.argsort(preds, dim=1)[:, -(top-1):]
        return out.float()
    
    def _get_metrics(self, loader, inner_pbar_n, load_type):
        y_true, y_pred = torch.tensor([]),torch.tensor([])#.cuda(), torch.tensor([]).cuda()
        correct = 0
        running_loss = torch.zeros(2)#.cuda()
        if self.rank == 0:
            inner_pbar = tqdm.tqdm(
                range(len(loader)), colour="green", desc=inner_pbar_n
            )
        with torch.no_grad():
            for x, y in loader:
                output = self.model(x) 
                predicted = self.decode(output, 1)
                running_loss[1] += y.shape[0]
                running_loss[0] += self.criterion(output, y).item()
                
                y = y.argmax(1)
                correct += (predicted == y).sum().item() #only for top-1 acc, # for i in range(y.shape[0]): top-5 acc correct += (y[i] in predicted[i])
                y_pred = torch.cat((y_pred, predicted),dim=0) # TODO: is supposed to be arg?
                y_true = torch.cat((y_true, y),dim=0)
                if self.rank == 0:
                    inner_pbar.update(1)

        if dist.is_initialized():
            dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        loss = running_loss[0] / running_loss[1]
        
        if self.rank == 0:
            inner_pbar.close()
            print(f"{load_type} Loss: {loss:.4f}")
            
        v_acc = correct/running_loss[1]
        return v_acc, loss, y_pred, y_true

    def evaluate_chunk(self):
        """Evaluate the `model` on the train and validation set."""
        self.model.eval()

        train_acc, train_loss, _, _ = self._get_metrics(self.tr_loader, "Training Metrics", 'tr')
        val_acc, val_loss, y_pred, y_true = self._get_metrics(self.val_loader, 'Validation Epoch', 'val')
        
        self.scheduler.step(val_loss)
        
        stats = dict(train_loss=train_loss,train_accuracy=train_acc,val_loss=val_loss,val_accuracy=val_acc)
        return stats, y_pred, y_true
    
    def save_checkpoint(self, checkpoint_dir):
        """Save a checkpoint file to `checkpoint_dir`."""
        if dist.is_initialized():
            with FSDP.state_dict_type(self.model,
                                        StateDictType.FULL_STATE_DICT,
                                        FullStateDictConfig(True),
                                        FullOptimStateDictConfig()
                                        ):
                state = {
                    "epoch": self.epoch,
                    "state_dict": self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "stats": self.metrics
                }
        else:
            state = {
                    "epoch": self.epoch,
                    "state_dict": self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "stats": self.metrics
                }

        filename = os.path.join(checkpoint_dir, f"epoch={self.epoch}.checkpoint.pt")
        torch.save(state, filename)

    def early_stopping(self):
        """Calculate new patience and validation loss.

        Increment curr_patience by one if new loss is not less than global_min_loss
        Otherwise, update global_min_loss with the current val loss, and reset curr_count_to_patience to 0
        Increment curr_patience by one if new loss is not less than global_min_loss
        Otherwise, update global_min_loss with the current val loss, and reset curr_count_to_patience to 0

        Returns: new values of curr_patience and global_min_loss
        """
        if self.metrics[-1]['val_loss'] >= self.min_loss:
            self.curr_count_to_patience += 1
        else:
            self.min_loss = self.metrics[-1]['val_loss']
            self.curr_count_to_patience = 0
            if self.rank==0:
                print(f"New Val Loss: {self.min_loss}")

    def train(self, start_epoch):       
        self.epoch = start_epoch
        while self.curr_count_to_patience < self.patience:
            inner_pbar = None
            if self.rank == 0:
                inner_pbar = tqdm.tqdm(
                    range(len(self.tr_loader)), colour="blue", desc="r0 Training Epoch"
                )
            if dist.is_initialized():
                self.tr_loader.sampler.set_epoch(self.epoch)
                self.val_loader.sampler.set_epoch(self.epoch)
                dist.barrier()

            if self.regulaizer:
                self.train_chunk(inner_pbar, self.regulaizer)
            else:
                self.train_chunk(inner_pbar)
            
            stats, y_pred, y_true = self.evaluate_chunk()
            
            self.y_pred = torch.cat((self.y_pred,y_pred))
            self.y_true = torch.cat((self.y_true,y_true))
            self.metrics.append(stats)
            self.epoch += 1
            
            if self.rank == 0:
                self.save_checkpoint(self.checkpoint_path)
            
            self.early_stopping()
            if self.curr_count_to_patience == self.patience:
                self.best_epoch = self.epoch - self.patience
        return y_pred, y_true

    def restore_model(self, best_or_restart):
        """Restore model from checkpoint if it exists.

        Returns the model and the current epoch.
        """
        cp_files = os.listdir(self.checkpoint_path)
        if not cp_files:
            os.makedirs(self.checkpoint_path)
            return
        if best_or_restart == 'best':
            filename = self.checkpoint_path + f"epoch={self.best_epoch}.checkpoint.pt"
        else:
            filename = self.checkpoint_path + cp_files[-1]
        print(f"Loading from checkpoint {filename}.")

        checkpoint = torch.load(filename, self.map_location)

        try:
            self.model.load_state_dict(checkpoint['state_dict'])
            self.metrics = checkpoint['stats']
            self.epoch = checkpoint['epoch']
            if best_or_restart == "restart":
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> Successfully restored checkpoint (trained for {checkpoint['epoch']} epochs)")
        except:
            print("=> Checkpoint not successfully restored")
            raise
        
def clear_checkpoint(checkpoint_dir):
    """Remove checkpoints in `checkpoint_dir`."""
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")