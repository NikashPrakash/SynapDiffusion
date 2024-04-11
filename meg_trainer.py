#training.py
import torch, os, pdb
import numpy as np
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.elastic.multiprocessing.errors import record
# from torch.distributed.fsdp.wrap import (
#     size_based_auto_wrap_policy,
#     enable_wrap,
#     wrap,
# )
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import config
from training import Trainer, clear_checkpoint
from models.model import *
from dataset import DDPDataset


def setup(local_rank):
    # dist.init_process_group("nccl")
    # torch.cuda.set_device(local_rank)
    # torch.cuda.manual_seed_all(448)
    torch.manual_seed(448)
    np.random.seed(448)
    
def cleanup():
    dist.destroy_process_group()

class MEGTrainer(Trainer):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.patience = args[0]
        self.curr_count_to_patience = args[1]
        self.checkpoint_path = args[2]
        self.min_loss = args[3]
        self.best_epoch = args[4]
        self.regulaizer = args[5]
        self.is_regulaizer = True
    """def hyper_search(self, tr_loader, val_loader): 
        params = {
            # TODO: Use other hyperparams if needed
            'lr': [5e-6, 5e-5, 1e-4, 1e-3],
            'module__drop_rate': [0.1,0.3,0.5,0.75],
            'optimizer__weight_decay':[1e-5, 1e-6, 1e-3, 0]
        }

        _vals = np.meshgrid(params['lr'],params['module__drop_rate'],params['optimizer__weight_decay'])
        param_set = np.array([_vals[0].ravel(), _vals[1].ravel(),_vals[2].ravel()]).T
        np.random.seed(42)
        param_set = param_set[np.random.randint(0,65,size=(15))]

        best_performance = float('inf')
        for lr, drop_rate, w_d in param_set:
            stats = []
            model = 1# TODO: Initalize model
            criterion=nn.CrossEntropyLoss() # TODO: use chosen loss models
            optimizer=torch.optim.Adam(model.parameters(),weight_decay=w_d,lr=lr) # TODO: This optimizer or another?

            self.evaluate_chunk(tr_loader, val_loader, model, criterion, stats)
            
            best_epoch = self.train(model, optimizer, criterion, tr_loader, val_loader, stats, 0, True)
            model, stats = restore_best(model, best_epoch, config('.checkpoint'))
            clear_checkpoint(config('.checkpoint'))
            
            if stats[-1][1] < best_performance:
                best_performance = stats[-1][1]
                best_model = model
                best_stats = stats
                params = lr,drop_rate,w_d
                self.save_checkpoint(model, optimizer, best_epoch, config(".checkpoint"), stats, params)
        return best_performance, best_stats, best_model  """

@record
def main(retrain, local_rank, rank):
    setup(local_rank)
    #if you need local_rank device= torch.cuda.current_device()
    
    fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/MEG_data/'
    # print("device: ",device)

    #args use yaml file
    hyper_params = dict(architecture = 'var-cnn', n_classes = 2)
    params = dict(n_ls = 32, learn_rate = 3e-4, dropout = 0.5,
                  nonlin_in = nn.Identity, nonlin_hid = nn.ReLU,
                  nonlin_out = nn.Identity, filter_length = 7,
                  pooling = 2, stride = 1)
    # pdb.set_trace()
    
    dataset = DDPDataset(fpath+"meg_data.hdf5",('meg_data','labels'))

    tr_idx, test_idx = train_test_split(range(len(dataset)),test_size=0.2, shuffle=True)
    tr_idx, val_idx = train_test_split(tr_idx, test_size=0.1875, shuffle=True)
    
    train_dataset = Subset(dataset, tr_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    gen = torch.Generator()
    gen.manual_seed(448)
    # tr_samp = DistributedSampler(train_dataset, rank=rank, shuffle=False)
    # val_samp =  DistributedSampler(val_dataset, rank=rank, shuffle=False)
    # te_samp = DistributedSampler(test_dataset, rank=rank, shuffle=False)
    tr_samp = RandomSampler(train_dataset, generator=gen)
    val_samp =  RandomSampler(val_dataset, generator=gen) 
    te_samp = RandomSampler(test_dataset, generator=gen)

    train_kwargs = {'batch_size': 32, 'sampler': tr_samp}
    val_kwargs = {'batch_size': 32, 'sampler': val_samp}
    test_kwargs = {'batch_size': 32, 'sampler':te_samp }
    # cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False}
    # train_kwargs.update(cuda_kwargs)
    # val_kwargs.update(cuda_kwargs)
    # test_kwargs.update(cuda_kwargs)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)
 
    # args use yaml file
    best_epoch = 0
    patience = 7
    curr_count_to_patience = 0
    min_loss = 1000
    regulaizer = lambda loss, params: loss + 3e-4 * sum(p.abs().sum() for p in params)

    train_args = (patience, 
                  curr_count_to_patience,
                  config("MEG-Encoder.checkpoint"),
                  min_loss, 
                  best_epoch,
                  regulaizer
                  )
    
    model = MEGDecoder(hyper_params, params)

    sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD
    amp = MixedPrecision(
        param_dtype=torch.half,
        reduce_dtype=torch.half,
        buffer_dtype=torch.half
        )
    # model = FSDP(model, 
    #              device_id=torch.cuda.current_device(),
    #              sharding_strategy=sharding_strategy,
    #              mixed_precision=amp
    #              )
    model = nn.utils.convert_conv2d_weight_memory_format(model, torch.channels_last)

    optimizer = optim.Adam(model.parameters(), weight_decay=.03, lr=0.003) #Tune
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.4,patience=4,min_lr=5e-9) #Tune
    
    criterion = CrossEntropyLoss()
    trainer = MEGTrainer(train_args, model=model, tr_loader=train_loader, optimizer=optimizer,
                         criterion=criterion, val_loader=val_loader, scheduler=scheduler)
    if rank == 0:
        if retrain:
            trainer.restore_model('restart')
            # dist.barrier()
        else:
            clear_checkpoint(config("MEG-Encoder.checkpoint"))

    epoch = 0
    trainer.train(epoch)
    if rank == 0:
        model, bestStats = trainer.restore_model('best')
        # dist.barrier()

    test_acc, test_loss, test_pred, test_true =  trainer._get_metrics(test_loader)
    # dist.barrier()
    
    # cleanup()
    
    import pandas as pd
    print("Stats at best epoch: ", pd.DataFrame(bestStats[-1]))
    print(f"Global min val loss {min_loss}")
    print(f"test_acc {test_acc}, test_loss {test_loss}")
    
    val_cm = confusion_matrix(trainer.y_pred,trainer.y_true,normalize='pred')
    test_cm = confusion_matrix(test_true, test_pred,normalize='pred')
    cmap=plt.cm.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(30, 7))
    plt.suptitle("Confusion Matricies")
    sns.heatmap(val_cm, annot=True,ax=axes[0],cmap=cmap)
    sns.heatmap(test_cm, annot=True,ax=axes[1],cmap=cmap)
    fig.savefig("Test and Val CM's.png")
    
if __name__ == "__main__":
    retrain = False
    # local_rank = int(os.environ["LOCAL_RANK"])
    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ['WORLD_SIZE'])
    # print(f"rank={rank}, world_size={world_size}\n")
    main(retrain, 0,0)#local_rank, rank)


"""
#EXTRAS
# for i in range(numChunks):  #TESTING LOOP - PER CHUNK: (train function should be run on each CHUNK)
    # trdat = {}
    # valdat = {}
        # with torch.no_grad():
        #     model.eval()
        #     eegdata = torch.load('/EEG_data/chunks/sub-0' + str(i)) #TODO fix case where participant number >9 lol
        #     numBatches = eegdata.shape[0]
        #     idxs = tr_idx[np.where((tr_idx > currBatch) & (tr_idx < (currBatch + numBatches)))] #TODO check bounds
        #     tr_idx, val_idx = train_test_split(idxs,shuffle=True,test_size=0.2)
        #     test["data"]  = eegdata[tr_idx - currBatch, :, :] #makes index wrt current chunk
        #     test["labels"]  = mapping[tr_idx, :, :] #TODO: Any necessary data transformations (if any)... 
    # temp.append(train(model, optimizer, criterion, trdat, valdat, stats))
    #    currBatch = currBatch + numBatches

    
#perf, stats, model = hyper_search(pt_models, train_loader, val_loader, device)
#print(f"hyperparam search best val loss: {perf}")

#utils.make_training_plot(np.array(stats))
#stats[-1] += list(_get_metrics(model, criterion, test_loader))
# np.save("best_model_stats.npy",np.array(stats))
# utils.make_training_plot(np.array(stats))
"""