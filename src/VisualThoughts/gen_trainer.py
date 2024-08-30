import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.distributed.elastic.multiprocessing.errors import record
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from src.common.utils import *
import seaborn as sns
from src.common.utils import config
from models.FusionModels import *
from common.dataset import CombDataset
import pandas as pd

from training import FusionTrainer, clear_checkpoint

def setup(local_rank):
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank) #TODO CHANGE WHEN RUNNING JOB
    torch.cuda.manual_seed_all(448)
    torch.manual_seed(448)
    np.random.seed(448)

def cleanup():
    dist.destroy_process_group()

# @record
def main(retrain, local_rank, rank):
    setup(local_rank)
    #if you need local_rank device= torch.cuda.current_device()
    
    fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Combined_data/'
    # print("device: ",device)

    # pdb.set_trace()
    
    dataset = CombDataset(fpath+"combined_data.hdf5", ('meg','eeg','labels'))
    # dataset.to_cuda_naive()
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tr_idx, test_idx = train_test_split(range(len(dataset)),test_size=0.2, shuffle=True)
    tr_idx, val_idx = train_test_split(tr_idx, test_size=0.1875, shuffle=True)
    
    train_dataset = Subset(dataset, tr_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    tr_samp = DistributedSampler(train_dataset, rank=rank, shuffle=False) #for distributed training with gpus
    va_samp =  DistributedSampler(val_dataset, rank=rank, shuffle=False)
    te_samp = DistributedSampler(test_dataset, rank=rank, shuffle=False)
    # tr_samp = SequentialSampler(train_dataset) #For cpu
    # va_samp =  SequentialSampler(val_dataset)  #For cpu
    # te_samp = SequentialSampler(test_dataset)  #For cpu 

    train_kwargs = {'batch_size': 32, 'sampler': tr_samp}
    val_kwargs = {'batch_size': 32, 'sampler': va_samp}
    test_kwargs = {'batch_size': 32, 'sampler': te_samp}
    cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    val_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)
 
    # args use yaml file
    best_epoch = 0
    patience = 2
    curr_count_to_patience = 0
    min_loss = 1000
    train_args = (patience, 
                  curr_count_to_patience,
                  config("Synaptic-Encoder.checkpoint"),
                  min_loss, 
                  best_epoch,
                  None
                  )
    
    sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD
    amp = MixedPrecision(
        param_dtype=torch.half,
        reduce_dtype=torch.half,
        buffer_dtype=torch.half
        )

    model = FusionClassification()#.to(device)
    model = nn.utils.convert_conv2d_weight_memory_format(model, torch.channels_last)

    model = FSDP(model,
                 device_id=torch.cuda.current_device(),
                 sharding_strategy=sharding_strategy,
                 mixed_precision=amp)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003) #Tune
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.4, patience=4, min_lr=5e-9) #Tune
    
    criterion = CrossEntropyLoss()
    trainer = FusionTrainer(train_args, model=model, tr_loader=train_loader, optimizer=optimizer,
                         criterion=criterion, val_loader=val_loader, scheduler=scheduler)
    if rank == 0:
        if retrain:
            trainer.restore_model('restart')
        else:
            clear_checkpoint(config("EEG-Encoder.checkpoint"))

    epoch = 0
    trainer.train(epoch)
    if rank == 0:
        trainer.restore_model('best')
    test_acc, test_loss, test_pred, test_true = trainer.get_metrics(test_loader, "Testing Metrics", 'te')
    
    # dist.barrier()
    # cleanup()
    
    print("Stats at best epoch: ", pd.DataFrame(trainer.metrics[-1], index=[0]))
    print(f"Global min val loss {trainer.min_loss}")
    print(f"test_acc {test_acc}, test_loss {test_loss}")
    
    val_cm = confusion_matrix(trainer.y_pred.cpu(), trainer.y_true.cpu(), normalize='pred')
    test_cm = confusion_matrix(test_true.cpu(), test_pred.cpu(), normalize='pred')
    cmap=plt.cm.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(30, 7))
    plt.suptitle("Confusion Matricies")
    sns.heatmap(val_cm, annot=True,ax=axes[0],cmap=cmap)
    sns.heatmap(test_cm, annot=True,ax=axes[1],cmap=cmap)
    fig.savefig("Fusion Test and Val CM's.png")

if __name__ == "__main__":
    retrain = False
    # local_rank = int(os.environ["LOCAL_RANK"])
    # rank = int(os.environ["RANK"])
    # print(f"rank={rank}, world_size={world_size}\n")
    rank = local_rank = 0
    main(retrain, local_rank, rank)