#training.py
# import argparse
import torch#, os, pdb
import numpy as np
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
# import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
# from torch.distributed.fsdp import (
#     FullyShardedDataParallel as FSDP,
#     MixedPrecision,
#     ShardingStrategy,
# )
from torch.distributed.elastic.multiprocessing.errors import record
# from torch.distributed.fsdp.wrap import (
#     size_based_auto_wrap_policy,
#     enable_wrap,
#     wrap,
# )
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt, tqdm
import seaborn as sns

from filelock import FileLock
import ray
from ray import tune, train
from ray.train import FailureConfig, RunConfig, ScalingConfig, CheckpointConfig, DataConfig
from ray.train.torch import TorchTrainer, TorchConfig
from ray.tune.schedulers.pb2 import PB2
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner

from utils import config
from training_ray import Trainer, clear_checkpoint
from models.MEGDecoder import MEGDecoder
from dataset import HDF5Dataset


def setup(local_rank):
    # dist.init_process_group("nccl")
    # torch.cuda.set_device(local_rank)
    torch.cuda.manual_seed_all(448)
    torch.manual_seed(448)
    np.random.seed(448)

# class TuneTrainable(tune.Trainable):
#     def setup(self, config):
#         # self.trainer = Trainer()
#         pass


@record
def main(retrain, local_rank, rank):
    setup(local_rank)
    #if you need local_rank device= torch.cuda.current_device()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/MEG_data/'
    # print("device: ",device)

    #args eventaully use yaml file instead
    hyper_params = dict(architecture = 'var-cnn', n_classes = 2)
    
    params = dict(n_ls = 32, dropout = 0.5, filter_length = 7,
                  nonlin_in = nn.Identity, nonlin_hid = nn.ReLU,
                  nonlin_out = nn.Identity, pooling = 2, stride = 1)
    # pdb.set_trace()
    
    dataset = HDF5Dataset(fpath+"meg_data.hdf5",('meg_data','labels'))
    
    tr_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, shuffle=True)
    tr_idx, val_idx = train_test_split(tr_idx, test_size=0.1875, shuffle=True) #TODO: don't shuffle cause fsdp/ddp + ray does already?
    
    train_dataset = Subset(dataset, tr_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
 
    #args eventaully use yaml file instead
    best_epoch = 0
    patience = 13 #tuneable
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
    model = model.to(device)
    # FSDP config
    # sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD
    # amp = MixedPrecision(
    #     param_dtype=torch.half,
    #     reduce_dtype=torch.half,
    #     buffer_dtype=torch.half
    #     )
    # model = FSDP(model, 
    #              device_id=torch.cuda.current_device(),
    #              sharding_strategy=sharding_strategy,
    #              mixed_precision=amp
    #              )
    
    if torch.cuda.is_available():
        model = nn.utils.convert_conv2d_weight_memory_format(model, torch.channels_last)

    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5, lr=0.0005, betas=(0.9,0.95)) #Tune
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=config['lr_factor'], patience=config['lr_patience'],min_lr=5e-9) #Tune
    
    criterion = CrossEntropyLoss()
    trainer = Trainer(train_args, model=model, optimizer=optimizer,
                         criterion=criterion, scheduler=scheduler)
    
    config_params = {"parallel_setup":{},"lr_patience":4,"lr_factor":0.4,"lr":4e-4,"batch_size":32}
    
    ray_trainer = TorchTrainer(
        train_loop_per_worker=trainer.train,
        train_loop_config=config_params,
        scaling_config=ScalingConfig(num_workers=1,use_gpu=True,resources_per_worker={"GPU": 0.5}),
        run_config=RunConfig(checkpoint_config=CheckpointConfig(1,checkpoint_score_attribute="mean_accuracy"),stop=stopper),
        datasets={"train": train_dataset,"val":val_dataset},
    )
    
    ray.init(_temp_dir=config("MEG-Hyper.store"))
    
    perturbation_interval = 5
    pbt = PB2(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        hyperparam_bounds={
            # hyperparameter bounds.
            "lr": [0.0001, 0.0275],
        },
    )
    class CustomStopper(tune.Stopper):
        def __init__(self):
            self.should_stop = False

        def __call__(self, trial_id, result):
            max_iter = 50
            if not self.should_stop and result["mean_accuracy"] > 0.8:
                self.should_stop = True
            return self.should_stop or result["training_iteration"] >= max_iter

        def stop_all(self):
            return self.should_stop

    stopper = CustomStopper()
    tuner = tune.Tuner(
        ray_trainer,
        run_config=train.RunConfig(
            name="pbt_test",
            verbose=False,
            failure_config=train.FailureConfig(
                fail_fast=True,
            ),
            storage_path=config("MEG-Hyper.checkpoint")
        ),
        tune_config=tune.TuneConfig(
            scheduler=pbt,
            metric="mean_accuracy",
            mode="max",
            num_samples=8,
            reuse_actors=True,
        ),
        param_space={
            "lr": 0.0001,
            # note: this parameter is perturbed but has no effect on
            # the model training in this example
            "some_other_factor": 1,
            # This parameter is not perturbed and is used to determine
            # checkpoint frequency. We set checkpoints and perturbations
            # to happen at the same frequency.
            "checkpoint_interval": perturbation_interval,
        },
    )
    if retrain:
        if train.get_checkpoint(): # FINISH SETTING UP RETRAIN
            trainer.restore_model('restart',train.get_checkpoint())
    else:
        clear_checkpoint(config("MEG-Hyper.checkpoint"))
    results = tuner.fit()
    best_result = results.get_best_result("val_loss", "min")
    trainer.restore_model('best',best_result.get_best_checkpoint('val_loss','min'))

    
    te_samp = DistributedSampler(test_dataset, rank=rank, shuffle=False)
    test_kwargs = {'batch_size': 32, 'sampler': te_samp,'num_workers': 2,
                   'pin_memory': True, 'shuffle': False}
    test_loader = DataLoader(test_dataset, **test_kwargs)
    
    test_acc, test_loss, test_pred, test_true = trainer.get_metrics(test_loader)
    
    
    import pandas as pd
    print("Stats at best epoch: ", pd.DataFrame(trainer.metrics[-1], index=[0]))
    print(f"Global min val loss {min_loss}")
    print(f"test_acc {test_acc}, test_loss {test_loss}")
    
    val_cm = confusion_matrix(trainer.y_pred.cpu(), trainer.y_true.cpu(), normalize='pred')
    test_cm = confusion_matrix(test_true.cpu(), test_pred.cpu(), normalize='pred')
    cmap=plt.cm.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(30, 7))
    plt.suptitle("Confusion Matricies")
    sns.heatmap(val_cm, annot=True,ax=axes[0],cmap=cmap)
    sns.heatmap(test_cm, annot=True,ax=axes[1],cmap=cmap)
    fig.savefig(f"MEG Test and Val CM, Post Hyper.png")
    
if __name__ == "__main__":
    retrain = False
    # parser = argparse.ArgumentParser()
    
    # dont need rank and stuff anymore?
    # local_rank = int(os.environ["LOCAL_RANK"])
    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ['WORLD_SIZE'])
    # print(f"rank={rank}, world_size={world_size}\n")
    
    # set up other necessary arguments
    # parser.add_argument(
    #     "--address", required=False, type=str, help="The address to use for Redis."
    # )
    # parser.add_argument(
    #     "--num-workers",
    #     "-n",
    #     type=int,
    #     default=2,
    #     help="Sets number of workers for training.",
    # )
        
    main(retrain, 0,0)#local_rank, rank)