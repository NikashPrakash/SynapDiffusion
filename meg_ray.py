#meg_ray.py
# import argparse
import torch, pdb#, os
import numpy as np
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
# import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
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
import matplotlib.pyplot as plt, tqdm
import seaborn as sns

# from filelock import FileLock
import ray
from ray import tune, train
from ray.data import from_torch
from ray.train import FailureConfig, RunConfig, ScalingConfig, CheckpointConfig, DataConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner
from utils import config
from training_ray import Trainer, clear_checkpoint
from models.MEGDecoder import MEGDecoder
from dataset import HDF5Dataset


def set_seed():
    torch.cuda.manual_seed_all(448)
    torch.manual_seed(448)
    np.random.seed(448)

# class TuneTrainable(tune.Trainable):
#     def setup(self, config):
#         # self.trainer = Trainer()
#         pass


@record
def main(retrain, rank):
    # set_seed()
    #if you need local_rank device= torch.cuda.current_device()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("device: ",device)
    # pdb.set_trace()
    
    ray.init()

    fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/MEG_data/'
    dataset = HDF5Dataset(fpath+"meg_data.hdf5",('meg_data','labels'))
    
    tr_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, shuffle=True)
    tr_idx, val_idx = train_test_split(tr_idx, test_size=0.1875, shuffle=True) #TODO: don't shuffle cause fsdp/ddp + ray does already?
    
    #TODO: Optimize data storage and loading
    train_dataset = from_torch(Subset(dataset, tr_idx))
    breakpoint()
    val_dataset = from_torch(Subset(dataset, val_idx))
    test_dataset = from_torch(Subset(dataset, test_idx))
    
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = True
    #args eventaully use yaml file instead
    regulaizer = lambda loss, params: loss + 3e-4 * sum(p.abs().sum() for p in params) #tune the 3e-4?
    
    #Trainer class, boiler plate code
    trainer = Trainer()
    
    #Distributed Training Strategy Args 
    sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD
    amp = MixedPrecision(
        param_dtype=torch.half,
        reduce_dtype=torch.half,
        buffer_dtype=torch.half
    )
    config_params = {
        "train_loop_config":{
            "dataset_size":len(train_dataset),
            "model_class": MEGDecoder,
            "hparams": {'architecture':tune.choice(['var-cnn','lf-cnn']), 'n_classes':2},
            "params": dict(n_ls = tune.choice([32,16,64]), dropout = tune.choice([0.1,0.25,0.375,0.5]), 
                           filter_length = 7, nonlin_in = tune.choice([nn.Identity,nn.LeakyReLU]), 
                           nonlin_hid = tune.choice([nn.ReLU,nn.LeakyReLU]), pooling = 2, stride = 1,
                           nonlin_out = tune.choice([nn.Identity,nn.LeakyReLU])),
            "patience": tune.choice([5,8,11]),
            "parallel_setup":{
                "sharding_strategy":sharding_strategy,
                "mixed_precision":amp
            },
            "weight_decay":tune.uniform(1e-4,1e-2),
            "lr_patience":tune.randint(1,6),
            "lr_factor":tune.choice([0.1,0.25,0.3,0.4]), 
            "lr": tune.uniform(1e-5,5e-3),
            "batch_size":tune.choice([32,64,128]),
            'regulaizer':regulaizer
        }
    }
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

    #Trainable for Ray Tune
    ray_trainer = TorchTrainer(
        train_loop_per_worker=trainer.train,
        scaling_config=ScalingConfig(num_workers=1,use_gpu=True,resources_per_worker={"GPU": 0.5}),
        run_config=RunConfig(checkpoint_config=CheckpointConfig(1,checkpoint_score_attribute="mean_accuracy"),stop=stopper),
        datasets={"train": train_dataset,"val":val_dataset},
    )
    
    #Temp Results Dir used by Ray
    
    #Hyperparam Search Algo: https://arxiv.org/abs/1807.01774
    bohb_hb = HyperBandForBOHB(
        time_attr="training_iteration",
        metric='val_loss',
        mode='min',
        reduction_factor=4,#assume 2 gpus, 0.5 gpu per trial
        stop_last_trials=False
    )
    
    searchBoHB = TuneBOHB(config_params, metric='val_loss', mode='min')
    
    tuner = Tuner(
        ray_trainer,
        run_config=RunConfig(
            name="bohb_exp",
            verbose=False,
            failure_config=FailureConfig(
                fail_fast=True,
            ),
            storage_path=config("MEG-Hyper.checkpoint")
        ),
        tune_config=TuneConfig(
            scheduler=bohb_hb,
            search_alg=searchBoHB,
            num_samples=40,
            reuse_actors=True,
            max_concurrent_trials=4
        ),
        param_space=config_params
    )

    if retrain:
        if train.get_checkpoint(): # FINISH SETTING UP RETRAIN
            trainer.restore_model('restart',train.get_checkpoint())
    else:
        clear_checkpoint(config("MEG-Hyper.checkpoint"))
    results = tuner.fit()
    best_result = results.get_best_result("val_loss", "min")
    trainer.restore_model('best',best_result.get_best_checkpoint('val_loss','min'))

    
    # te_samp = DistributedSampler(test_dataset, rank=rank, shuffle=False)
    # test_kwargs = {'batch_size': 32, 'sampler': te_samp,'num_workers': 2,
    #                'pin_memory': True, 'shuffle': False}
    # test_loader = DataLoader(test_dataset, **test_kwargs)
    test_acc, test_loss, test_pred, test_true = trainer.get_metrics(test_dataset.iterator())
    
    
    import pandas as pd
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
        
    main(retrain, 0)#local_rank, rank)