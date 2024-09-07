#meg_ray.py
# import argparse
import torch, os, sys#, pdb, 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pandas as pd
import numpy as np
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
# from torch.distributed.fsdp.wrap import (
#     size_based_auto_wrap_policy,
#     enable_wrap,
#     wrap,
# )
from typing import Dict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt#, tqdm
import seaborn as sns

# from filelock import FileLock
import ray
from ray import tune, train, data
from ray.train import FailureConfig, RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB, MedianStoppingRule
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner
# from src.models import D_MI_WaveletCNN
from src.common import config
from src.common import ray_data_from_numpy
from training_ray import Trainer, clear_checkpoint

from torchvision.models import ResNet

__dirname__ = "/src/MotorMovement"

def set_seed():
    # torch.cuda.manual_seed_all(448)
    torch.manual_seed(448)
    np.random.seed(448)

def split_channels(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    batch['data'] = batch['data'].reshape(-1, 3, 3, 160)
    return batch

def main(retrain, rank):
    set_seed()
    
    if not retrain:
        clear_checkpoint(config(__dirname__,"resnet18-hyper.checkpoint"))
        ray.init(object_store_memory=2*1024**3)
    else:
        ray.init()
        if train.get_checkpoint(): # FINISH SETTING UP RETRAIN
            trainer.restore_model('restart',train.get_checkpoint())

    #TODO: optimize loadingS
    eeg_data = np.load("."+__dirname__+'/data/X_eeg.npy')
    labels = np.load("."+__dirname__+'/data/y_labels.npy') #
    eeg_data = eeg_data[0:len(eeg_data)//8]
    labels = labels[0:len(labels)//8]
    dataset = ray_data_from_numpy(eeg_data, labels).map_batches(split_channels,batch_format="numpy")
    # dataset = data.range_tensor(1000,shape=(281,271))
    del eeg_data
    del labels
    train_dataset, test_dataset = dataset.train_test_split(0.2)
    del dataset
    train_dataset, val_dataset = train_dataset.train_test_split(0.1875)
    # pdb.set_trace()
    ctx = data.DataContext.get_current()
    ctx.execution_options.preserve_order = True
    #lab: 0000
    #lab2:
    regulaizer = lambda loss, params: loss + 3e-4 * sum(p.abs().sum() for p in params) #tune the 3e-4?
    
    #Trainer class, boiler plate code
    trainer = Trainer()
    
    #Distributed Training Strategy Args 
    sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD
    amp = MixedPrecision(
        param_dtype=torch.half
        , reduce_dtype=torch.half
        , buffer_dtype=torch.half
    )
    # parallel = {
    #             "sharding_strategy":sharding_strategy,
    #             "mixed_precision":amp
    #         }
    parallel = {
        'sharding_strategy':"CPU"
    }
    config_params = {
        "train_loop_config":{
            "dataset_size":train_dataset.count(),
            "model_class": ResNet,
            "batch_size":64,
            # "hparams": {'architecture':tune.choice(['var-cnn','lf-cnn']), 'n_classes':2},
            "lr": tune.loguniform(1e-5,5e-3),
            "lr_factor":tune.uniform(0.1,0.4), 
            "lr_patience":3,
            # "params": dict(dropout = tune.choice([0.1,0.25,0.375,0.5]), filter_length = tune.choice([3,5,7,9,11]),
            #                n_ls = tune.choice([32,16,64]), nonlin_hid = tune.choice([nn.ReLU,nn.LeakyReLU]),
            #                nonlin_in = tune.choice([nn.Identity,nn.LeakyReLU]), nonlin_out = tune.choice([nn.Identity,nn.LeakyReLU]),
            #                pooling = 2, stride = 1
            #                ),
            "patience": 5,
            "parallel_setup":parallel,
            "weight_decay": tune.loguniform(1e-4,1e-2)#,    
            # 'regulaizer':regulaizer
        }
    }
    
    class CustomStopper(tune.Stopper):
        def __init__(self):
            self.should_stop = False

        def __call__(self, trial_id, result):
            max_iter = 75
            if not self.should_stop and result["mean_accuracy"] > 0.8:
                self.should_stop = True
            return self.should_stop or result["training_iteration"] >= max_iter

        def stop_all(self):
            return self.should_stop

    stopper = CustomStopper()

    #Trainable for Ray Tune
    ray_trainer = TorchTrainer(
        train_loop_per_worker=trainer.train
        # , scaling_config=ScalingConfig(num_workers=1,use_gpu=True,resources_per_worker={"GPU": 1})
        , datasets={"train": train_dataset,"val":val_dataset}
    )
    
    #Hyperparam Search Algo: https://arxiv.org/abs/1807.01774
    # schedule = HyperBandForBOHB(
    #     time_attr="training_iteration"
    #     , metric='val_loss'
    #     , mode='min'
    #     # , reduction_factor=1#assume 2 gpus, 0.5 gpu per trial
    #     , stop_last_trials=False
    # )
    # searchBoHB = TuneBOHB(metric='val_loss', mode='min')
    
    
    # schedule = ASHAScheduler(
    #     time_attr="training_iteration"
    #     , metric='val_loss'
    #     , mode='min'
    # )
    # hebo = HEBOSearch(metric="val_loss", mode="min")
    
    schedule = MedianStoppingRule(
        time_attr="training_iteration"
        , metric='val_loss'
        , mode='min'
        # , reduction_factor=1#assume 2 gpus, 0.5 gpu per trial
        )
    search = BayesOptSearch(
        metric="val_loss", mode="min"
    )
    tuner = Tuner(
        ray_trainer 
        , run_config=RunConfig(
            name="bohb_exp"
            , verbose=False
            , failure_config=FailureConfig(
                fail_fast=True
            )
            , storage_path=config(__dirname__,"resnet18-hyper.checkpoint")
            , checkpoint_config=CheckpointConfig(1,checkpoint_score_attribute="mean_accuracy")
            , stop=stopper
        )
        , tune_config=TuneConfig(
            scheduler=schedule
            , search_alg=search
            , num_samples=5
            , reuse_actors=True
            #, max_concurrent_trials=4
        )
        , param_space=config_params
    )
    
    results = tuner.fit()
    print(results)
    # best_result = results.get_best_result("val_loss", "min")
    # trainer.restore_model('best',best_result.get_best_checkpoint('val_loss','min'))
    
    # test_acc, test_loss, test_pred, test_true = trainer.get_metrics(test_dataset)
    # print("Stats at best epoch: ", pd.DataFrame(trainer.metrics[-1], index=[0]))
    # print(f"Global min val loss {trainer.min_val_loss}")
    # print(f"test_acc {test_acc}, test_loss {test_loss}")
    
    # val_cm = confusion_matrix(trainer.y_pred.cpu(), trainer.y_true.cpu(), normalize='pred')
    # test_cm = confusion_matrix(test_true.cpu(), test_pred.cpu(), normalize='pred')
    # cmap=plt.cm.get_cmap("viridis")
    # fig, axes = plt.subplots(1, 2, figsize=(30, 7))
    # plt.suptitle("Confusion Matricies")
    # sns.heatmap(val_cm, annot=True,ax=axes[0],cmap=cmap)
    # sns.heatmap(test_cm, annot=True,ax=axes[1],cmap=cmap)
    # fig.savefig(f"MEG Test and Val CM, Post Hyper.png")
    
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