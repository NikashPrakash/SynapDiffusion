#training.py
import xarray,zarr
import mne
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import transformers, diffusers
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.distributed.elastic as dpe
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import utils
from utils import config
from model import *
import tensorflow as ts
from tensorflow import keras



def save_checkpoint(model, epoch, checkpoint_dir, stats, params=None):
    """Save a checkpoint file to `checkpoint_dir`."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": stats
    }
    if params:
        state['params'] = params
    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    torch.save(state, filename)

def restore_best(model, best_epoch, checkpoint_dir, pretrain=False):
    """Restore model from checkpoint if it exists.

    Returns the model and the current epoch.
    """
    try:
        cp_files = [
            file_
            for file_ in os.listdir(checkpoint_dir)
            if file_.startswith("epoch=") and file_.endswith(".checkpoint.pth.tar")
        ]
    except FileNotFoundError:
        cp_files = None
        os.makedirs(checkpoint_dir)
    if not cp_files:
        print("No saved model parameters found")
        return model, []

    filename = os.path.join(
        checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(best_epoch)
    )
    
    print("Loading from checkpoint {}?".format(filename))

    checkpoint = torch.load(filename)

    try:
        stats = checkpoint["stats"]
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise
    
    return model, stats

def clear_checkpoint(checkpoint_dir):
    """Remove checkpoints in `checkpoint_dir`."""
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")

def _get_metrics(model, criterion, loader):
    y_true, y_pred = [], []
    correct, total, true_posit, false_posit, false_neg = 0, 0, 0, 0, 0
    running_loss = []
    with torch.no_grad():
        for batch in loader:
            y = batch['labels']
            # input_ids = batch['input_ids']
            # attention_mask = batch['attention_mask']
            output = model() 
            predicted = output.argmax(1)
            
            y_true.append(y)
            y_pred.append(predicted)                
            total += y.size(0)
            correct += (predicted == y).sum().item()
            # TODO: Add calculations for metrics used, currently gets accuracy, recall, and precision
            predicted = predicted / y
            true_posit += torch.sum(predicted == 1).item()
            false_posit += torch.sum(predicted == float('inf')).item()
            false_neg += torch.sum(predicted == 0).item()
            running_loss.append(criterion(output, y).item())
            
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    loss = np.mean(running_loss)
    acc = correct / total
    recall = true_posit / (true_posit + false_neg)
    precision = true_posit / (true_posit + false_posit)
    return acc, loss, recall, precision

def evaluate_epoch(
    tr_loader,
    val_loader,
    model,
    criterion,
    stats
):
    """Evaluate the `model` on the train and validation set."""
    model.eval()

    train_acc, train_loss, train_rec, train_prec = _get_metrics(model, criterion, tr_loader)
    val_acc, val_loss, val_rec, val_prec = _get_metrics(model, criterion, val_loader)

    stats_at_epoch = [
        val_acc,
        val_loss,
        val_rec, 
        val_prec,
        train_acc,
        train_loss,
        train_rec, 
        train_prec
    ]

    stats.append(stats_at_epoch)


def hyper_search(pt_models, tr_loader, val_loader, device): 
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

        evaluate_epoch(tr_loader, val_loader, model, criterion, stats)
        
        best_epoch = train(model, optimizer, criterion, tr_loader, val_loader, stats, 0, True)
        model, stats = restore_best(model, best_epoch, config('.checkpoint'))
        clear_checkpoint(config('.checkpoint'))
        
        if stats[-1][1] < best_performance:
            best_performance = stats[-1][1]
            best_model = model
            best_stats = stats
            params = lr,drop_rate,w_d
            save_checkpoint(model, best_epoch, config(".checkpoint"), stats, params)
    return best_performance, best_stats, best_model



def train_epoch(model, train, optimizer,criterion, model_input_vars):
    '''
    params: 
        model_input_vars: List, names of all the input variables for a model (ex. EEG Encoder Model - EEG inp data)
    '''
    model.train()
    for batch in train:
        optimizer.zero_grad()
        y = batch['labels']
        # TODO: Use whatever necessary inputs for the models
        # y = F.one_hot(y, num_classes=2).float()
        # input_ids = batch['input_ids']
        # attention_mask = batch['attention_mask']
        pred = model()
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()


def train(model, optimizer, criterion, train_loader, val_loader, stats, start_epoch, cv):
    patience = 5
    curr_count_to_patience = 0
    global_min_loss = stats[0][1]

    epoch = start_epoch
    while curr_count_to_patience < patience:
        train_epoch(model,train_loader, optimizer, criterion)
        evaluate_epoch(
            train_loader,
            val_loader,
            model,
            criterion,
            stats
        )
        if cv:
            save_checkpoint(model, epoch + 1, config(".checkpoint"), stats)
        else:
            save_checkpoint(model, epoch + 1, config(".checkpoint"), stats)

        curr_count_to_patience, global_min_loss = early_stopping(
            stats, curr_count_to_patience, global_min_loss
        )
        
        epoch += 1
    return np.array(stats)[:,1].argmin()


def early_stopping(stats, curr_count_to_patience, global_min_loss):
    """Calculate new patience and validation loss.

    Increment curr_patience by one if new loss is not less than global_min_loss
    Otherwise, update global_min_loss with the current val loss, and reset curr_count_to_patience to 0
    Increment curr_patience by one if new loss is not less than global_min_loss
    Otherwise, update global_min_loss with the current val loss, and reset curr_count_to_patience to 0

    Returns: new values of curr_patience and global_min_loss
    """
    if stats[-1][1] >= global_min_loss:
      curr_count_to_patience += 1
    else:
      global_min_loss = stats[-1][1]
      curr_count_to_patience = 0
    return curr_count_to_patience, global_min_loss



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_train, y_train, x_test, y_test = 1,1,1,1 
    print("device: ",device)
    y_test = y_test.to(device)
        
    # for param in model.parameters(): #transfer learning, comment out for fine-tuning
    #     param.requires_grad = False    
    
    # imgGen_id = "prompthero/openjourney"
    # gen_pipe = diffusers.StableDiffusionPipeline(imgGen_id,torch_dtype=torch.float32).to(device)                                                              
    pt_models = {}#TODO: Add pretrained models to this
    tr_idx, val_idx = train_test_split(np.arange(y_train.size(0)),shuffle=True,test_size=0.1,stratify=y_train)
    x_train, x_val = x_train[tr_idx], x_train[val_idx]
    y_train, y_val = y_train[tr_idx], y_train[val_idx]
    
    train_inputs = 1 # TODO: Any necessary data transformations (if any)... 
    val_inputs = 1   # ^
    test_inputs = 1  # ^^
    
    train_inputs['labels'] = y_train
    train_loader = MultiSignalDataset(train_inputs.to(device))
    train_loader = DataLoader(train_loader, batch_size=16)
    
    val_inputs['labels'] = y_val
    val_loader = MultiSignalDataset(val_inputs.to(device))
    val_loader = DataLoader(val_loader, batch_size=16)
    
    test_inputs['labels'] = y_test
    test_loader = MultiSignalDataset(test_inputs.to(device))
    test_loader = DataLoader(test_loader,batch_size=16)
    
    
    perf, stats, model = hyper_search(pt_models, train_loader, val_loader, device)
    print(f"hyperparam search best val loss: {perf}")
    #  = StanceDetect(bert_model, 2, param["module__drop_rate"]).to(device)
    # model = StanceDetect(bert_model, 2, 0.5).to(device)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=param["lr"] , weight_decay=param["optimizer__weight_decay"])     
    
    # model, start_epoch, stats = restore_checkpoint(model, config("DistilBert_FineTune.checkpoint"), torch.cuda.is_available())
    # model.to(device)
    criterion = CrossEntropyLoss()
    
    # evaluate_epoch(#axes,
    #                train_loader, val_loader, test_loader, model, criterion, stats)
    
    # best_epoch = train(model, optimizer, criterion, train_loader, val_loader, stats, 0)
    # utils.save_dbert_training_plot()
    
    # model, stats = restore_best(model, best_epoch, config("DistilBert_FineTune.checkpoint"))
    
    # evaluate_epoch(#axes,
    #                train_loader, val_loader, model, criterion, stats, test_loader, include_test=True)
    
    utils.make_training_plot(np.array(stats))
    stats[-1] += list(_get_metrics(model, criterion, test_loader))
    # np.save("best_model_stats.npy",np.array(stats))
    print(pd.DataFrame(np.array(stats[-1]).reshape(3,4), columns=['Accuracy','Loss','Recall','Precision'],index=["Val","Train","Test"]).reindex(["Train","Val","Test"]))


if __name__ == "__main__":
    main()
    
