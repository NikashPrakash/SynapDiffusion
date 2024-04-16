#training.py
import mne
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import pandas as pd
import os
import utils
from utils import config
from models.MEGDecoder import *



def save_checkpoint(model, optimizer, epoch, checkpoint_dir, stats, params=None):
    """Save a checkpoint file to `checkpoint_dir`."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        'optimizer': optimizer.state_dict(),
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

def _get_metrics(model, criterion, data, labels):
    y_true, y_pred = [], []
    correct, total, true_posit, false_posit, false_neg = 0, 0, 0, 0, 0
    running_loss = []
    with torch.no_grad():
        for batchNum in range(data.shape[0]):
            y = labels[batchNum]
            X = data[batchNum]
            # input_ids = batch['input_ids']
            # attention_mask = batch['attention_mask']
            output = model(X) 
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

def evaluate_chunk(
    trainDat,
    trainLabels,
    valDat,
    valLabels,
    model,
    criterion,
    stats
):
    """Evaluate the `model` on the train and validation set."""
    model.eval()

    train_acc, train_loss, train_rec, train_prec = _get_metrics(model, criterion, trainDat, trainLabels)
    val_acc, val_loss, val_rec, val_prec = _get_metrics(model, criterion, valDat, valLabels)

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

        evaluate_chunk(tr_loader, val_loader, model, criterion, stats)
        
        best_epoch = train(model, optimizer, criterion, tr_loader, val_loader, stats, 0, True)
        model, stats = restore_best(model, best_epoch, config('.checkpoint'))
        clear_checkpoint(config('.checkpoint'))
        
        if stats[-1][1] < best_performance:
            best_performance = stats[-1][1]
            best_model = model
            best_stats = stats
            params = lr,drop_rate,w_d
            save_checkpoint(model, optimizer, best_epoch, config(".checkpoint"), stats, params)
    return best_performance, best_stats, best_model



def train_chunk(model, trainDat, trainLabels, optimizer,criterion):
    '''
    params: 
        model_input_vars: List, names of all the input variables for a model (ex. EEG Encoder Model - EEG inp data)
    '''
    model.train()
    for batchNum in range(trainDat.shape[0]):
        optimizer.zero_grad()
        y = trainLabels[batchNum]
        data = trainDat[batchNum]
        pred = model(data)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()


def train(model, optimizer, criterion, trainDat, trainLabels, valDat, valLabels, stats):
    train_chunk(model,trainDat, trainLabels, optimizer, criterion)
    evaluate_chunk(
        trainDat,
        trainLabels,
        valDat,
        valLabels,
        model,
        criterion,
        stats
    )
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
    fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ",device)
    stats = []
    model = EEGDecoder(63, 63, 27) #TODO GET PARAMS FROM PAPER
    criterion=nn.CrossEntropyLoss() # TODO: use chosen loss models
    optimizer=torch.optim.Adam(model.parameters(),weight_decay=1,lr=0.1) # TODO: This optimizer or another? TODO choose weight/lr
    mapping = torch.load(fpath + 'EEG_data/chunks/mapping.pt')
    totBatches = mapping.shape[0] #TODO should be mapping.shape[0] * batches_per_chunk or something
    numChunks = 1 #TODO just use mapping file for this once we have multiple files
    
    X = np.arange(0, totBatches) #THIS WILL NOT WORK FOR >1 CHUNK!
        
    # for param in model.parameters(): #transfer learning, comment out for fine-tuning
    #     param.requires_grad = False    
                                                           
    pt_models = {} #TODO: Add pretrained models to this
    tr_idx, test_idx = train_test_split(X,shuffle=True,test_size=0.2) #INDICIES WRT ALL SAMPLES ACROSS ALL PARTICIPANTS
    currBatch = 0
    epoch = 0
    patience = 5
    curr_count_to_patience = 0
    #global_min_loss = stats[0][1]
    while curr_count_to_patience < patience:
        temp = []
        for i in range(numChunks):  #TRAINING LOOP- PER CHUNK: (train function should be run on each CHUNK)
            trdat = {}
            valdat = {}
            currSub = "sub-0" + str(i+1) #TODO fix case where participant number >9 lol
            eegdata = torch.load(fpath + '/EEG_data/chunks/' + currSub + ".pt") 
            numBatches = eegdata.shape[0]
            idxs = tr_idx[np.where((tr_idx >= currBatch) & (tr_idx < (currBatch + numBatches)))] #TODO check bounds
            tr_idx, val_idx = train_test_split(idxs,shuffle=True,test_size=0.1)
            trdateeg  = eegdata[tr_idx - currBatch, :, :] #makes index wrt current chunk
            trdatlabels  = mapping[tr_idx, :, :] #TODO: Any necessary data transformations (if any)... 
            valdateeg = eegdata[val_idx - currBatch, :, :] #makes index wrt current chunk 
            valdatlabels = mapping[val_idx, :, :] #TODO: Any necessary data transformations (if any)... 
            temp.append(train(model, optimizer, criterion, trdateeg, trdatlabels, valdateeg, valdatlabels, stats))
            currBatch = currBatch + numBatches
        stats = np.mean(temp)
        #save_checkpoint(model, optimizer, epoch + 1, config(".checkpoint"), stats) #TODO checkpoint stuff
        curr_count_to_patience, global_min_loss = early_stopping(
            stats, curr_count_to_patience, global_min_loss
        )
        for i in range(numChunks):  #TESTING LOOP- PER CHUNK: (train function should be run on each CHUNK)
            trdat = {}
            valdat = {}
            eegdata = torch.load('/EEG_data/chunks/sub-0' + str(i)) #TODO fix case where participant number >9 lol
            numBatches = eegdata.shape[0]
            idxs = tr_idx[np.where((tr_idx > currBatch) & (tr_idx < (currBatch + numBatches)))] #TODO check bounds
            tr_idx, val_idx = train_test_split(idxs,shuffle=True,test_size=0.2)
            #test["data"]  = eegdata[tr_idx - currBatch, :, :] #makes index wrt current chunk
            #test["labels"]  = mapping[tr_idx, :, :] #TODO: Any necessary data transformations (if any)... 
            temp.append(train(model, optimizer, criterion, trdat, valdat, stats))
            currBatch = currBatch + numBatches
    #TESTING LOOP- PER CHUNK: ( test function should be run on each CHUNK) 
        #get labels from corresponding label tensor
        #grab training indicies in chunk from big train array above
        #PER TESTING BATCH: 
            #test_inputs = chunk[indicies from big array]
            #test_labels = chunk[label indicies from big array]
            #evaluate performance
        #average performances across batches
    #average performance across chunks

    
    #perf, stats, model = hyper_search(pt_models, train_loader, val_loader, device)
    #print(f"hyperparam search best val loss: {perf}")
    
    #utils.make_training_plot(np.array(stats))
    #stats[-1] += list(_get_metrics(model, criterion, test_loader))
    # np.save("best_model_stats.npy",np.array(stats))
    #print(pd.DataFrame(np.array(stats[-1]).reshape(3,4), columns=['Accuracy','Loss','Recall','Precision'],index=["Val","Train","Test"]).reindex(["Train","Val","Test"]))


if __name__ == "__main__":
    main()
    
