#training.py
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import pandas as pd
import os
from utils import *
import matplotlib as plt
import seaborn as sns
from utils import config
from model import *



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


def decode(preds, top):
    out =  torch.argsort(preds, axis=1)[:, -top:]
    return out.float()
    
def _get_metrics(model, criterion, data, labels):
    y_true, y_pred = [], []
    correct, total = 0, 0
    running_loss = []
    with torch.no_grad():
        for batchNum in range(data.shape[0]):
            y = torch.argmax(labels[batchNum],1)
            X = data[batchNum]
            output = model(X) 
            predicted = decode(output,1).int()
            # for i in range(y.shape[0]): top-5 acc
            #     correct += (y[i] in predicted[i])
            correct += (predicted[:,0] == y).sum().item() #only for top-1 acc
            total += y.size(0)
            running_loss.append(criterion(output, y).item())
            y_pred.append(predicted)
    y_pred = torch.stack(y_pred)
    loss = np.mean(running_loss)
    acc = correct / total
    return acc, loss, y_pred

def evaluate_chunk(
    trainDat,
    trainLabels,
    valDat,
    valLabels,
    model,
    criterion,
):
    """Evaluate the `model` on the train and validation set."""
    model.eval()

    train_acc, train_loss, _ = _get_metrics(model, criterion, trainDat, trainLabels)
    val_acc, val_loss, y_pred = _get_metrics(model, criterion, valDat, valLabels)

    stats = [
        val_acc,
        val_loss,
        train_acc,
        train_loss,
    ]
    return stats, y_pred


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



def train_chunk(model, trainDat, trainLabels, optimizer, criterion):
    '''
    params: 
        model_input_vars: List, names of all the input variables for a model (ex. EEG Encoder Model - EEG inp data)
    '''
    trainStats = []
    model.train()
    for batchNum in range(trainDat.shape[0]):
        optimizer.zero_grad()
        y = torch.argmax(trainLabels[batchNum],1) 
        data = trainDat[batchNum]
        pred = model(data)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        trainStats.append(loss.cpu().detach())
    return trainStats

def train(model, optimizer, criterion, trainDat, trainLabels, valDat, valLabels):
    trainStats = train_chunk(model,trainDat, trainLabels, optimizer, criterion)
    make_minibatch_training_plot(trainStats,"Chunk-1")
    stats,y_pred = evaluate_chunk(
        trainDat,
        trainLabels,
        valDat,
        valLabels,
        model,
        criterion,
    )
    return stats, y_pred

def early_stopping(stats, curr_count_to_patience, global_min_loss):
    """Calculate new patience and validation loss.

    Increment curr_patience by one if new loss is not less than global_min_loss
    Otherwise, update global_min_loss with the current val loss, and reset curr_count_to_patience to 0
    Increment curr_patience by one if new loss is not less than global_min_loss
    Otherwise, update global_min_loss with the current val loss, and reset curr_count_to_patience to 0

    Returns: new values of curr_patience and global_min_loss
    """
    if stats[-1]['val_loss'] >= global_min_loss:
      curr_count_to_patience += 1
    else:
      global_min_loss = stats[-1]['val_loss']
      curr_count_to_patience = 0
    return curr_count_to_patience, global_min_loss


def main():
    fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapping = torch.load(fpath + 'EEG_data/chunks/mapping.pt').to(device)
    model = EEGDecoder(63, 63, 3).to(device) 
    #criterion= nn.BCEWithLogitsLoss()
    criterion= nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001) # TODO choose weight/lr - in hyperparam search
    totBatches = mapping.shape[0] #TODO should be mapping.shape[0] * batches_per_chunk or something
    numChunks = 1 #TODO just use mapping file for this once we have multiple files
    
    X = np.arange(0, totBatches) #THIS WILL NOT WORK FOR >1 CHUNK!
        
    # for param in model.parameters(): #transfer learning, comment out for fine-tuning
    #     param.requires_grad = False                                          
    # pt_models = {} #TODO: Add pretrained models to this
    
    clear_checkpoint(config("EEG-Encoder.checkpoint"))
    
    trMain, test_idx = train_test_split(X,shuffle=True,test_size=0.2) #INDICIES WRT ALL SAMPLES ACROSS ALL PARTICIPANTS
    tr_idx, val_idx = train_test_split(trMain,shuffle=True,test_size=0.1) #INDICIES WRT ALL SAMPLES ACROSS ALL PARTICIPANTS
    epoch = 0
    best_epoch = 0
    patience = 7
    curr_count_to_patience = 0
    global_min_loss = 1000
    global_metrics = [] #list of metrics from each epoch- across chunks
    while curr_count_to_patience < patience:
        chunk_metrics = {
            'train_loss': 0,
            'val_loss': 0,
            'train_accuracy': 0,
            'val_accuracy': 0
        }
        currBatch = 0
        for i in range(numChunks):  #TRAINING LOOP- PER CHUNK: (train function should be run on each CHUNK)
            currSub = "sub-0" + str(i+1) #TODO fix case where participant number >9 lol
            eegdata = torch.load(fpath + '/EEG_data/chunks/' + currSub + ".pt",map_location=device)  #load chunk
            # numBatches = eegdata.shape[0] #num batches in  file
            #idxs = tr_idx[np.where((tr_idx >= currBatch) & (tr_idx < (currBatch + numBatches)))] #TODO check bounds MAKE WORK FOR MULTIPLE CHUNKS
            #print(idxs)            
            tr_eeg  = eegdata[tr_idx - currBatch, :, :].to(device) #makes index wrt current chunk
            tr_labels  = mapping[tr_idx, :, :] #Any necessary data transformations (if any)... 
            val_eeg = eegdata[val_idx - currBatch, :, :].to(device) #makes index wrt current chunk 
            val_labels = mapping[val_idx, :, :] #Any necessary data transformations (if any)... 
            
            out,y_pred = train(model, optimizer, criterion, tr_eeg, tr_labels, val_eeg, val_labels)
            chunk_metrics['train_loss'] = out[1]
            chunk_metrics['train_accuracy'] = out[0]
            chunk_metrics['val_loss'] = out[3]
            chunk_metrics['val_accuracy'] = out[2]
            #currBatch = currBatch + numBatches
        epoch += 1
        print("Epoch " + str(epoch))
        print("\tEpoch Metric ",chunk_metrics,"\n")
        print(f"Predictions {y_pred.flatten(start_dim=1)}")
        global_metrics.append(chunk_metrics) #TODO GLOBAL STATS SHOULD BE AVERAGE OF CHUNK STATS WHEN WE HAVE >1 CHUNK 
        save_checkpoint(model, optimizer, epoch, config("EEG-Encoder.checkpoint"), global_metrics)
        curr_count_to_patience, global_min_loss = early_stopping(
            global_metrics, curr_count_to_patience, global_min_loss
        )
        if curr_count_to_patience == patience:
            best_epoch = epoch - patience

    model, bestStats = restore_best(model, best_epoch, config("EEG-Encoder.checkpoint"), pretrain=False)
    test_eeg  = eegdata[test_idx, :, :].to(device) #makes index wrt current chunk
    test_labels  = mapping[test_idx, :, :] #Any necessary data transformations (if any)... 
    test_acc, test_loss, test_pred =  _get_metrics(model, criterion, test_eeg, test_labels)


        # for i in range(numChunks):  #TESTING LOOP- PER CHUNK: (train function should be run on each CHUNK)
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
    #print(pd.DataFrame(np.array(stats[-1]).reshape(3,4), columns=['Accuracy','Loss','Recall','Precision'],index=["Val","Train","Test"]).reindex(["Train","Val","Test"]))
    print("Global min val loss ", global_min_loss,"\n")
    print(f"test_acc {test_acc}, test_loss {test_loss}")
    #print(global_metrics)

if __name__ == "__main__":
    main()