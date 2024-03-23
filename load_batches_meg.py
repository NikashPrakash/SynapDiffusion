import mne
import os
import torch
import pandas as pd
import math

#This script creates one chunk of data per participant, segmenting it into 100ms fragments corresponding to the image the 
#participant was shown. This should only be run once assuming no changes to data/preprocessing

#TODO LOOP THROUGH ALL SUBJECTS + SAVE TO ONE FILE
#TODO write interface for creating graphs of each 100ms segment
fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/MEG_data/'
out_path_mapping = fpath + "chunks/mapping.pt"
batchSize = 32
#DOUBLE CHECK IF MAPPING CODE NEEDS TO BE DIFFERENT FOR MEG
obj_map = torch.tensor(pd.read_csv("/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Things_Images/THINGS/27 higher-level categories/category_mat_manual.tsv",delimiter="\t").values)

meg_file = fpath + f"preprocessed_P{1}-epo-{1}.fif"

raw = mne.read_epochs(meg_file,preload=True)

print(raw.info)

def normalize_tensor(inp):
    inp = (inp - torch.mean(inp,1,keepdim=True)) / torch.std(inp,1,keepdim=True) #spatial
    return (inp - torch.mean(inp,0,keepdim=True)) / torch.std(inp,0,keepdim=True) #temporal

labels = []
'''for i in range(1, 51): #TODO CHANGE TO 51
    sub_id = f"0{i}" if i < 10 else i
    eeg_file = fpath + f"sub-{sub_id}/eeg/sub-{sub_id}_task-rsvp_eeg.vhdr"
    tsv_file = fpath + f"sub-{sub_id}/eeg/sub-{sub_id}_task-rsvp_events.tsv"
    out_path_eeg = fpath + f"chunks/sub-{sub_id}.pt"
    

    raw = mne.io.read_raw_brainvision(eeg_file,preload=True)

    sfreq = raw.info['sfreq']

    segment_duration = 0.1  # 100 ms

    segment_samples = int(sfreq * segment_duration)

    total_samples = raw.n_times

    tensor_array = []
    tensors = []
    labels_array = []
    unlabeled = []
    skipped = []
    task_f = pd.read_csv(tsv_file, delimiter="\t")
    objects_classes = task_f["objectnumber"]
    startTimes = task_f["onset"]
    startTimes = startTimes.values
    objects_classes = objects_classes.values
    j = 0 #note: full path given for debug purposes
    k = 0
    for i in startTimes:
        start = math.floor(i*sfreq)
        segment_data, _ = raw[:, start:start+segment_samples]  #type(segement_data) = ,
        segment_tensor = torch.tensor(segment_data,dtype=torch.float32)
        start += segment_samples
        classification = obj_map[objects_classes[j]][[10,11,16]] #selecting cols 11, 12, 17
        if torch.sum(classification) == 1:
            if (k < batchSize):
                tensor_array.append(normalize_tensor(segment_tensor))
                labels_array.append(classification) #one hot vector of label for each object (image) 
                k += 1
            if (k == batchSize):
                tensor_array = torch.stack(tensor_array)
                labels_array = torch.stack(labels_array)
                tensors.append(tensor_array)
                labels.append(labels_array)
                k = 0
                tensor_array = []
                labels_array = []
        else:
            if (torch.sum(classification) == 0):
                unlabeled.append(j)
            else:
                skipped.append(j)
        j += 1

    tensors = torch.stack(tensors).permute(0,1,3,2) #CURRENT IMPLEMENTATION OF CODE JUST GETS RID OF SAMPLES THAT DONT FIT INTO BATCHES OF SIZE BATCHSIZE
    print("Shape of EEG tensor array:", tensors.shape)
    print(f"Shape of Label tensor array: {len(labels), len(labels[0]), len(labels[0][0])}")
    torch.save(tensors, out_path_eeg)
    print("EEG Tensor Saved to " + out_path_eeg)
    print("Label tensor Saved to " + out_path_mapping)
    print("Skipped " + str(len(unlabeled)) + " unlabeled samples")
    print("Skipped " + str(len(skipped)) + " multilabeled samples")
labels = torch.stack(labels)
torch.save(labels, out_path_mapping)
print(labels.shape)
#TODO DOUBLE CHECK DATA'''