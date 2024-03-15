import mne
import os
import torch
import pandas as pd
import math

#This script creates one chunk of data per participant, segmenting it into 100ms fragments corresponding to the image the 
#participant was shown. This should only be run once assuming no changes to data/preprocessing

#TODO LOOP THROUGH ALL SUBJECTS + SAVE TO ONE FILE
#TODO write interface for creating graphs of each 100ms segment
fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/EEG_data/'
out_path_mapping = fpath + "chunks/mapping.pt"
batchSize = 32

def normalize_tensor(inp):
    inp = (inp - torch.mean(inp,1,keepdim=True)) / torch.std(inp,1,keepdim=True) #spatial
    return (inp - torch.mean(inp,0,keepdim=True)) / torch.std(inp,0,keepdim=True) #temporal

for i in range(1, 2): #TODO CHANGE TO 51
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
    labels = []
    unlabeled = []
    skipped = []
    task_f = pd.read_csv(tsv_file, delimiter="\t")
    objects_classes = task_f["objectnumber"]
    startTimes = task_f["onset"]
    startTimes = startTimes.values
    objects_classes = objects_classes.values
    obj_map = torch.tensor(pd.read_csv("/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Things_Images/THINGS/27 higher-level categories/category_mat_manual.tsv",delimiter="\t").values)
    j = 0 #note: full path given for debug purposes
    k = 0
    for i in startTimes:
        start = math.floor(i*sfreq)
        segment_data, _ = raw[:, start:start+segment_samples]  #type(segement_data) = ,
        segment_tensor = torch.tensor(segment_data,dtype=torch.float32)
        start += segment_samples
        classification = obj_map[objects_classes[j]][[10,11,16]] #selecting rows 11, 17, 12
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
    labels =  torch.stack(labels) #torch.vstack((mapping_sub1, torch.stack(labels))
    print("Shape of EEG tensor array:", tensors.shape)
    print("Shape of Label tensor array:", labels.shape)
    torch.save(tensors, out_path_eeg)
    torch.save(labels, out_path_mapping)
    print("EEG Tensor Saved to " + out_path_eeg)
    print("Label tensor Saved to " + out_path_mapping)
    print("Skipped " + str(len(unlabeled)) + " unlabeled samples")
    print("Skipped " + str(len(skipped)) + " multilabeled samples")

#TODO DOUBLE CHECK DATA