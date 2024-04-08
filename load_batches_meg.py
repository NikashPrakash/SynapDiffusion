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
out_path_mapping = fpath + "mapping.pt"
batchSize = 32

obj_map = torch.tensor(pd.read_csv("/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Things_Images/THINGS/27 higher-level categories/category_mat_manual.tsv",delimiter="\t").values)
pic_map = torch.tensor(pd.read_csv("/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Things_Images/THINGS/Metadata/Concept-specific/image_concept_index.csv").values)
labels = []

for ep in range(1,5):
    meg_file = fpath + f"preprocessed_P{ep}-epo.fif"

    raw = mne.read_epochs(meg_file,preload=True)

    # Get all data points from Epochs object
    all_data_points = raw.get_data()

    # Dimensions of the data array
    n_epochs, n_channels, n_time_points = all_data_points.shape

    print(f"Number of epochs: {n_epochs}")
    print(f"Number of channels: {n_channels}")
    print(f"Number of time points per epoch: {n_time_points}")

    event_id_mapping = raw.event_id  # Mapping of event IDs to event descriptions

    # Get labels for each epoch
    epoch_labels = [event_id_mapping[str(event_id)] for event_id in raw.events[:, 2] if event_id_mapping[str(event_id)] != 999999]

    
    j = 0

    for i in range(1, 5):
        out_path_meg = fpath + f"epo-{ep}-chunk-{i}.pt"
        tensor_array = []
        tensors = []
        labels_array = []
        k = 0
        unlabeled = 0
        skipped = 0
        for j in range(0, int(n_epochs/4)):
            segment_tensor = torch.tensor(all_data_points[j],dtype=torch.float32)
            classification = obj_map[pic_map[epoch_labels[j] - 1] - 1,[2,10]]
            if torch.sum(classification) == 1:
                if (k < batchSize):
                    tensor_array.append(segment_tensor)
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
                    unlabeled += 1
                else:
                    skipped += 1
            j += 1
        tensors = torch.stack(tensors)
        print("Shape of MEG tensor array:", tensors.shape)
        print(f"Shape of Label tensor array: {len(labels), len(labels[0]), len(labels[0][0])}")
        torch.save(tensors, out_path_meg)
        print("MEG Tensor Saved to " + out_path_meg)
        print("Label tensor Saved to " + out_path_mapping)
        print("Skipped ", unlabeled, " unlabeled samples")
        print("Skipped ", skipped, " multilabeled samples")

labels = torch.stack(labels)
torch.save(labels, out_path_mapping)
print(labels.shape)

#TODO DOUBLE CHECK DATA