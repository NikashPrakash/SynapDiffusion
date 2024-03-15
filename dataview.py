# import mne 
# import h5py
import numpy as np
import scipy.io
import os
import torch
import pandas as pd
import math
# import matplotlib as plt

# datareader = mne.io.read_raw_brainvision("Data\sub-01_task-rsvp_eeg.vhdr",preload=True)
# mne.set_config("MNE_USE_CUDA", "True")
# print(mne.get_config())  # same as mne.get_config(key=None)


#This script creates one chunk of data per participant, segmenting it into 100ms fragments corresponding to the image the 
#participant was shown. This should only be run once assuming no changes to data/preprocessing

#TODO LOOP THROUGH ALL SUBJECTS + SAVE TO ONE FILE
#TODO write interface for creating graphs of each 100ms segment

batchSize = 32
fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/EEG_data/'
meg_file_template = 'preprocessed_P1-epo-1.fif'
tsv_file = fpath + 'sub-01/eeg/sub-01_task-rsvp_events.tsv'
out_path_eeg = fpath + 'chunks/sub-01.pt' 
out_path_mapping = fpath + 'chunks/mapping.pt' #MAPPING IS ONLY ONE FILE-- FOR ALL CHUNKS

eeg_dat = torch.load(out_path_eeg)
print(eeg_dat.shape)


# raw = mne.io.read_raw_fif("MEG_data/preprocessed_P1-epo-1.fif")
# for fname in os.listdir(fpath):
#     with open(fname) as f:
        

# sfreq = raw.info['sfreq']

# segment_duration = 0.1  # 100 ms

# segment_samples = int(sfreq * segment_duration)

# total_samples = raw.n_times

# tensor_array = []
# tensors = []
# labels_array = []
# labels = []
# task_f = pd.read_csv(tsv_file, delimiter="\t")
# objects_classes = task_f["objectnumber"]
# startTimes = task_f["onset"]
# startTimes = startTimes.values
# objects_classes = objects_classes.values
# obj_map = torch.tensor(pd.read_csv("Things_Images/THINGS/27 higher-level categories/category_mat_manual.tsv",delimiter="\t").values)
# j = 0
# k = 0
# for i in startTimes: #TODO LOOP THROUGH ALL FILES 
#     start = math.floor(i*sfreq)
#     segment_data, _ = raw[:, start:start+segment_samples]  #type(segement_data) = ,
#     segment_tensor = torch.tensor(segment_data, dtype=torch.float32)
#     start += segment_samples
#     print("Grabbing " + str(segment_samples) + " samples starting at time(ms) " + str(i))
#     if (k < batchSize):
#         tensor_array.append(segment_tensor)
#         labels_array.append(torch.tensor(obj_map[objects_classes[j]])) #one hot vector of label for each object (image) 
#         k += 1
#     if (k == batchSize):
#         tensor_array = torch.stack(tensor_array)
#         labels_array = torch.stack(labels_array)
#         tensors.append(tensor_array)
#         labels.append(labels_array)
#         k = 0
#         tensor_array = []
#         labels_array = []
#     j += 1

# tensors = torch.stack(tensors) #CURRENT IMPLEMENTATION OF CODE JUST GETS RID OF SAMPLES THAT DONT FIT INTO BATCHES OF SIZE BATCHSIZE
# labels = torch.stack(labels)

# print("Shape of EEG tensor array:", tensors.shape)
# print("Shape of Label tensor array:", labels.shape)
# torch.save(tensors, out_path_eeg)
# torch.save(labels, out_path_mapping)
# print("EEG Tensor Saved to " + out_path_eeg)
# print("Label tensor Saved to " + out_path_mapping)