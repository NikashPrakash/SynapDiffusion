# import pywt
# import numpy as np

# # Assuming 'eeg_data' is your EEG data tensor of shape [channels, samples]
# # For simplicity, let's assume we have a 1D tensor (a single EEG channel)
# eeg_channel = eeg_data[0, :]  # take the first channel for example

# # Choose a wavelet
# wavelet_type = 'db4'  # A common type of Daubechies wavelet
# wavelet = pywt.Wavelet(wavelet_type)

# # Determine the maximum decomposition level
# max_level = pywt.dwt_max_level(data_len=len(eeg_channel), filter_len=wavelet.dec_len)

# # Perform DWT - You may want to choose a specific level less than the max level
# coeffs = pywt.wavedec(eeg_channel, wavelet_type, level=max_level)

# # 'coeffs' is a list of arrays containing the approximation and detail coefficients:
# # [cAn, cDn, cDn-1, ..., cD2, cD1] where 'A' denotes approximation and 'D' denotes details

# # Perform DWT for each channel
# all_coeffs = [pywt.wavedec(eeg_data[channel, :], wavelet_type, level=max_level)
#               for channel in range(eeg_data.shape[0])]

import mne
import math
import os
import torch
import pandas as pd
import h5py
#import pdb; pdb.set_trace() 

#This script creates a HDF5 dataset for all data, we extracted the 281ms fragments corresponding to the image the 
#participant was shown. This should only be run once assuming no changes to data/preprocessing

fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/EEG_data/'
hdf5_path = fpath + "eeg_data.h5"

obj_map = torch.tensor(pd.read_csv("/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Things_Images/THINGS/27 higher-level categories/category_mat_manual.tsv",delimiter="\t").values)
eeg_data_dataset = None

with h5py.File(hdf5_path, 'w') as hdf_file:
    for k in range(1, 2): #TODO CHANGE TO 51
        sub_id = f"0{k}" if k < 10 else k
        eeg_file = fpath + f"sub-{sub_id}/eeg/sub-{sub_id}_task-rsvp_eeg.vhdr"
        tsv_file = fpath + f"sub-{sub_id}/eeg/sub-{sub_id}_task-rsvp_events.tsv"
        out_path_eeg = fpath + f"chunks/sub-{sub_id}.pt"
        
        raw = mne.io.read_raw_brainvision(eeg_file,preload=False)

        sfreq = raw.info['sfreq']

        segment_duration = 0.1  # 100 ms

        segment_samples = int(sfreq * segment_duration)

        total_samples = raw.n_times

        tensor_array = []
        labels_array = []
        unlabeled = 0
        skipped = 0
        task_f = pd.read_csv(tsv_file, delimiter="\t")
        objects_classes = task_f["objectnumber"]
        startTimes = task_f["onset"]
        startTimes = startTimes.values
        objects_classes = objects_classes.values
        j = 0
        for i in startTimes:
            start = math.floor(i*sfreq)
            segment_data, _ = raw[:, start:start+segment_samples]  #type(segement_data) = ,
            segment_tensor = torch.tensor(segment_data, dtype=torch.float32)
            start += segment_samples
            classification = obj_map[objects_classes[j]][[10,11,16]] #selecting cols 11, 12, 17
            if torch.sum(classification) == 1:
                tensor_array.append(segment_tensor)
                labels_array.append(classification)  # One hot vector of label for each object (image)
            else:
                if torch.sum(classification) == 0:
                    unlabeled += 1
                else:
                    skipped += 1
            j += 1

        print("Skipped ", unlabeled, " unlabeled samples")
        print("Skipped ", skipped, " multilabeled samples")
        tensor_array = torch.cat(tensor_array,dim=0)
        labels_array = torch.cat(labels_array,dim=0)        
        # Add data points to HDF5 - 
        if k == 1:
            eeg_data_dataset = hdf_file.create_dataset("eeg_data", data=tensor_array, maxshape=(None, tensor_array.shape[1], tensor_array.shape[2]))
            labels_dataset = hdf_file.create_dataset("labels", data=labels_array, maxshape=(None, labels_array.shape[1]))
        else: #TODO FIX
            eeg_data_dataset.resize((eeg_data_dataset.shape[0] + tensor_array.shape[0]), axis=0)
            eeg_data_dataset[-tensor_array.shape[0]:] = segment_data
            labels_dataset.resize(segment_data.shape[0] + labels_array.shape[0], axis=0)
            labels_dataset[-segment_data.shape[0]:] = labels_array
    print(eeg_data_dataset)
    print(k)