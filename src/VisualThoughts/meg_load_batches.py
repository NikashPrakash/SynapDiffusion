import mne
import os
import torch
import pandas as pd
import h5py

#This script creates a HDF5 dataset for all data, we extracted the 281ms fragments corresponding to the image the 
#participant was shown. This should only be run once assuming no changes to data/preprocessing

fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/MEG_data/'
hdf5_path = fpath + "meg_data.hdf5"

obj_map = torch.tensor(pd.read_csv("/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Things_Images/THINGS/27 higher-level categories/category_mat_manual.tsv",delimiter="\t").values)
pic_map = torch.tensor(pd.read_csv("/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Things_Images/THINGS/Metadata/Concept-specific/image_concept_index.csv").values)

with h5py.File(hdf5_path, 'w') as hdf_file:
    for ep in range(1, 5):
        meg_file = fpath + f"preprocessed_P{ep}-epo.fif"
        raw = mne.read_epochs(meg_file, preload=True)

        # epochs that (datapoints) have high level object classes
        epoch_labels = torch.tensor([raw.event_id[str(event_id)] for event_id in raw.events[:, 2] if raw.event_id[str(event_id)] != 999999])
        #select binary class indicies from epoch labels
        label_idx = torch.where(obj_map[pic_map[epoch_labels - 1] - 1, [2, 10]] > 0)[0]
        #select binary classes from original map 
        labels_array = obj_map[pic_map[epoch_labels - 1] - 1, [2, 10]][label_idx].clone()
        #select binary class data from original set
        all_data_points = torch.tensor(raw.get_data(copy=True)[epoch_labels[label_idx] - 1], dtype=torch.float32)

        
        # Add data points to HDF5
        if ep == 1:
            hdf_data = hdf_file.create_dataset("meg_data", data=all_data_points.numpy(), maxshape=(None,) + all_data_points.shape[1:], chunks=True,scaleoffset=0)
            labels_dataset = hdf_file.create_dataset("labels", data=labels_array.numpy(), maxshape=(None, 2), dtype='float32', chunks=True)
        else:
            hdf_data.resize((hdf_data.shape[0] + all_data_points.shape[0]), axis=0)
            hdf_data[-all_data_points.shape[0]:] = all_data_points.numpy()
            labels_dataset.resize(labels_dataset.shape[0] + labels_array.shape[0], axis=0)
            labels_dataset[-labels_array.shape[0]:] = labels_array.numpy()