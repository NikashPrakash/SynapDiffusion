import mne
import math
import os
import torch
import pandas as pd
import h5py

#This script creates a HDF5 dataset for all data, we extracted the 281ms fragments corresponding to the image the 
#participant was shown. This should only be run once assuming no changes to data/preprocessing

fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/EEG_data/'
hdf5_path = fpath + "eeg_data.h5"

obj_map = torch.tensor(pd.read_csv("/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Things_Images/THINGS/27 higher-level categories/category_mat_manual.tsv",delimiter="\t").values)

with h5py.File(hdf5_path, 'w') as hdf_file:
    eeg_data_dataset = None
    labels_dataset = None
    for ep in range(1, 49):
        sub_id = f"0{ep}" if ep < 10 else ep
        eeg_file = fpath + f"sub-{sub_id}/eeg/sub-{sub_id}_task-rsvp_eeg.vhdr"
        tsv_file = fpath + f"sub-{sub_id}/eeg/sub-{sub_id}_task-rsvp_events.tsv"

        raw = mne.io.read_raw_brainvision(eeg_file,preload=True)
        task_f = pd.read_csv(tsv_file, delimiter="\t")
        objects_classes = task_f["objectnumber"].values
        startTimes = torch.tensor(task_f["onset"].values * raw.info['sfreq'], dtype=torch.int32).floor()
        segment_duration = 0.1  # 100 ms
        segment_samples = int(raw.info['sfreq'] * segment_duration)
        
        label_idx = torch.where(obj_map[objects_classes][:,[10,11,16]] > 0)[0]
        
        labels_array = obj_map[objects_classes][:,[10,11,16]][label_idx].numpy()
        
        endTimes = startTimes + segment_samples
        all_data_points = torch.tensor(raw.get_data(), dtype=torch.float32).numpy()

        # Add data points to HDF5
        if ep == 1:
            eeg_data_dataset = hdf_file.create_dataset("eeg_data", data=all_data_points, maxshape=(None, all_data_points.shape[1], all_data_points.shape[2]))
            labels_dataset = hdf_file.create_dataset("labels", data=labels_array, maxshape=(None, labels_array.shape[1]))
        else:
            eeg_data_dataset.resize((eeg_data_dataset.shape[0] + all_data_points.shape[0]), axis=0)
            eeg_data_dataset[-all_data_points.shape[0]:] = all_data_points
            labels_dataset.resize(labels_dataset.shape[0] + labels_array.shape[0], axis=0)
            labels_dataset[-labels_array.shape[0]:] = labels_array

#This script creates one chunk of data per participant, segmenting it into 100ms fragments corresponding to the image the 
#participant was shown. This should only be run once assuming no changes to data/preprocessing

#TODO LOOP THROUGH ALL SUBJECTS + SAVE TO ONE FILE
#TODO write interface for creating graphs of each 100ms segment

# def normalize_tensor(inp):
#     inp = (inp - torch.mean(inp,1,keepdim=True)) / torch.std(inp,1,keepdim=True) #spatial
#     return (inp - torch.mean(inp,0,keepdim=True)) / torch.std(inp,0,keepdim=True) #temporal

labels = []
tensors = []
for i in range(1, 49): #TODO CHANGE TO 51
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
    labels_array = []
    unlabeled = 0
    skipped = 0
    task_f = pd.read_csv(tsv_file, delimiter="\t")
    objects_classes = task_f["objectnumber"]
    startTimes = task_f["onset"]
    startTimes = startTimes.values
    objects_classes = objects_classes.values
    j = 0
    k = 0
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
    if tensor_array:
        tensor_array = torch.stack(tensor_array)
        labels_array = torch.stack(labels_array)
        tensors.append(tensor_array)
        labels.append(labels_array)

    print("Shape of EEG tensor array:", tensor_array.shape)
    print(f"Shape of Label tensor array: {len(labels), len(labels[-1]), len(labels[0][-1])}")
    # torch.cat(tensors,dim=0)
    # torch.save(tensors, out_path_eeg)
    print("Skipped ", unlabeled, " unlabeled samples")
    print("Skipped ", skipped, " multilabeled samples")

tensors = torch.cat(tensors,dim=0)
print(tensors.shape)
torch.save(tensors, fpath + f"chunks/all_eeg.pt")
print("EEG Tensor Saved to " + out_path_eeg)

labels = torch.cat(labels,dim=0)
torch.save(labels, out_path_mapping)
print(labels.shape)
#TODO DOUBLE CHECK DATA