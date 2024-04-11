import mne
import numpy as np
import pandas as pd
import h5py
import pywt#, scipy.signal as signal
import pdb;# pdb.set_trace() 

#This script creates a HDF5 dataset for all data, we extracted the 100ms fragments corresponding to the image the 
#participant was shown. This should only be run once assuming no changes to data/preprocessing

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

fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/EEG_data/'
hdf5_path = fpath + "eeg_data.h5"

obj_map = pd.read_csv("/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Things_Images/THINGS/27 higher-level categories/category_mat_manual.tsv",delimiter="\t").values

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
        startTimes = (task_f["onset"].values * raw.info['sfreq']).astype(np.int32)
        segment_duration = 0.1  # 100 ms
        segment_samples = int(raw.info['sfreq'] * segment_duration)

        label_idx = np.where(obj_map[objects_classes][:,[10,11,16]] > 0)[0]
        labels_array = obj_map[objects_classes][:,[10,11,16]][label_idx]
        
        all_data_points = np.array([raw[:, startTimes[i]:startTimes[i]+segment_samples][0] for i in label_idx], dtype=np.float32)
        (N, 63, 100)
        
        raw.close()
        # Add data points to HDF5
        if ep == 1:
            eeg_data_dataset = hdf_file.create_dataset("eeg_data", data=all_data_points, maxshape=(None, all_data_points.shape[1], all_data_points.shape[2]))
            labels_dataset = hdf_file.create_dataset("labels", data=labels_array, maxshape=(None, labels_array.shape[1]))
        else:
            eeg_data_dataset.resize((eeg_data_dataset.shape[0] + all_data_points.shape[0]), axis=0)
            eeg_data_dataset[-all_data_points.shape[0]:] = all_data_points
            labels_dataset.resize(labels_dataset.shape[0] + labels_array.shape[0], axis=0)
            labels_dataset[-labels_array.shape[0]:] = labels_array
    
   
# with h5py.File(hdf5_path, 'w') as hdf_file:
 # for k in range(1, 3): #TODO CHANGE TO 51
    #     sub_id = f"0{k}" if k < 10 else k
    #     eeg_file = fpath + f"sub-{sub_id}/eeg/sub-{sub_id}_task-rsvp_eeg.vhdr"
    #     tsv_file = fpath + f"sub-{sub_id}/eeg/sub-{sub_id}_task-rsvp_events.tsv"
    #     out_path_eeg = fpath + f"chunks/sub-{sub_id}.pt"
        
    #     raw = mne.io.read_raw_brainvision(eeg_file,preload=False)

    #     sfreq = raw.info['sfreq']

    #     segment_duration = 0.1  # 100 ms

    #     segment_samples = int(sfreq * segment_duration)

    #     total_samples = raw.n_times

    #     tensor_array = []
    #     labels_array = []
    #     unlabeled = 0
    #     skipped = 0
    #     task_f = pd.read_csv(tsv_file, delimiter="\t")
    #     objects_classes = task_f["objectnumber"]
    #     startTimes = task_f["onset"]
    #     startTimes = startTimes.values
    #     objects_classes = objects_classes.values
    #     j = 0
    #     for i in startTimes:
    #         start = math.floor(i*sfreq)
    #         segment_data, _ = raw[:, start:start+segment_samples]  #type(segement_data) = ,
    #         segment_tensor = torch.tensor(segment_data, dtype=torch.float32)
    #         start += segment_samples
    #         classification = obj_map[objects_classes[j]][[10,11,16]] #selecting cols 11, 12, 17
    #         if torch.sum(classification) == 1:
    #             tensor_array.append(segment_tensor)
    #             labels_array.append(classification)  # One hot vector of label for each object (image)
    #         else:
    #             if torch.sum(classification) == 0:
    #                 unlabeled += 1
    #             else:
    #                 skipped += 1
    #         j += 1

    #     print("Skipped ", unlabeled, " unlabeled samples")
    #     print("Skipped ", skipped, " multilabeled samples")
    #     tensor_array = torch.stack(tensor_array)
    #     labels_array = torch.stack(labels_array)
    #     # Add data points to HDF5 - 
    #     if k == 1:
    #         eeg_data_dataset = hdf_file.create_dataset("eeg_data", data=tensor_array, maxshape=(None, tensor_array.shape[1], tensor_array.shape[2]))
    #         labels_dataset = hdf_file.create_dataset("labels", data=labels_array, maxshape=(None, labels_array.shape[1]))
    #     else: #TODO FIX
    #         eeg_data_dataset.resize((eeg_data_dataset.shape[0] + tensor_array.shape[0]), axis=0)
    #         eeg_data_dataset[-tensor_array.shape[0]:] = segment_data
    #         labels_dataset.resize(labels_dataset.shape[0] + labels_array.shape[0], axis=0)
    #         labels_dataset[-labels_array.shape[0]:] = labels_array
    # print(k)

# # def normalize_tensor(inp):
# #     inp = (inp - torch.mean(inp,1,keepdim=True)) / torch.std(inp,1,keepdim=True) #spatial
# #     return (inp - torch.mean(inp,0,keepdim=True)) / torch.std(inp,0,keepdim=True) #temporal