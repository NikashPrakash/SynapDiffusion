import mne
import numpy as np
import pandas as pd
import h5py
import pywt#, scipy.signal as signal
import pdb; #pdb.set_trace() 

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
hdf5_path = fpath + "eeg_data.hdf5"

obj_map = pd.read_csv("/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Things_Images/THINGS/27 higher-level categories/category_mat_manual.tsv",delimiter="\t").values

with h5py.File(hdf5_path, 'w') as hdf_file:
    eeg_data_dataset = None
    labels_dataset = None
    for ep in range(1, 10): #TODO CHANGE TO USE ALL DATA
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
        all_data_points = all_data_points * 100
        #(N, 63, 100)

        # Choose a wavelet TODO MOVE ALL THIS TO MODEL/TRAINING CODE, USE CWT?
        wavelet_type = 'sym9'
        wavelet = pywt.Wavelet(wavelet_type)

        # Determine the maximum decomposition level 
        #max_level = pywt.dwt_max_level(data_len=len(all_data_points[0][0]), filter_len=wavelet.dec_len)
        #max_level = 2 
        all_coeffs2d = [pywt.dwt2(all_data_points[i, :, :], wavelet_type, 'per') for i in range(all_data_points.shape[0])]
        arr = []

        for i in all_coeffs2d: #each sample consists of 4 32x50 matrices- 1st is approximation, rest are detail #TODO DOWNSAMPLE TO DIFFERENT BANDS???
            arr.append([i[0], i[1][0], i[1][1], i[1][2]])
        all_coeffs  = np.array(arr)
        raw.close()
        # Add data points to HDF5
        if ep == 1:
            eeg_data_dataset = hdf_file.create_dataset("eeg_data", data=all_data_points, maxshape=(None, all_data_points.shape[1], all_data_points.shape[2]))
            labels_dataset = hdf_file.create_dataset("labels", data=labels_array, maxshape=(None, labels_array.shape[1]))
            wavelet_dataset = hdf_file.create_dataset('wavelets', data=all_coeffs, maxshape=(None, all_coeffs.shape[1], all_coeffs.shape[2], all_coeffs.shape[3]))
        else:
            eeg_data_dataset.resize((eeg_data_dataset.shape[0] + all_data_points.shape[0]), axis=0)
            eeg_data_dataset[-all_data_points.shape[0]:] = all_data_points
            labels_dataset.resize(labels_dataset.shape[0] + labels_array.shape[0], axis=0)
            labels_dataset[-labels_array.shape[0]:] = labels_array
            wavelet_dataset.resize(wavelet_dataset.shape[0] + all_coeffs.shape[0], axis=0)
            wavelet_dataset[-all_coeffs.shape[0]:] = all_coeffs

# # def normalize_tensor(inp):
# #     inp = (inp - torch.mean(inp,1,keepdim=True)) / torch.std(inp,1,keepdim=True) #spatial
# #     return (inp - torch.mean(inp,0,keepdim=True)) / torch.std(inp,0,keepdim=True) #temporal