import mne
import os
import torch
import pandas as pd
import h5py
import numpy as np

fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/MEG_data/'
combined_fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Combined_data/'
hdf5_path = combined_fpath + "combined_data.hdf5"

obj_map = pd.read_csv("/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Things_Images/THINGS/27 higher-level categories/category_mat_manual.tsv",delimiter="\t").values
pic_map = pd.read_csv("/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Things_Images/THINGS/Metadata/Concept-specific/image_concept_index.csv").values

concept_map = {}
label_map = {}

with h5py.File(hdf5_path, 'w') as hdf_file:
    for ep in range(1, 5):
        meg_file = fpath + f"preprocessed_P{ep}-epo.fif"

        raw = mne.read_epochs(meg_file, preload=True)
        all_data_points = raw.get_data()
        event_id_mapping = raw.event_id 
        epoch_labels = [event_id_mapping[str(event_id)] for event_id in raw.events[:, 2]]

        for j in range(0, len(all_data_points)): 
            if epoch_labels[j] != 999999:
                segment_tensor = np.array(all_data_points[j])
                classification = obj_map[pic_map[epoch_labels[j] - 1][0] - 1]
                if classification[2] + classification[10] > 0:
                    concept = pic_map[epoch_labels[j] - 1][0] - 1
                    if concept not in concept_map:
                        label_map[concept] = []
                        label = [1 if classification[2] == 1 else 0, 1 if classification[10] == 1 else 0]
                        label_map[concept].append(label)
                        concept_map[concept] = []
                    concept_map[concept].append([segment_tensor])

    fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/EEG_data/'

    eeg_data_dataset = None
    labels_dataset = None
    num_entries = {}
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

        for i in range(len(objects_classes)):
            label = obj_map[objects_classes[i]]
            if label[2] + label[10] > 0:
                c = objects_classes[i]
                if c not in num_entries:
                    num_entries[c] = 0
                if len(concept_map[c]) > num_entries[c]:
                    data = np.array(raw[:, startTimes[i]:startTimes[i]+segment_samples][0])
                    data *= 100
                    concept_map[c][num_entries[c]].append(np.array(data))
                    num_entries[c] += 1

    for c in concept_map.keys():
        temp = [x for x in concept_map[c] if len(x) > 1]
        concept_map[c] = temp

    final_eeg = []
    final_meg = []
    final_labels = []
    for c in concept_map.keys():
        meg = [x[0] for x in concept_map[c]]
        eeg = [x[1] for x in concept_map[c]]
        final_eeg.extend(eeg)
        final_meg.extend(meg)
        lbls = [label_map[c][0]]*len(concept_map[c])
        final_labels.extend(lbls)

    hdf_file.create_dataset("eeg", data=np.array(final_eeg), chunks=True, scaleoffset=0)
    hdf_file.create_dataset("meg", data=np.array(final_meg), chunks=True, scaleoffset=0)
    hdf_file.create_dataset("labels", data=np.array(final_labels), chunks=True, scaleoffset=0)