import torch
import pandas as pd
from collections import Counter

#This script creates one chunk of data per participant, segmenting it into 100ms fragments corresponding to the image the 
#participant was shown. This should only be run once assuming no changes to data/preprocessing

#TODO LOOP THROUGH ALL SUBJECTS + SAVE TO ONE FILE
fpath = '/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/EEG_data/'
out_path_mapping = fpath + f"chunks/mapping.pt"
mapping_sub1 = torch.load(out_path_mapping)
unlabeled = []
multilabeled = []

obj_map = torch.tensor(pd.read_csv("/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/Things_Images/THINGS/27 higher-level categories/category_mat_manual.tsv",delimiter="\t").values)
j = 0
k = 0
labels_array = []
concepts_array = []
for i in range(1, 2):
    if i < 10:
        istr = str("sub-0" + str(i))
    else:
        istr = str("sub-" + str(i))
    tsv_file = fpath + istr + "/eeg/" + istr + "_task-rsvp_events.tsv"
    task_f = pd.read_csv(tsv_file, delimiter="\t")
    objects_classes = task_f["objectnumber"]
    startTimes = task_f["onset"]
    startTimes = startTimes.values
    objects_classes = objects_classes.values
    j = 0
    for i in startTimes:
        classification = obj_map[objects_classes[j]].clone().detach()
        if torch.sum(classification) == 1:
            labels_array.append(torch.argmax(classification).item()) #one hot vector of label for each object (image)
            concepts_array.append(objects_classes[j])
        else:
            if (torch.sum(classification) == 0):
                unlabeled.append(j)
            else:
                multilabeled.append(j)
        j += 1
countsLabels = Counter(labels_array)
counts_dictLabels = dict(countsLabels)
print(counts_dictLabels)
conceptsLabels = Counter(concepts_array)
counts_dictConcepts = dict(conceptsLabels)
print(counts_dictConcepts)
print("Skipped " + str(len(unlabeled)) + " unlabeled samples")
print("Skipped " + str(len(multilabeled)) + " multilabeled samples")