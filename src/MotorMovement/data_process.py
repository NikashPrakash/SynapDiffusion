import os
import mne
import numpy as np
import pandas as pd

# Define the path to the data folder
data_path = "src/MotorMovement/data"

# Define the runs corresponding to each code
runs_for_left_right = [3, 4, 7, 8, 11, 12]
runs_for_both_fists_feet = [5, 6, 9, 10, 13, 14]
channels = [8,10,12,23,48,50,52,60,62]
window_size = 160  # Number of samples in each segment (1 second at 160 Hz)
step_size = 80 # Overlap between segments (50% overlap here)

# Initialize lists to store the formatted data and labels
X = []
y = []

def process_edf(subject_id, run_id):
    file_path = os.path.join(data_path, f"S{subject_id:03d}", f"S{subject_id:03d}R{run_id:02d}.edf")
    raw = mne.io.read_raw_edf(file_path, preload=True)

    # Get EEG data (excluding annotation channel)
    eeg_data = raw.get_data()[channels,:]  # Assuming first 64 channels are EEG
    sfreq = raw.info['sfreq']
    # Get annotations
    annotations = raw.annotations
    for annot in annotations:
        if annot['description'] == 'T0':
            label = [1,0,0,0,0]  # Rest
        elif annot['description'] == 'T1':
            if run_id in runs_for_left_right:
                label = [0,1,0,0,0]  # Left Fist
            elif run_id in runs_for_both_fists_feet:
                label = [0,0,1,0,0]  # Both Fists
        elif annot['description'] == 'T2':
            if run_id in runs_for_left_right:
                label = [0,0,0,1,0]  # Right Fist
            elif run_id in runs_for_both_fists_feet:
                label = [0,0,0,0,1]  # Both Feet
        
        start_sample = int(annot['onset'] * sfreq)
        end_sample = start_sample + int(annot['duration'] * sfreq)
        
        # Extract and label the data in fixed-size segments
        for start in range(start_sample, end_sample, step_size):
            end = start + window_size if end_sample >= start + window_size else end_sample
            segment = eeg_data[:, start:end]
            if segment.shape[1] < window_size:
                pad_width = window_size - segment.shape[1]
                segment = np.pad(segment, pad_width=((0, 0), (0, pad_width)), mode='constant')
            X.append(segment)
            y.append(label)

def aggregate_data():
    # Loop through all subjects and runs
    for subject_id in range(1, 110):  # Assuming subject IDs range from 001 to 109
        for run_id in range(1, 15):  # 14 recordings per subject
            process_edf(subject_id, run_id)
    
    # Convert lists to numpy arrays
    X_array = np.array(X,dtype=np.float32)
    y_array = np.array(y,dtype=np.float32)
    
    # Save the arrays for later use
    np.save("src/MotorMovement/data/X_eeg.npy", X_array)
    np.save("src/MotorMovement/data/y_labels.npy", y_array)

if __name__ == "__main__":
    aggregate_data()
    print("Data aggregation and segmentation complete. Data saved as 'X_eeg.npy' and 'y_labels.npy'.")
