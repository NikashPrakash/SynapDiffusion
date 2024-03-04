import mne 
import matplotlib as plt

datareader = mne.io.read_raw_brainvision("Data\sub-01_task-rsvp_eeg.vhdr",preload=True)
# mne.set_config("MNE_USE_CUDA", "True")
# print(mne.get_config())  # same as mne.get_config(key=None)
mne.viz.plot_raw(datareader, block=True)
