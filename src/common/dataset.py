#dataset.py
import torch, h5py
from torch.utils.data import Dataset
from torch import float32
from ray import data
import pandas as pd

def ray_dataset(dataset_base):
    dataset = data.from_numpy(dataset_base.data.numpy()) #2d data
    dataset = dataset.to_pandas()
    dataset['labels'] = list(dataset_base.labels.numpy()) #2d set of labels
    return data.from_pandas(dataset)

def ray_data_from_numpy(dataset, labels):
    dataset = data.from_numpy(dataset).to_pandas()
    dataset['labels'] = list(labels)
    return data.from_pandas(dataset)

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_filename, dataset_names=("meg_data","labels")):
        """
        Args:
            hdf5_filename: path to the hdf5 data file.
        """
        super().__init__()
        # self.hdf5_file = h5py.File(hdf5_filename, 'r', libver='latest', swmr=True)
        with h5py.File(hdf5_filename, 'r') as hdf5_file:
            self.data = torch.tensor(hdf5_file[dataset_names[0]][:],dtype=float32)
            self.labels = torch.tensor(hdf5_file[dataset_names[1]][:],dtype=float32)
        
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def to_cuda_naive(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = self.data.to(device)
        self.labels = self.labels.to(device)
        
class CombDataset(Dataset):
    def __init__(self, hdf5_filename, dataset_names=("meg","eeg","labels")):
        """
        Args:
            hdf5_filename: path to the hdf5 data file.
        """
        super().__init__()
        # self.hdf5_file = h5py.File(hdf5_filename, 'r', libver='latest', swmr=True)
        with h5py.File(hdf5_filename, 'r') as hdf5_file:
            self.meg = torch.tensor(hdf5_file[dataset_names[0]][:],dtype=float32)
            self.eeg = torch.tensor(hdf5_file[dataset_names[1]][:],dtype=float32)
            self.labels = torch.tensor(hdf5_file[dataset_names[2]][:],dtype=float32)
        
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.eeg[idx], self.meg[idx], self.labels[idx]

    def to_cuda_naive(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eeg = self.eeg.to(device)
        self.meg = self.meg.to(device)
        self.labels = self.labels.to(device)