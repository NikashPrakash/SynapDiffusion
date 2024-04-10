#dataset.py
import torch, h5py
from torch.utils.data import Dataset
from torch import float32


class DDPDataset(Dataset):
    def __init__(self, hdf5_filename, dataset_names=("meg_data","labels")):
        """
        Args:
            hdf5_filename: path to the hdf5 data file.
        """
        # self.hdf5_file = h5py.File(hdf5_filename, 'r', libver='latest', swmr=True)
        with h5py.File(hdf5_filename, 'r') as hdf5_file:
            self.data = torch.tensor(hdf5_file[dataset_names[0]][:],dtype=float32)
            self.labels = torch.tensor(hdf5_file[dataset_names[1]][:],dtype=float32)
        
        # Later useage if 10G+ data
        # self.samples_per_gpu = self.total_len // world_size
        # self.start_idx = rank * self.samples_per_gpu
        # if rank == world_size - 1:  # Make sure the last GPU takes the remainder
        #     self.end_idx = self.total_len
        # self.labels = torch.from_numpy(self.labels)
        # self.data = torch.from_numpy(self.data)
        
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # data_sample = torch.from_numpy(self.data[idx].astype('float32'))
        # label_sample = torch.from_numpy(self.labels[idx].astype('int'))
        return self.data[idx], self.labels[idx]