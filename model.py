import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class MEGDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        
    def forward(self, input_data):
        return input_data
    
class EEGDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        
    def forward(self, input_data):
        return input_data
    
class NeuralGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        
    def forward(self, input_data):
        return input_data
    
class MultiSignalDataset(Dataset):
    """Dataset class for EEG and MEG visual evoked potentials (VEPs)."""

    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data['labels'])
    

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.data.items()}