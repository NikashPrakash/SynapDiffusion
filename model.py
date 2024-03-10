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
    def __init__(self, input_size, hidden_size, output_size):
        super(EEGDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(32, 32, batch_first=True) #TODO UPDATE ACCORDING TO EEG NUMBER OF CHANNELS!
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        # input_data shape: (batch_size, sequence_length, input_size)
        lstm_out, tmp = self.lstm(input_data)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        output = self.fc(lstm_out[:, -1, :])  # Take the last timestamp output
        # output shape: (batch_size, output_size)
        output = self.softmax(output)
        return output
    
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