import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pywt
from models.MEGDecoder import ConvLayer

wavelet_type = 'sym9'

class FusionClassification(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.meg_layer_conv = ConvLayer()
        self.meg_layer_fc = nn.Linear(32*141, 128)
        self.eeg_layer_conv = nn.Sequential(
            # nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=1),
            # nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(24),
            # nn.MaxPool2d(kernel_size=2, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.MaxPool2d(kernel_size=2, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            # nn.MaxPool2d(kernel_size=2, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=1),
            # nn.ReLU()
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
            )
        self.eeg_layer_fc = nn.Linear(64 * 13 * 22, 128)
        
        self.feat_select = nn.Linear(256,64) #TODO FIX THESE NUMBERS BASED ON PREVIOUS LAYERS
        self.classifer = nn.Linear(64,2)
        self.d1 = nn.Dropout(0.15)
        self.d2 = nn.Dropout(0.3)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, meg, eeg):
        ## EEG Wavelet Model
        x = pywt.dwt2(eeg.cpu(), wavelet_type, 'per')
        arr = torch.tensor([x[0][:], x[1][0][:], x[1][1][:], x[1][2][:]])
        inArr = []
        for i in arr: #each sample consists of 4 32x50 matrices- 1st is approximation, rest are detail #DOWNSAMPLE TO DIFFERENT BANDS???            
            outApprox = pywt.dwt2(i, wavelet_type, 'per')
            inArr.append(outApprox[0][:])
            inArr.append(outApprox[1][0][:])
            inArr.append(outApprox[1][1][:])
            inArr.append(outApprox[1][2][:])
        x = torch.tensor(inArr, device=self.device)
        x = x.permute(1, 2, 0, 3) #shape: 32, 16, 16, 25
        x = self.eeg_layer_conv(x) #shape: 32, 8, 12, 22, flatten shape: 32,2288, 2288 subject to batch size, wavelet, convs
        x = x.view(x.size(0), -1)
        x = self.eeg_layer_fc(x)
        ##MEG
        y = self.meg_layer_conv(meg) #out shape: 32, 32, 141
        y = y.view(y.size(0), -1) #shape 32, 32*141
        y = self.meg_layer_fc(y)

        z = torch.cat((x,y), dim=0)
        z = self.d1(self.feat_select(z))
        z = self.classifer(z)
        return z