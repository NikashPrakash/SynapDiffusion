import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
wavelet_type = 'sym9'

class WaveletCNN(nn.Module): #should take input of size Nbatch x 4 x 32 x 50, first matrix being approximation, rest being detail
    def __init__(self):
        super(WaveletCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.fc1 = nn.Linear(8 * 25 * 16, 128)
        self.fc2 = nn.Linear(128, 3)
        
        self.relu = nn.ReLU()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device) 
    
    def forward(self, x):
        x = pywt.dwt2(x, wavelet_type, 'per') #TODO DECOMPOSE THIS PROPERLY!!!
        arr = torch.tensor([x[0][:], x[1][0][:], x[1][1][:], x[1][2][:]])
        inArr = []
        for i in arr: #each sample consists of 4 32x50 matrices- 1st is approximation, rest are detail #TODO DOWNSAMPLE TO DIFFERENT BANDS???            
            outApprox = pywt.dwt2(i[0], wavelet_type, 'per')
            inArr.append([outApprox[0][:]])
            inArr.append(outApprox[1][0][:])
            inArr.append(outApprox[1][1][:])
            inArr.append(outApprox[1][2][:])
        x = torch.tensor(inArr)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, 4 * 25 * 16)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
