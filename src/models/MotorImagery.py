import torch
import torch.nn as nn

class MiCNN(nn.Module):
    def __init__(self):
        super().__init__()
        conv1 = nn.Conv2d(9,64,7,2,"same")
        bn1 = nn.BatchNorm2d(64)
        silu = nn.SiLU()
        resB1 = ResBlock2D()
        resB2 = ResMix()
        
        
        
        
    def forward(self, x):
        return x
    
def ResBlock2D():
    def __init__(self):
        super().__init__()
        conv1 = nn.Conv2d()
        conv2 = nn.Conv2d()
        conv3 = nn.Conv2d()
        
    def forward(self, x):
        return x

def ResMix():
    def __init__(self):
        super().__init__()
        conv1 = nn.Conv3d()
        conv2 = nn.Conv2d()
        conv3 = nn.Conv1d()
        
    def forward(self, x):
        return x