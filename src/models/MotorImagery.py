import torch.nn.functional as F
import torch.nn as nn
import torch

class MI_CNN(nn.Module):
    def __init__(
        self,
        init_features=64
    ):
        super().__init__()
        self.conv1 = nn.Sequential([
             nn.Conv2d(9,init_features,7,2,"same"), #64=init_features
             nn.BatchNorm2d(init_features), #64=init_features
             nn.Hardswish(True),
             nn.MaxPool2d(3,2,1)
        ])
        resB1 = ResBlock2D()
        resB2 = ResMix()
        
        maxPool1 = nn.MaxPool2d(3,2)
        avgPool = nn.AdaptiveAvgPool2d(128)
        
        classifier = nn.Linear(128,5)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.resB1(out) + x
        out = self.resB2(out + x)
        out = F.relu(out, True)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = torch.flatten(out,1)
        return out
    
class ResBlock2D(nn.Module):
    def __init__(self):
        super().__init__()
        convs = nn.ModuleList([nn.Conv2d(64,128,3,padding='same'),
                              nn.Conv2d(128,64,5,2,'same'),
                              nn.Conv2d(64,32,3,padding='same')])
        
    def forward(self, x):
        orig = x
        for conv in self.convs:
            x = orig + conv(x)
            
        return x

class ResMix(nn.Module):
    def __init__(self):
        super().__init__()
        conv1 = nn.Conv3d()
        conv2 = nn.Conv2d()
        conv3 = nn.Conv1d()
        
    def forward(self, x):
        return x

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))