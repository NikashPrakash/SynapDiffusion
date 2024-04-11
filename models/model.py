import torch
from torch import nn
import math, pdb
import torch.nn.functional as F
from torch.utils.data import Dataset

class Dense(nn.Module):
    def __init__(self, in_features, out_feat, dropout=0.5, nonlin=nn.Identity()):
        super(Dense, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.nonlin = nonlin
        self.dense = nn.Linear(in_features, out_feat)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)  # Flatten the input if necessary
        x = self.dropout(x)
        x = self.nonlin()(self.dense(x))
        return x

class ConvLayer(nn.Module):
    def __init__(self, n_ls=32, nonlin_in=nn.Identity(), nonlin_out=nn.ReLU(),
                 filter_length=7, stride=1, pool=2, conv_type='var'):
        super().__init__()
        self.nonlin_in = nonlin_in
        self.nonlin_out = nonlin_out
        self.conv_type = conv_type
        self.pool = nn.MaxPool2d(kernel_size=(pool, 1), stride=(filter_length // 3, 1),ceil_mode=True)
        self.weights = nn.Parameter(torch.randn(271, n_ls))
        
        if conv_type == 'var':
            self.conv = nn.Conv1d(n_ls, n_ls, kernel_size=filter_length, stride=stride,padding='same')
        elif conv_type == 'lf':
            self.conv = nn.Conv2d(1, n_ls, kernel_size=(filter_length, 1), stride=(stride, 1),padding='same')

    def forward(self, x: torch.Tensor):
        x_reduced = torch.einsum('bct,cs->bst', x, self.weights)
        if 'var' == self.conv_type:
            conv_ = self.nonlin_out()(self.conv(x_reduced))
            conv_ = torch.unsqueeze(conv_,-1)
        elif 'lf' == self.conv_type:
            x_reduced = x_reduced.unsqueeze(-2)
            conv_ = self.nonlin_out()(self.conv(x_reduced)).permute(0, 1, 3, 2) #check if want to use
        conv_ = self.pool(conv_)
        return conv_[...,0]

class MEGDecoder(nn.Module):
    def __init__(self, h_params=None, params=None):
        super().__init__()
        if h_params['architecture'] == 'lf-cnn':
            self.conv = ConvLayer(n_ls=params['n_ls'], 
                                  filter_length=params['filter_length'],
                                  pool=params['pooling'],
                                  nonlin_in=params['nonlin_in'],
                                  nonlin_out=params['nonlin_hid'],
                                  conv_type='lf')
        elif h_params['architecture'] == 'var-cnn':
            self.conv = ConvLayer(n_ls=params['n_ls'],
                                  filter_length=params['filter_length'],
                                  pool=params['pooling'], 
                                  nonlin_in=params['nonlin_in'],
                                  nonlin_out=params['nonlin_hid'],
                                  conv_type='var')
        self.fin_fc = Dense(params['n_ls']*141,
                            out_feat=h_params['n_classes'], 
                            nonlin=params['nonlin_out'],
                            dropout=params['dropout'])

    def forward(self, x):
        return self.fin_fc(self.conv(x))

class EEGDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): #
        super(EEGDecoder, self).__init__()        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True,num_layers=1) # USE TEMPORAL ATTENTION INSTEAD OF LSTM, transformer or custom impl.
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=63, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=63, out_channels=math.ceil(hidden_size/2), kernel_size=3, padding=1)
        self.fc = nn.Linear(32, output_size)
        self.softmax = nn.Softmax(dim=1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, input_data):
        lstmOutput, (h_t,c_t) = self.lstm(input_data) #TODO DOES THIS FIX ISSUE WITH INPUT SIZE???? #
        lstmOutput = h_t.permute(1,2,0) # Dont forget this change
        convOutput = nn.functional.leaky_relu(self.conv1(lstmOutput)) #TODO GET OTHER HYPERPARAMS FROM PAPER
        convOutput = nn.functional.leaky_relu(self.conv2(convOutput))
        convOutput = convOutput.flatten(start_dim=1)
        #convOutput = torch.max(convOutput, dim=2)[0]
        output = self.fc(convOutput)
        output = self.softmax(output)
        return output

    
class NeuralGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        return input_data