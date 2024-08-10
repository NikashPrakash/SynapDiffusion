import torch
from torch import nn
# import pdb

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
        self.pool = nn.MaxPool2d(kernel_size=(pool, 1), stride=(filter_length//3, 1),ceil_mode=True)
        self.weights = nn.Parameter(torch.randn(271, n_ls))
        
        if conv_type == 'var':
            self.conv = nn.Conv1d(n_ls, n_ls, kernel_size=filter_length, stride=stride,padding='same')
        elif conv_type == 'lf':
            self.conv = nn.Conv2d(n_ls, n_ls, kernel_size=(filter_length, 1), stride=(stride, 1),padding='same',groups=n_ls) #shape:filter,1,n_ls,1: 9,1,1,1

    def forward(self, x: torch.Tensor):
        x_reduced = torch.einsum('bct,cs->bst', x, self.weights) #x.shpe=batch, 271, 281, x_redu.shpe=batch,n_ls,281
        if 'var' == self.conv_type:
            conv_ = self.nonlin_out()(self.conv(x_reduced))
            conv_ = torch.unsqueeze(conv_,-1)
        elif 'lf' == self.conv_type:
            x_reduced = x_reduced.unsqueeze(-2).to(memory_format=torch.channels_last) #x_red.shap = batch, 64, 1, 281: batch, feature, groups,time
            conv_ = self.nonlin_out()(self.conv(x_reduced)).permute(0, 1, 3, 2)
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
