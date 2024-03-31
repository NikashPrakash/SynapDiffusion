import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.utils.data import Dataset

class Dense(nn.Module):
    def __init__(self, size, dropout=0.5, nonlin=nn.Identity()):
        super(Dense, self).__init__()
        self.size = size
        self.dropout = nn.Dropout(dropout)
        self.nonlin = nonlin
        self.dense = None

    def forward(self, x):
        if self.dense is None:
            flat_size = x.view(x.size(0), -1).shape[1]
            self.dense = nn.Linear(flat_size, self.size)
            self.to(x.device)  # Move new layer parameters to the device of x
        x = self.dropout(x)
        return self.nonlin(self.dense(x))

class ConvLayer(nn.Module):
    def __init__(self, n_ls=32, nonlin_in=nn.Identity(), nonlin_out=nn.ReLU(),
                 filter_length=7, stride=1, pool=2, conv_type='var'):
        super().__init__()
        self.nonlin_in = nonlin_in
        self.nonlin_out = nonlin_out
        if conv_type == 'var':
            self.conv = nn.Conv1d(1, n_ls, kernel_size=filter_length, stride=stride)
        elif conv_type == 'lf':
            self.conv = nn.Conv2d(1, n_ls, kernel_size=(filter_length, 1), stride=(stride, 1))
        self.pool = nn.MaxPool2d(kernel_size=(pool, 1), stride=(filter_length//3, 1))

    def forward(self, x):
        x = self.nonlin_in(x)
        x = self.conv(x)
        x = self.nonlin_out(x)
        x = self.pool(x)
        return x

def spatial_dropout(x, keep_prob):
    # PyTorch's dropout works differently than TensorFlow's spatial_dropout
    # It automatically adjusts for the training/testing phase
    return F.dropout2d(x, 1 - keep_prob)

class ConvDSV(nn.Module):
    def __init__(self, n_ls=None, nonlin_out=nn.ReLU(), inch=None, 
                 domain=None, padding='SAME', filter_length=5, stride=1, 
                 pooling=2, dropout=0.5, conv_type='depthwise'):
        super(ConvDSV, self).__init__()
        self.nonlin_out = nonlin_out
        if conv_type == 'depthwise':
            self.conv = nn.Conv2d(inch, inch * n_ls, kernel_size=filter_length, stride=stride, groups=inch, padding=padding)
        elif conv_type == 'separable':
            self.depthwise_conv = nn.Conv2d(inch, inch, kernel_size=filter_length, stride=stride, groups=inch, padding=padding)
            self.pointwise_conv = nn.Conv2d(inch, n_ls, kernel_size=1)
        elif conv_type == '2d':
            self.conv = nn.Conv2d(inch, n_ls, kernel_size=filter_length, stride=stride, padding=padding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if hasattr(self, 'depthwise_conv'):
            x = self.depthwise_conv(x)
            x = self.pointwise_conv(x)
        else:
            x = self.conv(x)
        x = self.nonlin_out(x)
        x = self.dropout(x)
        return x

# Weight and bias initializations are handled differently in PyTorch and typically do not need to be manually specified as in TensorFlow.



class MEGDecoder(nn.Module):
    def __init__(self, h_params=None, params=None, data_paths=[], savepath=''):
        super().__init__()
        self.h_params = h_params
        self.params = params
        self.savepath = savepath
        self.dropout = params['dropout']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define model architecture
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
        self.fin_fc = Dense(size=h_params['n_classes'], 
                            nonlin=params['nonlin_out'],
                            dropout=self.dropout)
        self.to(self.device)

    def forward(self, x):
        x = self.conv(x)
        x = self.fin_fc(x)
        return x

class EEGDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): #
        super(EEGDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True,num_layers=1) # USE TEMPORAL ATTENTION INSTEAD OF LSTM, transformer or custom impl.
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=63, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=63, out_channels=math.ceil(hidden_size/2), kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        lstmOutput, (h_t,c_t) = self.lstm(input_data) #TODO DOES THIS FIX ISSUE WITH INPUT SIZE???? #
        lstmOutput = h_t.permute(1,2,0) # Dont forget this change  
        convOutput = self.relu(self.conv1(lstmOutput)) #TODO GET OTHER HYPERPARAMS FROM PAPER
        convOutput = self.relu(self.conv2(convOutput))
        convOutput = convOutput.flatten(start_dim=1)
        #convOutput = torch.max(convOutput, dim=2)[0]
        output = self.fc(convOutput)
        output = self.softmax(output)
        return output

    # def forward(self, input_data):
    #     print("Input shape:", input_data.shape)
    #     lstmOutput, _ = self.lstm(input_data)
    #     print("LSTM output shape:", lstmOutput.shape)
    #     lstmOutput = lstmOutput.permute(0, 2, 1)
    #     print("Permuted LSTM output shape:", lstmOutput.shape)
    #     convOutput = self.conv1(input_data)
    #     convOutput = self.relu(convOutput)
    #     convOutput = torch.max(convOutput, dim=2)[0]
    #     print("Conv output shape after max:", convOutput.shape)
    #     output = self.fc(convOutput)
    #     output = self.softmax(output)
    #     print("Final output shape:", output.shape)
    #     return output

    
class NeuralGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        return input_data