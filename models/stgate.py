import torch
from torch import nn
from models.gat_optim_conv import GAToptConv

    
class MultiKernelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels_list):
        super(MultiKernelConvBlock, self).__init__()
        self.convs = nn.Sequential(
                nn.Conv2d(in_channels, out_channels_list[0], kernel_size=1, padding='same'),
                nn.BatchNorm2d(out_channels_list[0]),
                nn.ReLU(),
                nn.Conv2d(out_channels_list[0], out_channels_list[1], kernel_size=3,padding='same'),
                nn.BatchNorm2d(out_channels_list[1]),
                nn.ReLU(),
                nn.Conv2d(out_channels_list[1],out_channels_list[2],3,padding='same'),
                nn.BatchNorm2d(out_channels_list[2]),
                nn.ReLU(),
            )

    def forward(self, x):
        # Apply each convolutional set and concatenate the outputs
        x = self.convs(x)
        return x

class TransformerLearningBlock(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads=8, num_layers=2, dropout=0.1):
        super(TransformerLearningBlock, self).__init__()
        self.conv_block = MultiKernelConvBlock(input_dim, model_dim)
        
        self.positional_embedding = nn.Parameter(torch.randn(1, 128, 1))
        
        encoder_layers = nn.TransformerEncoderLayer(128, nhead=num_heads, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):
        x = self.conv_block(x).squeeze(-1)
        x += self.positional_embedding
        x = x.permute(0, 2, 1)  # Reshape for the transformer (batch, seq_len, features)
        x = self.transformer_encoder(x)
        return x
    
class SpatialAttention(nn.Module):
    def __init__(self, T_r, C, N):
        super(SpatialAttention, self).__init__()
        self.V = nn.Parameter(torch.randn(N, N))
        self.W_1 = nn.Parameter(torch.randn(T_r, 1))
        self.W_2 = nn.Parameter(torch.randn(C, N))
        self.b_s = nn.Parameter(torch.randn(N, N))

    def forward(self, X_h):
        # Batch matrix multiplication
        S = torch.matmul(self.V, torch.tanh(torch.matmul(X_h, self.W_1) @ self.W_2 + self.b_s))
        dynamic_adj_mat = (S - torch.mean(S, dim=0)) / torch.sqrt(torch.var(S, dim=0, unbiased=False) + 1e-5)
        return dynamic_adj_mat

class TemporalAttention(nn.Module):
    def __init__(self, T_r, C, N):
        super(TemporalAttention, self).__init__()
        self.V = nn.Parameter(torch.randn(N, N))
        self.W_3 = nn.Parameter(torch.randn(T_r, 1))
        self.W_4 = nn.Parameter(torch.randn(C, N))
        self.b_t = nn.Parameter(torch.randn(N, N))

    def forward(self, X_h):
        T = torch.matmul(self.V, torch.tanh(torch.matmul(X_h, self.W_3) @ self.W_4 + self.b_t))
        T_normalized = (T - torch.mean(T, dim=0)) / torch.sqrt(torch.var(T, dim=0, unbiased=False) + 1e-5)
        return T_normalized

class STGATE(nn.Module):
    def __init__(self, input_dim=63, model_dim=[32, 64, 128], num_heads=8, num_layers=2, T_r=100, C=128, N=63, top_k=10, dropout=0.1):
        super(STGATE, self).__init__()
        # Transformer Learning Block (adapt the input/output dimensions as needed)
        self.transformer_learning_block = TransformerLearningBlock(input_dim,  model_dim, num_heads, num_layers, dropout)
        
        # Spatial-Temporal Graph Attention
        self.spatial_attention = SpatialAttention(T_r, C, N)
        self.temporal_attention = TemporalAttention(T_r, C, N)
        self.top_k = top_k

        # Xh ∈ RB×N×C×Tr 
        # V, bs ∈ RN×N, W1 ∈ RTr , and W2 ∈ RC×N
        # W3 ∈ RTr and W4 ∈ RC×N
        #T = Tr * BxNxCxTr * CxN
        self.gcn = GAToptConv(C, C) # C,C?
        
        num_classes = 3 # for the 3 high level object classes
        self.fc = nn.Linear(C, num_classes)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        # Pass input through the transformer learning block
        transformer_out = self.transformer_learning_block(x.unsqueeze(1))
        
        # Prepare the output of the transformer block for the graph attention network
        # This may involve reshaping or selecting features to match expected input dimensions
        
        # Apply spatial attention
        A = self.spatial_attention(transformer_out)
        A_topk = torch.topk(A, self.top_k, dim=2).values
        A = torch.where(A >= A_topk, A, torch.zeros_like(A))
        # Apply temporal attention
        T_hat = self.temporal_attention(transformer_out)
        X_h_hat = T_hat * transformer_out
        print(X_h_hat.shape)
        
        # Graph Convolution operation
        gcn_out = self.gcn(X_h_hat, A)  # Uncomment and adjust if using a GNN layer
        
        # Flatten the output and pass it through the final FC layer for emotion prediction
        flattened_out = torch.flatten(gcn_out, start_dim=1)
        emotion_prediction = self.fc(flattened_out)

        return emotion_prediction