import torch
from torch import nn, Tensor
import torch.nn.functional as F
from gat_optim_conv import GAToptConv
from typing import Optional, Tuple, Union
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax
)
from torch_geometric.utils.sparse import set_sparse_value
import typing
if typing.TYPE_CHECKING:
    from typing import overload
    
class MultiKernelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels_list):
        super(MultiKernelConvBlock, self).__init__()
        self.convs = nn.ModuleList()
        for out_channels in out_channels_list:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        self.output_dim = sum(out_channels_list)

    def forward(self, x):
        # Apply each convolutional set and concatenate the outputs
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        return x

class TransformerLearningBlock(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads=8, num_layers=2, dropout=0.1):
        super(TransformerLearningBlock, self).__init__()
        self.conv_block = MultiKernelConvBlock(input_dim, model_dim)
        
        self.positional_embedding = nn.Parameter(torch.randn(1, sum(model_dim), 1))
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=sum(model_dim), nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):
        x = self.conv_block(x)
        x += self.positional_embedding
        x = x.permute(2, 0, 1)  # Reshape for the transformer (seq_len, batch, features)
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
    def __init__(self, input_dim=63, model_dim=[32, 64, 128], num_heads=8, num_layers=2, T_r=100, C=224, N=63, top_k=10, dropout=0.1):
        super(STGATE, self).__init__()
        
        # Transformer Learning Block (adapt the input/output dimensions as needed)
        self.transformer_learning_block = TransformerLearningBlock(input_dim,  model_dim, num_heads, num_layers, dropout)
        
        # Spatial-Temporal Graph Attention
        self.spatial_attention = SpatialAttention(T_r, C, N)
        self.temporal_attention = TemporalAttention(T_r, C, N)
        self.top_k = top_k

        # Graph Convolution Network or any other GNN layer should be added here if needed
        #GATv2: attention for two nodes could result in a similar score, especially if a node has a edge to itself, one way to resolve is fix attention score of query node i to 1
        # Subtraction not possible so instead of orig update func, use a different transform for query node i, original update for all nodes but i: h′i = b + Θ_n h˜i + SUM_{j∈N_i,i̸=j} α_ij Θ_L h˜_j
        #   Another way is to use the Θ_r (starts in scoreing function), in the update equation (5) instead of Θ_n
        #   then if update func is only 0 if h_i = 0
        #   worth testing Θ_n^+, Θ_r^+ for classification
            # Θ_n^+ for regression (generation)
        
        self.gcn = GAToptConv(self.transformer_learning_block.output_dim, C)
        
        # Final fully connected layer for emotion prediction
        num_classes = 3 # for the 3 high level object classes
        self.fc = nn.Linear(C, num_classes)

    def forward(self, x):
        # Pass input through the transformer learning block
        transformer_out = self.transformer_learning_block(x)
        
        # Prepare the output of the transformer block for the graph attention network
        # This may involve reshaping or selecting features to match expected input dimensions
        
        # Apply spatial attention
        A = self.spatial_attention(x)
        A_topk = torch.topk(A, self.top_k, dim=2).values
        A = torch.where(A >= A_topk, A, torch.zeros_like(A))
        # Apply temporal attention
        T_hat = self.temporal_attention(x)
        X_h_hat = T_hat * x
        
        # Graph Convolution operation
        gcn_out = self.gcn(X_h_hat, A)  # Uncomment and adjust if using a GNN layer
        
        # Flatten the output and pass it through the final FC layer for emotion prediction
        flattened_out = torch.flatten(gcn_out, start_dim=1)
        emotion_prediction = self.fc(flattened_out)

        return emotion_prediction