import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.tgrn import TGRN

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, dropout_rate=0.2):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TGRN(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=3, dropout_rate=dropout_rate)

        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TGRN(in_channels=spatial_channels,
                                   out_channels=out_channels,
                                   kernel_size=3, dropout_rate=dropout_rate)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn_spatial = nn.BatchNorm2d(spatial_channels)
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        t = self.temporal1(X)

        t = t.permute(0, 3, 2, 1).contiguous()  # (batch_size, out_channels, num_timesteps, num_nodes)
        t = self.bn1(t)
        t = t.permute(0, 3, 2, 1).contiguous()  # (batch_size, num_nodes, num_timesteps, out_channels)

        # 1st order approximation
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t2 = t2.permute(0, 3, 2, 1)
        t2 = self.bn_spatial(t2)
        t2 = t2.permute(0, 3, 2, 1)
    
        t3 = self.temporal2(t2)
        # t3 = t3.permute(0, 3, 2, 1).contiguous()  # (batch_size, out_channels, num_timesteps, num_nodes)
        # t3 = self.bn2(t3)
        # t3 = t3.permute(0, 3, 2, 1).contiguous()  # (batch_size, num_nodes, num_timesteps, out_channels)
        return t3
        # return t3


class STGCN(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        super(STGCN, self).__init__()
        
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=128,
                                 spatial_channels=64, num_nodes=num_nodes, dropout_rate=0.2)
        self.block2 = STGCNBlock(in_channels=128, out_channels=128,
                                 spatial_channels=64, num_nodes=num_nodes, dropout_rate=0.2)
        self.last_temporal = TGRN(in_channels=128, out_channels=128, kernel_size=7, dropout_rate=0.2)
        # Calculate the size after temporal convolutions
        # After 3 TGRN layers, we still have num_timesteps_input timesteps
        self.bn = nn.BatchNorm1d(num_timesteps_input * 128)
        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(num_timesteps_input * 128, num_timesteps_output)

    def forward(self, A_hat, X):
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        # out3 = self.bn(out3)

        batch_size, num_nodes, num_timesteps, channels = out3.shape

        # Flatten the last two dimensions
        # (B, N, F)
        out3 = out3.reshape(batch_size, num_nodes, -1)

        B, N, F = out3.shape
        out3 = out3.view(B*N, F)
        out3 = self.bn(out3)
        out3 = out3.view(B, N, F)

        out3 = self.dropout(out3)

        out4 = self.fc(out3)
        return out4