import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.tgrn import TGRN

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TGRN(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=3)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TGRN(in_channels=spatial_channels,
                                   out_channels=out_channels,
                                   kernel_size=3)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TGRN(in_channels=64, out_channels=64, kernel_size=3)
        # Calculate the size after temporal convolutions
        # After 3 TGRN layers, we still have num_timesteps_input timesteps
        self.fully = nn.Linear(num_timesteps_input * 64, num_timesteps_output)

    def forward(self, A_hat, X):
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4