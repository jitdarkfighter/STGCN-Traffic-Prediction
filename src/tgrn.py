import torch
import torch.nn as nn
import torch.nn.functional as F

class TGRN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate=0.3):
        super(TGRN, self).__init__()
        # kernel_size -> Size of 1D temporal kernel
        padding = (kernel_size - 1) // 2  # Same padding to maintain time dimension
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding))
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding))
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Input: [batch_size, num_nodes, num_timesteps, in_channels]
        # Convert to NCHW: [batch_size, in_channels, num_nodes, num_timesteps]
        x = x.permute(0, 3, 1, 2)

        h = self.conv1(x) * torch.sigmoid(self.conv2(x))
        h = F.relu(h)
        h = self.dropout(h)  

        # Recurrent connection
        out = h + self.conv3(x)

        # Convert back to original format: [batch_size, num_nodes, num_timesteps, out_channels]
        out = out.permute(0, 2, 3, 1)

        return out
