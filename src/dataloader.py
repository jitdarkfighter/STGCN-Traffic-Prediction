import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class STGCNDataset(Dataset):
    def __init__(self, data, his_len, pred_len):
        self.data = data
        self.his_len = his_len
        self.pred_len = pred_len
        self.num_samples = data.shape[0] - his_len - pred_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # input: [his_len, num_nodes]
        x = self.data[idx : idx + self.his_len]
        # target: [pred_len, num_nodes]
        y = self.data[idx + self.his_len : idx + self.his_len + self.pred_len]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# # Example usage (can be removed or commented out)
# if __name__ == "__main__":
#     # Load data for testing
#     v_dataset = pd.read_csv("dataset/V_small_4.csv")
#     w_dataset = pd.read_csv("dataset/W_small_4.csv")
    
#     # Convert main traffic DataFrame to numpy: [time_steps, num_nodes]
#     data_np = v_dataset.values

#     # Define historical and prediction window lengths
#     n_his, n_pred = 12, 3  # e.g., use past 12 steps to predict next 3

#     # Instantiate dataset and loader
#     stgcn_dataset = STGCNDataset(data_np, n_his, n_pred)
#     stgcn_loader = DataLoader(stgcn_dataset, batch_size=64, shuffle=True)

#     # Load adjacency matrix (defines graph connectivity)
#     w = w_dataset.values
#     adj = torch.from_numpy(w).float()  # shape: [num_nodes, num_nodes]

#     # Inspect shapes
#     print(f"Dataset samples: {len(stgcn_dataset)}")
#     print(f"Batch X shape: {next(iter(stgcn_loader))[0].shape}")  # [batch, his_len, num_nodes]
#     print(f"Adjacency shape: {adj.shape}")