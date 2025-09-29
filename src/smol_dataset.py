import pandas as pd
import numpy as np

# Load full datasets
v_dataset = pd.read_csv("PeMS-M/V_228.csv", header=None)
w_dataset = pd.read_csv("PeMS-M/W_228.csv", header=None)

print("Original dataset shapes:")
print(f"Traffic data (V): {v_dataset.shape}")
print(f"Adjacency matrix (W): {w_dataset.shape}")
print()

num_nodes = 8

small_traffic = v_dataset.iloc[:, :num_nodes].copy()

small_adj = w_dataset.iloc[:num_nodes, :num_nodes].values

# Save the reduced datasets
small_traffic.to_csv(f"dataset/V_small_{num_nodes}.csv", index=False, header=False)
np.savetxt(f"dataset/W_small_{num_nodes}.csv", small_adj, delimiter=",", fmt="%.2f")

print("Small dataset created:")
print(f"Traffic data: {small_traffic.shape}")
print(f"Adjacency matrix: {small_adj.shape}")
print()
print("Sample traffic data (first 5 timesteps):")
print(small_traffic.head())
print()
print("Adjacency matrix:")
print(small_adj)
