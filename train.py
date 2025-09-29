import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
from torch.utils.data import DataLoader

from src.stgcn import STGCN
from src.dataloader import STGCNDataset


NUM_TIMESTEPS_INPUT = 40
NUM_TIMESTEPS_OUTPUT = 5
NUM_NODES = 8

epochs = 150
batch_size = 50

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def load_small_data(num_nodes=4):
    """
    Load the small 4 intersection data and return normalized values.
    """

    # If data is not there in datasets folder, please generate it using smol_dataset.py
    if not os.path.exists(f"dataset/V_small_{num_nodes}.csv") or not os.path.exists(f"dataset/W_small_{num_nodes}.csv"):
        print(f"Small dataset for {num_nodes} nodes not found.")
        return None

    v_dataset = pd.read_csv(f"dataset/V_small_{num_nodes}.csv", header=None)
    w_dataset = pd.read_csv(f"dataset/W_small_{num_nodes}.csv", header=None)
    
    data_np = v_dataset.values  # [time_steps, num_nodes]
    adj_np = w_dataset.values   # [num_nodes, num_nodes]
    
    print(f"Data shape: {data_np.shape}")
    print(f"Adjacency matrix shape: {adj_np.shape}")
    
    # Ensure adjacency matrix is square
    if adj_np.shape[0] != adj_np.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {adj_np.shape}")
    
    # Calculate mean and std for normalization
    means = np.mean(data_np, axis=0, keepdims=True)
    stds = np.std(data_np, axis=0, keepdims=True)
    
    # Normalize the data
    data_normalized = (data_np - means) / stds
    
    # Transpose to match expected format [num_nodes, time_steps]
    data_normalized = data_normalized.T
    
    return adj_np, data_normalized, means, stds


def generate_dataset_from_loader(data_loader, device):
    """
    Convert dataloader batches to single tensors for compatibility.
    """
    all_inputs = []
    all_targets = []
    
    for batch_x, batch_y in data_loader:
        # batch_x: [batch_size, his_len, num_nodes]
        # batch_y: [batch_size, pred_len, num_nodes]
        
        # Reshape to match expected format: [batch_size, num_nodes, timesteps, features]
        batch_x = batch_x.permute(0, 2, 1).unsqueeze(-1)  # [batch_size, num_nodes, his_len, 1]
        batch_y = batch_y.permute(0, 2, 1)  # [batch_size, num_nodes, pred_len]
        
        all_inputs.append(batch_x)
        all_targets.append(batch_y)
    
    training_input = torch.cat(all_inputs, dim=0).to(device)
    training_target = torch.cat(all_targets, dim=0).to(device)
    
    return training_input, training_target


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=device)
        y_batch = y_batch.to(device=device)

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':
    torch.manual_seed(7)

    # Load the small 4 intersection data
    A, X, means, stds = load_small_data(NUM_NODES)

    # Create dataset using the dataloader
    data_np = X.T  # Convert back to [time_steps, num_nodes] for dataset
    stgcn_dataset = STGCNDataset(data_np, NUM_TIMESTEPS_INPUT, NUM_TIMESTEPS_OUTPUT)
    
    # Split data into train/val/test (60%/20%/20%)
    total_samples = len(stgcn_dataset)
    train_size = int(total_samples * 0.6)
    val_size = int(total_samples * 0.2)
    test_size = total_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        stgcn_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(7)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Convert loaders to tensors for compatibility with original code
    training_input, training_target = generate_dataset_from_loader(train_loader, device)
    val_input, val_target = generate_dataset_from_loader(val_loader, device)
    test_input, test_target = generate_dataset_from_loader(test_loader, device)

    # Normalize adjacency matrix
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave).float()
    A_wave = A_wave.to(device=device)

    # Initialize model
    net = STGCN(A_wave.shape[0],
                training_input.shape[3],  # num_features
                NUM_TIMESTEPS_INPUT,
                NUM_TIMESTEPS_OUTPUT).to(device=device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_maes = []
    
    print(f"Training on {training_input.shape[0]} samples")
    print(f"Validation on {val_input.shape[0]} samples")
    print(f"Model parameters: {sum(p.numel() for p in net.parameters())}")
    
    for epoch in range(epochs):
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        training_losses.append(loss)

        # Run validation
        with torch.no_grad():
            net.eval()
            val_input = val_input.to(device=device)
            val_target = val_target.to(device=device)

            out = net(A_wave, val_input)
            val_loss = loss_criterion(out, val_target).to(device="cpu")
            validation_losses.append(val_loss.detach().numpy().item())

            out_unnormalized = out.detach().cpu().numpy()*stds.T+means.T
            target_unnormalized = val_target.detach().cpu().numpy()*stds.T+means.T
            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            validation_maes.append(mae)

            out = None
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")

        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Validation MAE: {}".format(validation_maes[-1]))
        
        # Plot every 5 epochs for a quick test
        if epoch % 5 == 0:
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(training_losses, label="training loss")
                plt.plot(validation_losses, label="validation loss")
                plt.legend()
                plt.title(f"Loss curves - Epoch {epoch}")
                plt.savefig(f"checkpoints/loss_epoch_{epoch}.png")
                plt.close()  # Close the figure to avoid memory issues
                print(f"Saved loss plot for epoch {epoch}")
            except Exception as e:
                print(f"Could not save plot: {e}")

        checkpoint_path = "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open("checkpoints/losses.pk", "wb") as fd:
            pk.dump((training_losses, validation_losses, validation_maes), fd)
    
    # Save final model
    torch.save(net.state_dict(), "checkpoints/stgcn_model.pth")
    print("Training completed! Model saved to checkpoints/stgcn_model.pth")