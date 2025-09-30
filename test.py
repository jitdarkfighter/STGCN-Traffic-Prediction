# eval.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.stgcn import STGCN
from src.dataloader import STGCNDataset

# ----- USER CONFIG -----
# SHOULD BE SAME AS WHAT WAS USED IN TRAINING
NUM_TIMESTEPS_INPUT = 30
NUM_TIMESTEPS_OUTPUT = 4
NUM_NODES = 8
batch_size = 64
checkpoint_path = "checkpoints_8/stgcn_model.pth"
# tolerance for "accuracy" in original units (set None to skip accuracy)
# default: 10% of per-node std mean
tol_multiplier = 0.1
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_small_data(num_nodes=4):
    # replicate your existing loader (must exist in dataset/ folder)
    import pandas as pd
    if not os.path.exists(f"dataset/V_small_{num_nodes}.csv") or not os.path.exists(f"dataset/W_small_{num_nodes}.csv"):
        raise FileNotFoundError("Dataset CSVs not found for given num_nodes.")
    v_dataset = pd.read_csv(f"dataset/V_small_{num_nodes}.csv", header=None)
    w_dataset = pd.read_csv(f"dataset/W_small_{num_nodes}.csv", header=None)
    data_np = v_dataset.values          # [time_steps, num_nodes]
    adj_np = w_dataset.values           # [num_nodes, num_nodes]
    means = np.mean(data_np, axis=0, keepdims=True)   # (1, num_nodes)
    stds = np.std(data_np, axis=0, keepdims=True)     # (1, num_nodes)
    data_normalized = (data_np - means) / (stds + 1e-8)
    data_normalized = data_normalized.T                # (num_nodes, time_steps)
    return adj_np, data_normalized, means, stds

def get_normalized_adj(A):
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D = np.clip(D, 1e-6, None)
    diag = np.reciprocal(np.sqrt(D))
    A_hat = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    return A_hat

def evaluate(checkpoint_path, num_nodes=NUM_NODES, batch_size=batch_size):
    # load data
    A, X, means, stds = load_small_data(num_nodes)
    data_np = X.T  # [time_steps, num_nodes] for dataset constructor
    dataset = STGCNDataset(data_np, NUM_TIMESTEPS_INPUT, NUM_TIMESTEPS_OUTPUT)

    # use DataLoader for evaluation (batched)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # adjacency
    A_hat = get_normalized_adj(A)
    A_hat = torch.from_numpy(A_hat).float().to(device)

    # instantiate model with same args as training
    model = STGCN(A_hat.shape[0],      # num_nodes
                  1,                   # num_features (your input feature dim; dataset yields single feature per node)
                  NUM_TIMESTEPS_INPUT,
                  NUM_TIMESTEPS_OUTPUT).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # load weights
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    mse_loss = nn.MSELoss(reduction="mean")

    # metrics accumulators
    mse_list = []
    mae_sum = 0.0
    rmse_sum = 0.0
    mape_sum = 0.0
    total_samples = 0
    within_tol_count = 0
    total_values = 0

    # prepare unnormalization vectors
    means_flat = means.flatten()  # shape (num_nodes,)
    stds_flat = stds.flatten()    # shape (num_nodes,)

    # choose tolerance in original units
    tol = tol_multiplier * float(np.mean(stds_flat))

    with torch.no_grad():
        for bx, by in test_loader:
            # bx: [batch, his_len, num_nodes], by: [batch, pred_len, num_nodes]
            # convert to model input shape: [B, N, T, 1]
            bx = bx.permute(0, 2, 1).unsqueeze(-1).to(device)   # float tensor
            by = by.permute(0, 2, 1).to(device)                 # [B, N, pred_len]

            preds = model(A_hat, bx)                            # [B, N, pred_len] (normalized)

            # compute MSE on normalized values (for consistency)
            batch_mse = mse_loss(preds, by).item()
            mse_list.append(batch_mse)

            # move to cpu / numpy and un-normalize per node
            preds_np = preds.detach().cpu().numpy()    # (B, N, pred_len)
            by_np = by.detach().cpu().numpy()          # (B, N, pred_len)

            # unnormalize: broadcast across batch & pred_len
            preds_unnorm = preds_np * stds_flat[None, :, None] + means_flat[None, :, None]
            targets_unnorm = by_np * stds_flat[None, :, None] + means_flat[None, :, None]

            # per-sample metrics in original units
            abs_err = np.abs(preds_unnorm - targets_unnorm)    # (B, N, pred_len)
            sq_err = (preds_unnorm - targets_unnorm) ** 2

            mae_sum += np.sum(abs_err)
            rmse_sum += np.sum(sq_err)
            # MAPE: avoid division by zero by adding tiny eps
            denom = np.abs(targets_unnorm) + 1e-8
            mape_sum += np.sum(abs_err / denom)

            total_values += np.prod(abs_err.shape)

            # tolerance-based accuracy (proportion within tol)
            within_tol_count += np.sum(abs_err <= tol)

            total_samples += preds_unnorm.shape[0]

    # aggregate
    mse = float(np.mean(mse_list))
    mae = mae_sum / total_values
    rmse = np.sqrt(rmse_sum / total_values)
    mape = (mape_sum / total_values) * 100.0  # percent
    acc_within_tol = float(within_tol_count) / float(total_values)

    # print results
    print("Evaluation results on test set:")
    print(f"  MSE (normalized): {mse:.6f}")
    print(f"  MAE (original units): {mae:.6f}")
    print(f"  RMSE (original units): {rmse:.6f}")
    print(f"  MAPE (%): {mape:.4f}")
    print(f"  Accuracy within tol={tol:.6f} (orig units): {acc_within_tol*100:.4f}%")

    return {
        "mse_normalized": mse,
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
        "accuracy_within_tol": acc_within_tol,
        "tol_used": tol
    }

if __name__ == "__main__":
    results = evaluate(checkpoint_path, num_nodes=NUM_NODES, batch_size=batch_size)

    
