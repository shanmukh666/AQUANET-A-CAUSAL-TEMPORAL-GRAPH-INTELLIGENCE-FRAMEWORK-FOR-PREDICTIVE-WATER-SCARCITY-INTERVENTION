"""
modules/module1_tgnn.py
Module 1 — Temporal Graph Neural Network (T-GNN) Data Fusion

Architecture:
  - Builds a spatial graph where nodes = monitoring stations,
    edges = river connectivity + spatial proximity
  - Stacks Graph Attention layers (GATv2) for spatial message-passing
  - Adds a Temporal Transformer encoder for time-axis dependencies
  - Outputs a fused spatiotemporal embedding for each node × timestep

Why T-GNN? Standard LSTM/Transformer treats each station independently.
T-GNN shares information across spatially connected nodes — capturing
the fact that upstream rainfall affects downstream groundwater.
Published in water research: <15 papers as of 2025 (very novel).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import get_device, save_checkpoint, compute_regression_metrics, normalise


# ─────────────────────────────────────────────────────────────────────────────
# Graph Attention Layer (Spatial)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialGATBlock(nn.Module):
    """
    GATv2 block: multi-head graph attention over the basin graph.
    Each node aggregates information from connected neighbors,
    weighted by learned attention scores.
    GATv2 fixes the static attention problem of original GAT.
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=1,       # scalar edge weight (connectivity strength)
            concat=True,
        )
        self.norm = nn.LayerNorm(out_dim)
        self.ffn  = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        # Pre-norm residual (more stable than post-norm)
        h = self.norm(x)
        h = self.gat(h, edge_index, edge_attr)
        x = x + self.dropout(h)          # residual
        x = x + self.dropout(self.ffn(self.norm(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Transformer (Time-Axis)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalTransformerBlock(nn.Module):
    """
    Standard Transformer encoder block applied along the time axis.
    Each node's T-length sequence is processed independently,
    then spatial GAT mixes information across nodes.
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        # x: (batch, T, d_model)
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                         attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full T-GNN Model
# ─────────────────────────────────────────────────────────────────────────────

class TemporalGNN(nn.Module):
    """
    Temporal Graph Neural Network for water feature fusion.

    Input:
      x:          (batch, T, N, F)  — time × nodes × features
      edge_index: (2, E)             — graph edges
      edge_attr:  (E, 1)             — edge weights

    Output:
      embedding:  (batch, T, N, hidden_dim)  — fused representation
      prediction: (batch, T, N)              — CWSI prediction
    """
    def __init__(self, num_features: int, hidden_dim: int = 128,
                 num_gat_layers: int = 3, num_transformer_layers: int = 4,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Positional encoding (learnable)
        self.pos_embed = nn.Embedding(512, hidden_dim)  # max 512 timesteps

        # Temporal Transformer layers
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])

        # Spatial GAT layers (interleaved with temporal)
        self.spatial_layers = nn.ModuleList([
            SpatialGATBlock(hidden_dim, hidden_dim, num_heads // 2, dropout)
            for _ in range(num_gat_layers)
        ])

        # Output head: predict CWSI per node
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),   # CWSI in [0, 1]
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x: (B, T, N, F)
        Returns: embedding (B, T, N, H), cwsi_pred (B, T, N)
        """
        B, T, N, F = x.shape
        device = x.device

        # Project features
        h = self.input_proj(x)  # (B, T, N, H)

        # Add positional encoding (time axis)
        positions = torch.arange(T, device=device)
        pos_enc   = self.pos_embed(positions)  # (T, H)
        h = h + pos_enc.unsqueeze(0).unsqueeze(2)  # broadcast (B, T, N, H)

        # Interleave temporal and spatial processing
        num_spatial = len(self.spatial_layers)
        num_temporal = len(self.temporal_layers)

        for layer_idx in range(max(num_temporal, num_spatial)):
            # Temporal attention: reshape to (B*N, T, H)
            if layer_idx < num_temporal:
                h_t = h.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
                h_t = self.temporal_layers[layer_idx](h_t)
                h   = h_t.reshape(B, N, T, self.hidden_dim).permute(0, 2, 1, 3)

            # Spatial attention: reshape to (B*T, N, H) → graph
            if layer_idx < num_spatial:
                h_s = h.reshape(B * T, N, self.hidden_dim)
                # Process each (B*T) graph separately via batched PyG
                h_s = h_s.reshape(B * T * N, self.hidden_dim)
                # Offset edge indices for batch
                batch_edge_index = edge_index.clone()
                # Apply spatial layer (operates on single graph topology)
                h_s = self.spatial_layers[layer_idx](h_s, edge_index, edge_attr)
                h   = h_s.reshape(B, T, N, self.hidden_dim)

        embedding = h  # (B, T, N, H)
        cwsi_pred = self.output_head(h).squeeze(-1)  # (B, T, N)
        return embedding, cwsi_pred


# ─────────────────────────────────────────────────────────────────────────────
# Dataset & DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class BasinDataset(torch.utils.data.Dataset):
    """
    Sliding window dataset: for each timestep t, take a window of
    `seq_len` past months as input and predict CWSI at t+1.
    """
    def __init__(self, data: np.ndarray, cwsi: np.ndarray,
                 seq_len: int = 24, stride: int = 1):
        super().__init__()
        self.data    = torch.FloatTensor(data)   # (T, N, F)
        self.cwsi    = torch.FloatTensor(cwsi)   # (T, N)
        self.seq_len = seq_len
        self.stride  = stride
        self.indices = list(range(0, len(data) - seq_len - 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.data[t : t + self.seq_len]        # (seq_len, N, F)
        y = self.cwsi[t + 1 : t + self.seq_len + 1]  # (seq_len, N) — shifted
        return x, y


def build_graph_tensors(adj: np.ndarray, threshold: float = 0.1,
                        device=None) -> tuple:
    """Convert adjacency matrix to PyG edge_index and edge_attr."""
    adj_tensor = torch.FloatTensor(adj)
    adj_tensor[adj_tensor < threshold] = 0
    edge_index, edge_attr = dense_to_sparse(adj_tensor)
    edge_attr = edge_attr.unsqueeze(-1)  # (E, 1)
    if device:
        edge_index = edge_index.to(device)
        edge_attr  = edge_attr.to(device)
    return edge_index, edge_attr


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_tgnn(dataset_dict: dict, cfg: dict) -> tuple:
    """
    Full training pipeline for Module 1.
    Returns (trained model, embeddings array, metrics dict).
    """
    device  = get_device()
    tcfg    = cfg["tgnn"]
    seed    = cfg["project"]["seed"]

    # ─ Prepare data ─
    data_arr = dataset_dict["data"]   # (T, N, F)
    cwsi_arr = dataset_dict["cwsi"]   # (T, N)
    adj      = dataset_dict["adj"]

    # Normalise features (per-feature across time & nodes)
    T, N, F = data_arr.shape
    data_flat = data_arr.reshape(T, -1)
    data_norm, mu, std = normalise(data_flat, axis=0)
    data_norm = data_norm.reshape(T, N, F)

    # Chronological split
    train_end = int(T * 0.70)
    val_end   = int(T * 0.85)

    train_ds = BasinDataset(data_norm[:train_end], cwsi_arr[:train_end])
    val_ds   = BasinDataset(data_norm[train_end:val_end], cwsi_arr[train_end:val_end])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=tcfg["batch_size"], shuffle=False, num_workers=0)

    # ─ Build graph ─
    edge_index, edge_attr = build_graph_tensors(adj, tcfg["edge_threshold"], device)

    # ─ Build model ─
    model = TemporalGNN(
        num_features=F,
        hidden_dim=tcfg["hidden_dim"],
        num_gat_layers=tcfg["num_layers"] - 1,
        num_transformer_layers=tcfg["num_layers"],
        num_heads=tcfg["num_heads"],
        dropout=tcfg["dropout"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"T-GNN parameters: {num_params:,}")

    optimizer = AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])

    # Huber loss (robust to outliers — important for drought spikes)
    criterion = nn.HuberLoss(delta=0.1)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    logger.info("── Training Module 1: T-GNN ──────────────────────")
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)   # (B, T, N, F)
            y_batch = y_batch.to(device)   # (B, T, N)

            optimizer.zero_grad()
            _, cwsi_pred = model(x_batch, edge_index, edge_attr)
            loss = criterion(cwsi_pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                _, cwsi_pred = model(x_batch, edge_index, edge_attr)
                val_loss += criterion(cwsi_pred, y_batch).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d}/{tcfg['epochs']} | "
                        f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch,
                           {"val_loss": val_loss},
                           "models/checkpoints/tgnn_best.pt")

    # ─ Extract embeddings for all timesteps ─
    logger.info("Extracting spatiotemporal embeddings...")
    model.eval()
    all_data = BasinDataset(data_norm, cwsi_arr, seq_len=tcfg["temporal_window"])
    all_loader = torch.utils.data.DataLoader(all_data, batch_size=8, shuffle=False)

    embeddings_list = []
    with torch.no_grad():
        for x_batch, _ in all_loader:
            x_batch = x_batch.to(device)
            emb, _  = model(x_batch, edge_index, edge_attr)
            embeddings_list.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)  # (T', window, N, H)
    # Take the last timestep embedding as the summary
    embeddings = embeddings[:, -1, :, :]  # (T', N, H)

    logger.success(f"Module 1 complete. Best val loss: {best_val_loss:.4f}")
    logger.success(f"Embeddings shape: {embeddings.shape}")

    return model, embeddings, {"train_losses": train_losses, "val_losses": val_losses}
