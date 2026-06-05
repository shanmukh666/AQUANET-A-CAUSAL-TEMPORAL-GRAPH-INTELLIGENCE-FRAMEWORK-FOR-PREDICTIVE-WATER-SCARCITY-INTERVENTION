"""
modules/module2_mamba_forecast.py
Module 2 — Mamba SSM + Spatial Transformer for Water Stress Forecasting

Architecture:
  MambaBlock (Selective State Space Model) for long-range temporal sequences.
  Mamba achieves O(n) complexity vs O(n²) for standard Transformers,
  making it ideal for year-long daily forecasts.

  Paper: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
         Gu & Dao, 2023 (https://arxiv.org/abs/2312.00752)

  We implement a CPU-compatible Mamba approximation (no CUDA kernel required),
  then wrap it with a spatial attention layer to propagate forecasts across nodes.

Outputs: Probabilistic forecasts at 30/90/365-day horizons per node.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from loguru import logger
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import get_device, save_checkpoint, compute_regression_metrics, normalise, denormalise


# ─────────────────────────────────────────────────────────────────────────────
# Mamba Block (CPU-Compatible Selective State Space)
# ─────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    Simplified Mamba SSM block — CPU compatible (no CUDA kernels).
    Approximates the selective state space mechanism using:
      1. Input-dependent SSM parameter selection (selectivity)
      2. 1D depthwise convolution (local context)
      3. Gated activation (SiLU)

    For full CUDA Mamba: pip install mamba-ssm, then replace this with:
        from mamba_ssm import Mamba
        self.ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model  = d_model
        self.d_inner  = int(expand * d_model)
        self.d_state  = d_state

        # Input projection (splits into x and z — gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise 1D conv (local context window)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters: A, B, C, D (input-dependent B, C = "selective")
        self.x_proj  = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A_log   = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                                               .unsqueeze(0).expand(self.d_inner, -1)))
        self.D       = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def ssm(self, x):
        """
        Selective State Space: O(n) recurrence.
        x: (B, L, d_inner)
        """
        B, L, D = x.shape
        N = self.d_state

        # Input-dependent parameters (selectivity = key Mamba innovation)
        xz = self.x_proj(x)                          # (B, L, 2N+1)
        B_coef = xz[..., :N]                         # (B, L, N)
        C_coef = xz[..., N:2*N]                      # (B, L, N)
        dt     = F.softplus(self.dt_proj(xz[..., 2:3]))  # (B, L, D)

        # Discretise A via ZOH: Ā = exp(dt ⊙ A)
        A = -torch.exp(self.A_log.float())            # (D, N) — stable negative
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)

        # dB: (B, L, D, N) — outer product of dt and B
        dB = dt.unsqueeze(-1) * B_coef.unsqueeze(-2)  # (B, L, D, N)

        # Parallel scan (sequential for simplicity; replace with CUDA scan for speed)
        h = torch.zeros(B, D, N, device=x.device)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y = (h * C_coef[:, t].unsqueeze(-2)).sum(-1)  # (B, D)
            ys.append(y)
        y = torch.stack(ys, dim=1)   # (B, L, D)
        return y + x * self.D.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        # Pre-norm residual
        residual = x
        x = self.norm(x)

        # Project and split
        xz = self.in_proj(x)               # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)      # (B, L, d_inner) each

        # Depthwise conv (local mixing)
        x_in = x_in.transpose(1, 2)        # (B, d_inner, L)
        x_in = self.conv1d(x_in)[..., :x.shape[1]]
        x_in = x_in.transpose(1, 2)        # (B, L, d_inner)

        # Activation
        x_in = F.silu(x_in)

        # SSM
        y = self.ssm(x_in)

        # Gating
        y = y * F.silu(z)

        # Output projection + residual
        out = self.out_proj(self.dropout(y)) + residual
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Spatial Attention Layer
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    """
    Cross-node attention: each node attends to all other nodes
    at a given timestep, sharing forecast information spatially.
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, d_model) — operate over node dimension
        h, _ = self.attn(self.norm(x), self.norm(x), self.norm(x), need_weights=False)
        return x + self.drop(h)


# ─────────────────────────────────────────────────────────────────────────────
# Full Mamba Forecasting Model
# ─────────────────────────────────────────────────────────────────────────────

class MambaForecaster(nn.Module):
    """
    Mamba SSM + Spatial Attention for multi-horizon water stress forecasting.

    Input:
      x_feat:  (B, T_in, N, F)    — raw features (T_in months)
      x_emb:   (B, T_in, N, H)    — T-GNN embeddings from Module 1 (optional)

    Output:
      preds:   (B, N, num_horizons, 3)  — [q10, q50, q90] quantiles per horizon
    """
    def __init__(self, input_dim: int, d_model: int = 256, d_state: int = 64,
                 d_conv: int = 4, expand: int = 2, num_layers: int = 6,
                 num_nodes: int = 20, num_heads: int = 8, dropout: float = 0.1,
                 forecast_horizons: list = None):
        super().__init__()
        self.forecast_horizons = forecast_horizons or [1, 3, 12]  # months
        num_horizons = len(self.forecast_horizons)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # Positional encoding (learnable — better than sinusoidal for irregular time)
        self.pos_embed = nn.Embedding(512, d_model)

        # Stacked Mamba + Spatial Attention layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "mamba":   MambaBlock(d_model, d_state, d_conv, expand, dropout),
                "spatial": SpatialAttention(d_model, num_heads // 2, dropout),
            }))

        # Quantile output heads (one per horizon × quantile)
        # CQR (Conformalized Quantile Regression) requires separate q10, q50, q90 heads
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 3),   # [q10, q50, q90]
                nn.Sigmoid(),                  # CWSI in [0,1]
            )
            for _ in range(num_horizons)
        ])

    def forward(self, x):
        """
        x: (B, T_in, N, F)
        Returns: (B, N, num_horizons, 3) — quantile predictions
        """
        B, T, N, F = x.shape

        # Project features
        h = self.input_proj(x)  # (B, T, N, d_model)

        # Positional encoding
        pos = self.pos_embed(torch.arange(T, device=x.device))  # (T, d_model)
        h = h + pos.unsqueeze(0).unsqueeze(2)

        # Process each node's time series through Mamba,
        # then apply spatial attention across nodes
        for layer in self.layers:
            # Mamba: temporal axis — reshape (B*N, T, d_model)
            h_t = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
            h_t = layer["mamba"](h_t)
            h   = h_t.reshape(B, N, T, -1).permute(0, 2, 1, 3)  # (B, T, N, d_model)

            # Spatial: node axis — reshape (B*T, N, d_model)
            h_s = h.reshape(B * T, N, -1)
            h_s = layer["spatial"](h_s)
            h   = h_s.reshape(B, T, N, -1)

        # Use last timestep representation for forecasting
        h_last = h[:, -1, :, :]  # (B, N, d_model)

        # Predict quantiles for each horizon
        preds = torch.stack([
            head(h_last) for head in self.quantile_heads
        ], dim=2)  # (B, N, num_horizons, 3)

        # Enforce monotonicity: q10 ≤ q50 ≤ q90
        q10 = preds[..., 0]
        q50 = preds[..., 0] + F.softplus(preds[..., 1] - preds[..., 0]) * 0.5
        q90 = preds[..., 0] + F.softplus(preds[..., 2] - preds[..., 0])
        return torch.stack([q10, q50, q90], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Quantile Loss (Pinball)
# ─────────────────────────────────────────────────────────────────────────────

def quantile_loss(pred: torch.Tensor, target: torch.Tensor,
                  quantiles: list = None) -> torch.Tensor:
    """
    Pinball loss for quantile regression.
    pred:   (B, N, H, 3) — [q10, q50, q90]
    target: (B, N, H)
    """
    if quantiles is None:
        quantiles = [0.10, 0.50, 0.90]
    target = target.unsqueeze(-1).expand_as(pred)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target[..., i] - pred[..., i]
        losses.append(torch.max(q * errors, (q - 1) * errors))
    return torch.mean(torch.stack(losses))


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Horizon Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ForecastDataset(torch.utils.data.Dataset):
    """
    For each timestep, predicts CWSI at multiple future horizons.
    Handles monthly data: horizon in months.
    """
    def __init__(self, data: np.ndarray, cwsi: np.ndarray,
                 seq_len: int = 36, horizons: list = None):
        self.data     = torch.FloatTensor(data)
        self.cwsi     = torch.FloatTensor(cwsi)
        self.seq_len  = seq_len
        self.horizons = horizons or [1, 3, 12]
        max_h = max(self.horizons)
        self.valid_t  = range(seq_len, len(data) - max_h)

    def __len__(self):
        return len(self.valid_t)

    def __getitem__(self, idx):
        t = self.valid_t[idx]
        x = self.data[t - self.seq_len : t]          # (seq_len, N, F)
        y = torch.stack([
            self.cwsi[t + h - 1] for h in self.horizons
        ], dim=-1).permute(1, 0)                      # (N, num_horizons)
        return x.unsqueeze(0), y  # add batch-like dim for model


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_mamba(dataset_dict: dict, cfg: dict,
                tgnn_embeddings: np.ndarray = None) -> tuple:
    """Train the Mamba forecaster (Module 2)."""
    device = get_device()
    mcfg   = cfg["mamba"]

    data_arr = dataset_dict["data"]   # (T, N, F)
    cwsi_arr = dataset_dict["cwsi"]   # (T, N)
    T, N, F  = data_arr.shape

    # Normalise
    data_flat = data_arr.reshape(T, -1)
    data_norm, mu, std = normalise(data_flat, axis=0)
    data_norm = data_norm.reshape(T, N, F)

    # If T-GNN embeddings available, concatenate as extra features
    input_dim = F
    if tgnn_embeddings is not None:
        H = tgnn_embeddings.shape[-1]
        # Align lengths
        min_T = min(T, len(tgnn_embeddings))
        emb_slice = tgnn_embeddings[:min_T]
        data_norm = data_norm[:min_T]
        cwsi_arr  = cwsi_arr[:min_T]
        T = min_T
        # Normalise embeddings
        emb_flat = emb_slice.reshape(T, -1)
        emb_norm, _, _ = normalise(emb_flat)
        emb_norm = emb_norm.reshape(T, N, H)
        data_aug = np.concatenate([data_norm, emb_norm], axis=-1)
        input_dim = F + H
        logger.info(f"Using T-GNN embeddings: input_dim = {input_dim}")
    else:
        data_aug = data_norm

    # Horizons: convert days → months (monthly data)
    horizons_months = [max(1, h // 30) for h in mcfg["forecast_horizons"]]

    # Split
    train_end = int(T * 0.70)
    val_end   = int(T * 0.85)

    train_ds = ForecastDataset(data_aug[:train_end], cwsi_arr[:train_end],
                                seq_len=36, horizons=horizons_months)
    val_ds   = ForecastDataset(data_aug[train_end:val_end], cwsi_arr[train_end:val_end],
                                seq_len=36, horizons=horizons_months)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=mcfg["batch_size"], shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=mcfg["batch_size"], shuffle=False, num_workers=0)

    # Model
    model = MambaForecaster(
        input_dim=input_dim,
        d_model=mcfg["d_model"],
        d_state=mcfg["d_state"],
        d_conv=mcfg["d_conv"],
        expand=mcfg["expand"],
        num_layers=mcfg["num_layers"],
        num_nodes=N,
        dropout=mcfg["dropout"],
        forecast_horizons=horizons_months,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Mamba parameters: {num_params:,}")

    optimizer = AdamW(model.parameters(), lr=mcfg["lr"], weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=mcfg["lr"],
                           steps_per_epoch=len(train_loader),
                           epochs=mcfg["epochs"])

    best_val = float("inf")
    train_losses, val_losses = [], []

    logger.info("── Training Module 2: Mamba SSM ───────────────────")
    for epoch in range(1, mcfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for (x_batch,), y_batch in train_loader:
            x_batch = x_batch.squeeze(1).to(device)   # (B, seq_len, N, F)
            # Add singleton time dim if needed
            if x_batch.ndim == 3:
                x_batch = x_batch.unsqueeze(0)
            y_batch = y_batch.to(device)              # (B, N, num_horizons)

            optimizer.zero_grad()
            pred = model(x_batch)                     # (B, N, num_horizons, 3)
            loss = quantile_loss(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        epoch_loss /= max(len(train_loader), 1)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_batch,), y_batch in val_loader:
                x_batch = x_batch.squeeze(1).to(device)
                if x_batch.ndim == 3:
                    x_batch = x_batch.unsqueeze(0)
                y_batch = y_batch.to(device)
                pred    = model(x_batch)
                val_loss += quantile_loss(pred, y_batch).item()
        val_loss /= max(len(val_loader), 1)
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d}/{mcfg['epochs']} | "
                        f"Train: {epoch_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch,
                           {"val_loss": val_loss},
                           "models/checkpoints/mamba_best.pt")

    logger.success(f"Module 2 complete. Best val quantile loss: {best_val:.4f}")
    return model, {"train_losses": train_losses, "val_losses": val_losses,
                   "horizons_months": horizons_months}
