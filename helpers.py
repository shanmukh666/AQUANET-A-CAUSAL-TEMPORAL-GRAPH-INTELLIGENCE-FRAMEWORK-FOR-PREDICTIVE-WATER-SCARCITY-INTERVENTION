"""
utils/helpers.py
Shared utilities: config loading, reproducibility, logging, metrics.
"""

import os
import random
import yaml
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config and return as nested dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Create directories from paths section
    for key, rel_path in cfg.get("paths", {}).items():
        Path(rel_path).mkdir(parents=True, exist_ok=True)
    return cfg


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")


def get_device() -> torch.device:
    """Return best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (consider GPU for Mamba SSM)")
    return device


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Comprehensive regression metrics for water stress prediction."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    # Nash-Sutcliffe Efficiency — standard in hydrology
    nse  = 1 - (np.sum((y_true - y_pred) ** 2) /
                np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "NSE": nse}
    for name, val in metrics.items():
        logger.info(f"  {name}: {val:.4f}")
    return metrics


def coverage_width_criterion(y_true, y_lower, y_upper) -> Dict[str, float]:
    """
    Evaluate conformal prediction intervals.
    Coverage: fraction of true values inside [lower, upper].
    Width:    average interval width (narrower = better, given coverage ≥ 1-alpha).
    """
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    width    = np.mean(y_upper - y_lower)
    cwc      = width * (1 + np.exp(-100 * (coverage - 0.9)))  # penalises under-coverage
    return {"Coverage": coverage, "Width": width, "CWC": cwc}


# ── Data utilities ────────────────────────────────────────────────────────────

def normalise(x: np.ndarray, axis=0) -> tuple:
    """Z-score normalise, return (normalised, mean, std)."""
    mu  = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True) + 1e-8
    return (x - mu) / std, mu, std


def denormalise(x_norm: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x_norm * std + mu


def train_val_test_split(data: np.ndarray, ratios=(0.7, 0.15, 0.15)):
    """Chronological split — never shuffle time-series data!"""
    n = len(data)
    i1 = int(n * ratios[0])
    i2 = int(n * (ratios[0] + ratios[1]))
    return data[:i1], data[i1:i2], data[i2:]


# ── Checkpoint utilities ───────────────────────────────────────────────────────

def save_checkpoint(model: torch.nn.Module, optimizer, epoch: int,
                    metrics: dict, path: str) -> None:
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }, path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(model: torch.nn.Module, path: str,
                    optimizer=None) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    logger.info(f"Checkpoint loaded from {path}  (epoch {ckpt['epoch']})")
    return ckpt
