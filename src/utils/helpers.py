"""
Miscellaneous helper utilities.
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: deterministic mode is NOT set here by default as it
    # significantly slows training. Enable explicitly if needed.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(num_params: int) -> str:
    """Format parameter count into human-readable string."""
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.1f}K"
    return str(num_params)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step_or_epoch: int,
    loss: float,
    save_path: str | Path,
    ema_model: Optional[torch.nn.Module] = None,
    extra: Optional[dict] = None,
):
    """Save a training checkpoint."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step_or_epoch": step_or_epoch,
        "loss": loss,
    }
    if ema_model is not None:
        checkpoint["ema_model_state_dict"] = ema_model.state_dict()
    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[torch.nn.Module] = None,
    device: str = "cpu",
) -> dict:
    """Load a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if ema_model is not None and "ema_model_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
    return checkpoint


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999):
    """Update exponential moving average model weights using fused in-place lerp."""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.lerp_(param.data, 1.0 - decay)
    for ema_buf, buf in zip(ema_model.buffers(), model.buffers()):
        ema_buf.data.copy_(buf.data)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
