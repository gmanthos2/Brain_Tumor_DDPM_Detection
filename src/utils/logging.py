"""
Logging utilities for training runs.
Supports TensorBoard and optional W&B logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Any

import torch


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    """Unified training logger for TensorBoard (and optionally W&B)."""

    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / experiment_name))

        self.logger = setup_logger(experiment_name, str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        self.tb_writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars under a main tag."""
        self.tb_writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Log an image tensor (C, H, W) or batch (N, C, H, W)."""
        if image.dim() == 4:
            from torchvision.utils import make_grid
            image = make_grid(image, nrow=4, normalize=True, value_range=(-1, 1))
        self.tb_writer.add_image(tag, image, step)

    def log_text(self, message: str, level: str = "info"):
        """Log a text message."""
        getattr(self.logger, level)(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def close(self):
        """Close all writers."""
        self.tb_writer.close()
