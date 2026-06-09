"""
PyTorch Dataset classes for brain MRI images and precomputed latents.
"""

import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class BrainMRIDataset(Dataset):
    """
    PyTorch Dataset for preprocessed brain MRI images.

    Loads grayscale PNG images, normalizes to [-1, 1], and applies
    optional data augmentation.

    Args:
        data_dir: Path to directory containing images (e.g., data/processed/train/healthy)
        image_size: Target image size (images should already be this size)
        augment: Whether to apply data augmentation (training only)
        label: Fixed label for all images in this directory (0=healthy, 1=anomalous)
    """

    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = 256,
        augment: bool = False,
        label: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.label = label

        # Gather image files
        self.image_files = sorted([
            f for f in self.data_dir.iterdir()
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {data_dir}")

        # Build transforms
        transform_list = []

        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=5,           # Slight rotation
                    translate=(0.02, 0.02),  # Small translation
                    scale=(0.98, 1.02),  # Slight scale variation
                ),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
            ])

        transform_list.extend([
            transforms.Resize(image_size),
            transforms.ToTensor(),          # [0, 1]
            transforms.Normalize([0.5], [0.5]),  # [-1, 1]
        ])

        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            image: Tensor of shape (1, H, W), values in [-1, 1]
            label: 0 for healthy, 1 for anomalous
            filename: Original filename (without path)
        """
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("L")  # Ensure grayscale
        image = self.transform(image)
        return image, self.label, img_path.name


class LatentDataset(Dataset):
    """
    PyTorch Dataset for precomputed VAE latent tensors.

    Loads all latents from a single consolidated .pt file for maximum
    throughput. One torch.load call for ~17MB, then pure tensor indexing.

    Args:
        latent_dir: Path to directory containing all_latents.pt
        label: Fixed label for all latents (0=healthy)
    """

    def __init__(self, latent_dir: str | Path, label: int = 0):
        self.latent_dir = Path(latent_dir)
        self.label = label

        consolidated_path = self.latent_dir / "all_latents.pt"
        if consolidated_path.exists():
            # Single-file loading: one torch.load, ~17MB, instant
            self.latents = torch.load(consolidated_path, weights_only=True)
        else:
            # Fallback: load individual .pt files (slower)
            latent_files = sorted([
                f for f in self.latent_dir.iterdir()
                if f.suffix == '.pt' and f.stem not in ('latent_stats', 'all_latents')
            ])
            if len(latent_files) == 0:
                raise ValueError(f"No latent files found in {latent_dir}")
            self.latents = torch.stack([
                torch.load(f, weights_only=True) for f in latent_files
            ])

        self.labels = torch.full((len(self.latents),), label, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            latent: Tensor of shape (C, H, W), typically (4, 32, 32)
            label: 0 for healthy
        """
        return self.latents[idx], self.labels[idx]


class CombinedTestDataset(Dataset):
    """
    Combined dataset for evaluation: loads both healthy and anomalous images.
    Used for computing AUROC and other binary classification metrics.
    """

    def __init__(
        self,
        healthy_dir: str | Path,
        anomalous_dir: str | Path,
        image_size: int = 256,
    ):
        self.healthy_dataset = BrainMRIDataset(
            healthy_dir, image_size=image_size, augment=False, label=0
        )
        self.anomalous_dataset = BrainMRIDataset(
            anomalous_dir, image_size=image_size, augment=False, label=1
        )

    def __len__(self) -> int:
        return len(self.healthy_dataset) + len(self.anomalous_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        if idx < len(self.healthy_dataset):
            return self.healthy_dataset[idx]
        else:
            return self.anomalous_dataset[idx - len(self.healthy_dataset)]


def create_dataloaders(
    train_dir: str | Path,
    val_dir: str | Path,
    image_size: int = 256,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders for healthy brain MRIs."""

    train_dataset = BrainMRIDataset(
        train_dir, image_size=image_size, augment=True, label=0
    )
    val_dataset = BrainMRIDataset(
        val_dir, image_size=image_size, augment=False, label=0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
