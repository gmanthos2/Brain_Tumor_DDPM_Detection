"""
Precompute VAE latent representations for DDPM training.

Encodes all training images through the trained VAE encoder, computes
dataset-wide normalization statistics, normalizes to ~N(0,1), and saves
the latent tensors as .pt files.

The normalization is critical: raw VAE latents have std≈7 which breaks
the DDPM noise schedule (designed for unit-variance data). Normalizing
to ~N(0,1) ensures the noise schedule properly corrupts the signal.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.vae import build_vae
from src.data.dataset import BrainMRIDataset
from src.utils.config import load_config
from src.utils.helpers import get_device, load_checkpoint, ensure_dir


def precompute_latents(
    vae_config_path: str,
    vae_checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
):
    """Encode all images to latent space, normalize, and save as .pt files."""
    device = get_device()

    # Load VAE config and model
    vae_config = load_config(vae_config_path)
    vae = build_vae(vae_config).to(device)

    # Load trained weights
    checkpoint = torch.load(vae_checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        vae.load_state_dict(checkpoint["model_state_dict"])
    else:
        vae.load_state_dict(checkpoint)
    vae.eval()
    print(f"Loaded VAE checkpoint from {vae_checkpoint_path}")

    # Dataset
    dataset = BrainMRIDataset(
        data_dir=data_dir,
        image_size=vae_config.data.image_size,
        augment=False,
        label=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    output_path = ensure_dir(output_dir)
    print(f"Encoding {len(dataset)} images to latent space...")

    # ── Pass 1: Encode all images, collect raw latents ──────────
    all_latents = []
    all_filenames = []

    with torch.no_grad():
        for images, labels, filenames in tqdm(loader, desc="Encoding"):
            images = images.to(device)
            latents = vae.encode_to_latent(images)
            all_latents.append(latents.cpu())
            all_filenames.extend(filenames)

    all_latents = torch.cat(all_latents, dim=0)
    print(f"\nRaw latent statistics:")
    print(f"  Shape: {all_latents.shape}")
    print(f"  Mean:  {all_latents.mean():.4f}")
    print(f"  Std:   {all_latents.std():.4f}")
    print(f"  Range: [{all_latents.min():.3f}, {all_latents.max():.3f}]")

    # ── Compute per-channel normalization stats ─────────────────
    # Normalize to approximately N(0,1) so DDPM noise schedule works
    latent_mean = all_latents.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
    latent_std = all_latents.std(dim=(0, 2, 3), keepdim=True)    # (1, C, 1, 1)

    print(f"\nPer-channel normalization:")
    for c in range(latent_mean.shape[1]):
        print(f"  Ch {c}: mean={latent_mean[0, c, 0, 0]:.4f}, std={latent_std[0, c, 0, 0]:.4f}")

    # ── Normalize and save ──────────────────────────────────────
    normalized_latents = (all_latents - latent_mean) / latent_std

    print(f"\nNormalized latent statistics:")
    print(f"  Mean:  {normalized_latents.mean():.4f}")
    print(f"  Std:   {normalized_latents.std():.4f}")
    print(f"  Range: [{normalized_latents.min():.3f}, {normalized_latents.max():.3f}]")

    # Save ALL latents as a single consolidated file (fast loading, no per-file overhead)
    consolidated_path = output_path / "all_latents.pt"
    torch.save(normalized_latents, consolidated_path)
    print(f"\n  Consolidated tensor saved: {consolidated_path} ({normalized_latents.nbytes / 1e6:.1f} MB)")

    # Save normalization stats (needed for inference denormalization)
    stats = {
        "latent_mean": latent_mean.squeeze().tolist(),  # [C]
        "latent_std": latent_std.squeeze().tolist(),     # [C]
    }
    stats_path = output_path / "latent_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Also save as a tensor for easy loading
    torch.save({
        "mean": latent_mean.squeeze(),  # (C,)
        "std": latent_std.squeeze(),    # (C,)
    }, output_path / "latent_stats.pt")

    print(f"\n✓ Precomputed {len(all_filenames)} normalized latent tensors")
    print(f"  Saved to: {output_path}")
    print(f"  Stats saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Precompute VAE latents for DDPM training")
    parser.add_argument(
        "--vae-config", type=str, default="configs/vae_config.yaml",
        help="Path to VAE config"
    )
    parser.add_argument(
        "--vae-checkpoint", type=str, default="checkpoints/vae/best.pt",
        help="Path to trained VAE checkpoint"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/processed/train/healthy",
        help="Path to training images"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/latents",
        help="Output directory for latent tensors"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    precompute_latents(
        vae_config_path=str(project_root / args.vae_config),
        vae_checkpoint_path=str(project_root / args.vae_checkpoint),
        data_dir=str(project_root / args.data_dir),
        output_dir=str(project_root / args.output_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
