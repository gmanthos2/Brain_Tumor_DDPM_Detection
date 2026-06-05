"""
Precompute VAE latent representations for DDPM training.

Encodes all training images through the trained VAE encoder and saves
the latent tensors as .pt files. This avoids re-encoding during DDPM
training, significantly speeding up iteration.
"""

import argparse
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
    """Encode all images to latent space and save as .pt files."""
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
    print(f"Output directory: {output_path}")

    count = 0
    with torch.no_grad():
        for images, labels, filenames in tqdm(loader, desc="Encoding"):
            images = images.to(device)

            # Encode to latent mean (deterministic)
            latents = vae.encode_to_latent(images)

            # Save individual latent tensors
            for i, (latent, filename) in enumerate(zip(latents, filenames)):
                stem = Path(filename).stem
                save_path = output_path / f"{stem}.pt"
                torch.save(latent.cpu(), save_path)
                count += 1

    print(f"\n✓ Precomputed {count} latent tensors")
    print(f"  Latent shape: {latents[0].shape}")
    print(f"  Saved to: {output_path}")


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
