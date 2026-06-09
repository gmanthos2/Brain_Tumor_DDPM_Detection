"""
Synthetic brain MRI generation pipeline.

Generates high-quality synthetic brain MRI images by sampling from the
learned healthy distribution using DDIM with Classifier-Free Guidance.
"""

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.vae import build_vae
from src.models.unet import build_unet
from src.models.scheduler import NoiseScheduler
from src.models.diffusion import GaussianDiffusion
from src.utils.config import load_config
from src.utils.helpers import get_device, ensure_dir, set_seed


class SyntheticGenerator:
    """
    Generates synthetic brain MRI images from the learned distribution.

    Handles latent denormalization: the DDPM operates in normalized
    latent space ~N(0,1), but the VAE decoder expects raw-scale latents.
    """

    def __init__(
        self,
        vae_config_path: str,
        ddpm_config_path: str,
        vae_checkpoint_path: str,
        ddpm_checkpoint_path: str,
        latent_stats_path: str = None,
        device: torch.device = None,
    ):
        self.device = device or get_device()

        # Load configs
        vae_config = load_config(vae_config_path)
        ddpm_config = load_config(ddpm_config_path)

        # Load VAE decoder
        self.vae = build_vae(vae_config).to(self.device)
        vae_ckpt = torch.load(vae_checkpoint_path, map_location=self.device, weights_only=False)
        self.vae.load_state_dict(
            vae_ckpt["model_state_dict"] if "model_state_dict" in vae_ckpt else vae_ckpt
        )
        self.vae.eval()

        # Load U-Net (EMA weights)
        unet = build_unet(ddpm_config).to(self.device)
        ddpm_ckpt = torch.load(ddpm_checkpoint_path, map_location=self.device, weights_only=False)
        unet_state = ddpm_ckpt.get("ema_model_state_dict",
                                    ddpm_ckpt.get("model_state_dict", ddpm_ckpt))
        unet.load_state_dict(unet_state)
        unet.eval()

        # Build scheduler and diffusion
        self.scheduler = NoiseScheduler(
            num_timesteps=ddpm_config.diffusion.num_timesteps,
            schedule_type=ddpm_config.diffusion.schedule_type,
            beta_start=ddpm_config.diffusion.beta_start,
            beta_end=ddpm_config.diffusion.beta_end,
            device=str(self.device),
        )

        self.diffusion = GaussianDiffusion(
            model=unet,
            scheduler=self.scheduler,
            num_classes=ddpm_config.model.num_classes,
        ).to(self.device)
        self.diffusion.eval()

        # Load latent normalization stats for denormalization
        if latent_stats_path is None:
            latent_stats_path = str(project_root / "data" / "latents" / "latent_stats.pt")
        stats = torch.load(latent_stats_path, map_location=self.device, weights_only=True)
        self.latent_mean = stats["mean"].view(1, -1, 1, 1).to(self.device)  # (1, C, 1, 1)
        self.latent_std = stats["std"].view(1, -1, 1, 1).to(self.device)    # (1, C, 1, 1)
        print(f"Loaded latent stats: mean={stats['mean'].tolist()}, std={stats['std'].tolist()}")

    def denormalize_latents(self, z_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized latents back to raw VAE latent scale."""
        return z_normalized * self.latent_std + self.latent_mean

    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 16,
        guidance_scale: float = 3.0,
        ddim_steps: int = 50,
        batch_size: int = 8,
        seed: int = None,
    ) -> torch.Tensor:
        """
        Generate synthetic brain MRI images.

        Args:
            num_samples: Total number of images to generate
            guidance_scale: CFG guidance scale
            ddim_steps: Number of DDIM sampling steps
            batch_size: Generation batch size
            seed: Optional random seed for reproducibility

        Returns:
            Generated images tensor (N, 1, 256, 256), values in [-1, 1]
        """
        if seed is not None:
            set_seed(seed)

        all_images = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Generating"):
            current_batch = min(batch_size, num_samples - i * batch_size)
            shape = (current_batch, 4, 32, 32)

            # Condition on healthy class
            class_labels = torch.zeros(current_batch, device=self.device, dtype=torch.long)

            # Sample latents via DDIM
            latents = self.diffusion.ddim_sample(
                shape=shape,
                class_labels=class_labels,
                guidance_scale=guidance_scale,
                num_steps=ddim_steps,
                device=self.device,
            )

            # Denormalize from DDPM space → raw VAE latent scale
            latents = self.denormalize_latents(latents)

            # Decode to image space
            images = self.vae.decode(latents)
            all_images.append(images.cpu())

        return torch.cat(all_images, dim=0)[:num_samples]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic brain MRI images")
    parser.add_argument("--output", type=str, default="results/synthetic", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of images to generate")
    parser.add_argument("--vae-config", type=str, default="configs/vae_config.yaml")
    parser.add_argument("--ddpm-config", type=str, default="configs/ddpm_config.yaml")
    parser.add_argument("--vae-checkpoint", type=str, default="checkpoints/vae/best.pt")
    parser.add_argument("--ddpm-checkpoint", type=str, default="checkpoints/ddpm/best.pt")
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-grid", action="store_true", help="Also save a grid of samples")
    args = parser.parse_args()

    generator = SyntheticGenerator(
        vae_config_path=str(project_root / args.vae_config),
        ddpm_config_path=str(project_root / args.ddpm_config),
        vae_checkpoint_path=str(project_root / args.vae_checkpoint),
        ddpm_checkpoint_path=str(project_root / args.ddpm_checkpoint),
    )

    output_dir = ensure_dir(project_root / args.output)

    print(f"Generating {args.num_samples} synthetic brain MRIs...")
    images = generator.generate(
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        ddim_steps=args.ddim_steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Save individual images
    individual_dir = ensure_dir(output_dir / "individual")
    for i, img in enumerate(images):
        save_image(
            img,
            individual_dir / f"synthetic_{i:04d}.png",
            normalize=True,
            value_range=(-1, 1),
        )

    # Save grid
    if args.save_grid:
        grid_samples = min(64, len(images))
        save_image(
            images[:grid_samples],
            output_dir / "sample_grid.png",
            nrow=8,
            normalize=True,
            value_range=(-1, 1),
        )

    print(f"\n✓ Generated {len(images)} synthetic images")
    print(f"  Saved to: {output_dir}")

    # Print statistics
    print(f"\n  Image shape: {images[0].shape}")
    print(f"  Value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Mean: {images.mean():.3f}")
    print(f"  Std: {images.std():.3f}")


if __name__ == "__main__":
    main()
