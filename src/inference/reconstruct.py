"""
Anomaly detection via diffusion-based reconstruction.

Pipeline:
1. Encode input image → latent z_0 via VAE encoder
2. Partial noising: z_{t_start} = √ᾱ_t · z_0 + √(1-ᾱ_t) · ε
3. Denoise conditioned on "healthy" class using DDIM with CFG
4. Decode reconstructed latent → x̂ via VAE decoder
5. Compute anomaly map: |x - x̂|
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.vae import build_vae
from src.models.unet import build_unet
from src.models.scheduler import NoiseScheduler
from src.models.diffusion import GaussianDiffusion
from src.utils.config import load_config
from src.utils.helpers import get_device, ensure_dir


class AnomalyDetector:
    """
    Diffusion-based anomaly detector for brain MRIs.

    Reconstructs input images conditioned on "healthy" class,
    then identifies anomalous regions via reconstruction error.
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

        # Load VAE
        self.vae = build_vae(vae_config).to(self.device)
        vae_ckpt = torch.load(vae_checkpoint_path, map_location=self.device, weights_only=False)
        self.vae.load_state_dict(
            vae_ckpt["model_state_dict"] if "model_state_dict" in vae_ckpt else vae_ckpt
        )
        self.vae.eval()

        # Load U-Net
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

        self.ddpm_config = ddpm_config

        # Load latent normalization stats
        if latent_stats_path is None:
            latent_stats_path = str(project_root / "data" / "latents" / "latent_stats.pt")
        stats = torch.load(latent_stats_path, map_location=self.device, weights_only=True)
        self.latent_mean = stats["mean"].view(1, -1, 1, 1).to(self.device)
        self.latent_std = stats["std"].view(1, -1, 1, 1).to(self.device)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def normalize_latents(self, z_raw: torch.Tensor) -> torch.Tensor:
        """Normalize raw VAE latents to ~N(0,1) for DDPM."""
        return (z_raw - self.latent_mean) / self.latent_std

    def denormalize_latents(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize DDPM latents back to raw VAE scale."""
        return z_norm * self.latent_std + self.latent_mean

    @torch.no_grad()
    def detect(
        self,
        image_path: str,
        t_start: int = 300,
        guidance_scale: float = 3.0,
        ddim_steps: int = 50,
        gaussian_sigma: float = 2.0,
    ) -> dict:
        """
        Detect anomalies in a single brain MRI image.

        Args:
            image_path: Path to input image
            t_start: Noise level for partial noising (higher = more "correction")
            guidance_scale: CFG guidance scale
            ddim_steps: Number of DDIM sampling steps
            gaussian_sigma: Gaussian smoothing sigma for anomaly map

        Returns:
            Dictionary containing:
                - original: Original image tensor
                - reconstruction: Healthy reconstruction tensor
                - anomaly_map: Pixel-wise anomaly scores (smoothed)
                - anomaly_score: Global anomaly score (max of map)
        """
        # Load and preprocess image
        img = Image.open(image_path).convert("L")
        x = self.transform(img).unsqueeze(0).to(self.device)  # (1, 1, 256, 256)

        # Encode to latent space and normalize for DDPM
        z_0_raw = self.vae.encode_to_latent(x)
        z_0 = self.normalize_latents(z_0_raw)

        # Partial noising (in normalized space)
        t = torch.tensor([t_start], device=self.device)
        noise = torch.randn_like(z_0)
        z_t, _ = self.scheduler.q_sample(z_0, t, noise)

        # Reconstruct conditioned on "healthy" (class 0)
        class_labels = torch.zeros(1, device=self.device, dtype=torch.long)

        z_recon_norm = self.diffusion.ddim_sample(
            shape=z_0.shape,
            class_labels=class_labels,
            guidance_scale=guidance_scale,
            num_steps=ddim_steps,
            device=self.device,
            x_start=z_t,
            t_start=t_start,
        )

        # Denormalize back to raw VAE scale and decode
        z_recon = self.denormalize_latents(z_recon_norm)
        x_recon = self.vae.decode(z_recon)

        # Compute anomaly map
        diff = torch.abs(x - x_recon)
        anomaly_map_raw = diff.squeeze().cpu().numpy()

        # Gaussian smoothing
        if gaussian_sigma > 0:
            anomaly_map_raw = gaussian_filter(anomaly_map_raw, sigma=gaussian_sigma)

        # Compute anomaly score from RAW error (before normalization)
        # Using mean L1 preserves magnitude differences between healthy/anomalous
        anomaly_score = float(anomaly_map_raw.mean())

        # Normalize to [0, 1] for visualization only
        if anomaly_map_raw.max() > anomaly_map_raw.min():
            anomaly_map_vis = (anomaly_map_raw - anomaly_map_raw.min()) / (
                anomaly_map_raw.max() - anomaly_map_raw.min()
            )
        else:
            anomaly_map_vis = anomaly_map_raw

        return {
            "original": x.squeeze().cpu(),
            "reconstruction": x_recon.squeeze().cpu(),
            "anomaly_map": anomaly_map_vis,
            "anomaly_map_raw": anomaly_map_raw,
            "anomaly_score": anomaly_score,
        }

    @torch.no_grad()
    def detect_batch(
        self,
        image_dir: str,
        t_start: int = 300,
        guidance_scale: float = 3.0,
        ddim_steps: int = 50,
        gaussian_sigma: float = 2.0,
    ) -> list:
        """Detect anomalies in all images in a directory."""
        image_dir = Path(image_dir)
        image_files = sorted([
            f for f in image_dir.iterdir()
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
        ])

        results = []
        for img_path in tqdm(image_files, desc="Detecting anomalies"):
            result = self.detect(
                str(img_path),
                t_start=t_start,
                guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
                gaussian_sigma=gaussian_sigma,
            )
            result["filename"] = img_path.name
            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(description="Anomaly detection via diffusion reconstruction")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default="results/anomaly_maps", help="Output directory")
    parser.add_argument("--vae-config", type=str, default="configs/vae_config.yaml")
    parser.add_argument("--ddpm-config", type=str, default="configs/ddpm_config.yaml")
    parser.add_argument("--vae-checkpoint", type=str, default="checkpoints/vae/best.pt")
    parser.add_argument("--ddpm-checkpoint", type=str, default="checkpoints/ddpm/best.pt")
    parser.add_argument("--t-start", type=int, default=300, help="Noise level for partial noising")
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--gaussian-sigma", type=float, default=2.0)
    args = parser.parse_args()

    detector = AnomalyDetector(
        vae_config_path=str(project_root / args.vae_config),
        ddpm_config_path=str(project_root / args.ddpm_config),
        vae_checkpoint_path=str(project_root / args.vae_checkpoint),
        ddpm_checkpoint_path=str(project_root / args.ddpm_checkpoint),
    )

    input_path = project_root / args.input
    output_dir = ensure_dir(project_root / args.output)

    if input_path.is_dir():
        results = detector.detect_batch(
            str(input_path),
            t_start=args.t_start,
            guidance_scale=args.guidance_scale,
            ddim_steps=args.ddim_steps,
            gaussian_sigma=args.gaussian_sigma,
        )
    else:
        results = [detector.detect(
            str(input_path),
            t_start=args.t_start,
            guidance_scale=args.guidance_scale,
            ddim_steps=args.ddim_steps,
            gaussian_sigma=args.gaussian_sigma,
        )]
        results[0]["filename"] = input_path.name

    # Save results
    import matplotlib.pyplot as plt

    for result in results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(result["original"].numpy(), cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(result["reconstruction"].numpy(), cmap="gray")
        axes[1].set_title("Healthy Reconstruction")
        axes[1].axis("off")

        axes[2].imshow(result["anomaly_map"], cmap="hot")
        axes[2].set_title(f"Anomaly Map (score={result['anomaly_score']:.3f})")
        axes[2].axis("off")

        plt.tight_layout()
        stem = Path(result.get("filename", "result")).stem
        plt.savefig(output_dir / f"{stem}_anomaly.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n✓ Saved anomaly maps to {output_dir}")
    print(f"  Processed {len(results)} images")

    # Print anomaly scores
    scores = [(r.get("filename", "?"), r["anomaly_score"]) for r in results]
    scores.sort(key=lambda x: x[1], reverse=True)
    print("\nTop anomaly scores:")
    for name, score in scores[:10]:
        print(f"  {name}: {score:.4f}")


if __name__ == "__main__":
    main()
