"""
Stage 2: Train the DDPM U-Net on precomputed latent representations.

Trains a class-conditional U-Net to predict noise in the VAE latent
space. Uses Classifier-Free Guidance training (random class dropout).
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.unet import build_unet
from src.models.scheduler import NoiseScheduler
from src.models.diffusion import GaussianDiffusion
from src.models.vae import build_vae
from src.data.dataset import LatentDataset
from src.utils.config import load_config
from src.utils.helpers import (
    set_seed, get_device, count_parameters, format_params,
    save_checkpoint, load_checkpoint, update_ema, ensure_dir,
)
from src.utils.logging import TrainingLogger


def generate_samples(
    diffusion: GaussianDiffusion,
    vae,
    num_samples: int,
    guidance_scale: float,
    ddim_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate sample images for visual monitoring during training."""
    diffusion.eval()
    shape = (num_samples, 4, 32, 32)  # Latent shape

    # Generate with healthy class conditioning
    class_labels = torch.zeros(num_samples, device=device, dtype=torch.long)

    latents = diffusion.ddim_sample(
        shape=shape,
        class_labels=class_labels,
        guidance_scale=guidance_scale,
        num_steps=ddim_steps,
        device=device,
    )

    # Decode through VAE if available
    if vae is not None:
        with torch.no_grad():
            images = vae.decode(latents)
    else:
        images = latents

    diffusion.train()
    return images


def train_ddpm(config_path: str, vae_config_path: str = None, resume: str = None, debug: bool = False):
    """Main DDPM training loop."""
    config = load_config(config_path)
    set_seed(42)
    device = get_device()

    # Setup logging
    logger = TrainingLogger(
        log_dir=config.paths.log_dir,
        experiment_name="ddpm_training",
    )

    logger.info(f"Device: {device}")

    # Build U-Net
    unet = build_unet(config).to(device)
    ema_unet = copy.deepcopy(unet)
    ema_unet.requires_grad_(False)

    num_params = count_parameters(unet)
    logger.info(f"U-Net parameters: {format_params(num_params)} ({num_params:,})")

    # Build noise scheduler
    scheduler = NoiseScheduler(
        num_timesteps=config.diffusion.num_timesteps,
        schedule_type=config.diffusion.schedule_type,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        device=str(device),
    )

    # Build diffusion wrapper
    diffusion = GaussianDiffusion(
        model=unet,
        scheduler=scheduler,
        num_classes=config.model.num_classes,
    ).to(device)

    # Optionally load VAE for sample visualization
    vae = None
    if vae_config_path:
        vae_config = load_config(vae_config_path)
        vae = build_vae(vae_config).to(device)
        vae_ckpt_path = project_root / "checkpoints" / "vae" / "best.pt"
        if vae_ckpt_path.exists():
            ckpt = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                vae.load_state_dict(ckpt["model_state_dict"])
            else:
                vae.load_state_dict(ckpt)
            vae.eval()
            vae.requires_grad_(False)
            logger.info("Loaded VAE for sample visualization")
        else:
            logger.warning("VAE checkpoint not found, samples will show raw latents")
            vae = None

    # Dataset
    latent_dir = project_root / config.data.latent_dir
    dataset = LatentDataset(latent_dir=latent_dir, label=0)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
    )
    logger.info(f"Training on {len(dataset)} latent samples")

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Learning rate scheduler (cosine with warmup)
    def lr_lambda(step):
        if step < config.training.warmup_steps:
            return step / config.training.warmup_steps
        progress = (step - config.training.warmup_steps) / (
            config.training.max_steps - config.training.warmup_steps
        )
        return max(0.1, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = GradScaler("cuda", enabled=config.training.use_amp)

    # Resume
    start_step = 0
    if resume:
        ckpt = load_checkpoint(resume, unet, optimizer, ema_unet, device=str(device))
        start_step = ckpt.get("step_or_epoch", 0) + 1
        logger.info(f"Resumed from step {start_step}")

    max_steps = 1000 if debug else config.training.max_steps

    # Training loop
    global_step = start_step
    best_loss = float("inf")
    running_loss = 0.0
    loss_count = 0

    logger.info(f"Starting training for {max_steps} steps...")

    while global_step < max_steps:
        pbar = tqdm(loader, desc=f"Step {global_step}/{max_steps}")

        for latents, labels in pbar:
            if global_step >= max_steps:
                break

            latents = latents.to(device)
            labels = labels.to(device)

            with autocast("cuda", enabled=config.training.use_amp):
                loss = diffusion.training_loss(latents, class_labels=labels)

            scaler.scale(loss).backward()

            if config.training.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    unet.parameters(), config.training.grad_clip_norm
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()

            # Update EMA
            update_ema(ema_unet, unet, config.training.ema_decay)

            running_loss += loss.item()
            loss_count += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{running_loss / loss_count:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            })

            # Logging
            if global_step % config.training.log_every_steps == 0:
                avg_loss = running_loss / loss_count
                logger.log_scalar("train/loss", avg_loss, global_step)
                logger.log_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                logger.info(f"Step {global_step}: loss={avg_loss:.4f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss

                running_loss = 0.0
                loss_count = 0

            # Generate samples
            if global_step % config.training.sample_every_steps == 0:
                logger.info("Generating samples...")
                # Use EMA model for sampling
                ema_diffusion = GaussianDiffusion(ema_unet, scheduler, config.model.num_classes)
                samples = generate_samples(
                    ema_diffusion, vae,
                    num_samples=min(config.training.num_sample_images, 16),
                    guidance_scale=config.diffusion.guidance_scale,
                    ddim_steps=config.diffusion.ddim_steps,
                    device=device,
                )
                logger.log_image("samples/generated", samples, global_step)

                # Save sample images
                from torchvision.utils import save_image
                sample_dir = ensure_dir(project_root / config.paths.sample_dir)
                save_image(
                    samples,
                    sample_dir / f"samples_step_{global_step}.png",
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1),
                )

            # Save checkpoint
            if global_step % config.training.save_every_steps == 0:
                save_checkpoint(
                    unet, optimizer, global_step, best_loss,
                    ensure_dir(project_root / config.paths.checkpoint_dir) / f"step_{global_step}.pt",
                    ema_model=ema_unet,
                )

    # Save final models
    ckpt_dir = ensure_dir(project_root / config.paths.checkpoint_dir)
    save_checkpoint(unet, optimizer, global_step, best_loss, ckpt_dir / "final.pt", ema_model=ema_unet)
    save_checkpoint(ema_unet, optimizer, global_step, best_loss, ckpt_dir / "best.pt")

    logger.info(f"✓ DDPM training complete! Best loss: {best_loss:.4f}")
    logger.close()


def main():
    parser = argparse.ArgumentParser(description="Train DDPM U-Net on latent space")
    parser.add_argument(
        "--config", type=str, default="configs/ddpm_config.yaml",
        help="Path to DDPM config"
    )
    parser.add_argument(
        "--vae-config", type=str, default="configs/vae_config.yaml",
        help="Path to VAE config (for sample visualization)"
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Debug mode: 1000 steps only")
    args = parser.parse_args()

    train_ddpm(
        config_path=str(project_root / args.config),
        vae_config_path=str(project_root / args.vae_config),
        resume=args.resume,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
