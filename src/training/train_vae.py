"""
Stage 1: Train the VAE for perceptual compression.

Trains an AutoencoderKL to compress 256×256 grayscale brain MRIs
into a 32×32×4 latent space. Uses L1 reconstruction + LPIPS perceptual
loss + KL divergence.
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.vae import build_vae, VAE
from src.data.dataset import create_dataloaders
from src.utils.config import load_config
from src.utils.helpers import (
    set_seed, get_device, count_parameters, format_params,
    save_checkpoint, load_checkpoint, update_ema, ensure_dir,
)
from src.utils.logging import TrainingLogger


def compute_perceptual_loss(lpips_model, recon, target):
    """Compute LPIPS perceptual loss (expects 3-channel input)."""
    # Repeat grayscale to 3 channels for LPIPS
    recon_3ch = recon.repeat(1, 3, 1, 1)
    target_3ch = target.repeat(1, 3, 1, 1)
    return lpips_model(recon_3ch, target_3ch).mean()


def train_vae(config_path: str, resume: str = None, debug: bool = False):
    """Main VAE training loop."""
    config = load_config(config_path)
    set_seed(42)
    device = get_device()

    # Setup logging
    logger = TrainingLogger(
        log_dir=config.paths.log_dir,
        experiment_name="vae_training",
    )

    logger.info(f"Device: {device}")
    logger.info(f"Config: {config.to_dict()}")

    # Build model
    vae = build_vae(config).to(device)
    ema_vae = copy.deepcopy(vae)
    ema_vae.requires_grad_(False)

    num_params = count_parameters(vae)
    logger.info(f"VAE parameters: {format_params(num_params)} ({num_params:,})")

    # LPIPS perceptual loss
    import lpips
    lpips_model = lpips.LPIPS(net="vgg").to(device)
    lpips_model.requires_grad_(False)

    # Data
    train_loader, val_loader = create_dataloaders(
        train_dir=project_root / config.paths.train_dir,
        val_dir=project_root / config.paths.val_dir,
        image_size=config.data.image_size,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Learning rate scheduler (cosine annealing)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs,
        eta_min=1e-6,
    )

    # Mixed precision
    scaler = GradScaler("cuda", enabled=config.training.use_amp)

    # Resume from checkpoint
    start_epoch = 0
    if resume:
        ckpt = load_checkpoint(resume, vae, optimizer, ema_vae, device=str(device))
        start_epoch = ckpt.get("step_or_epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # Debug mode: reduce epochs
    num_epochs = 5 if debug else config.training.num_epochs
    accum_steps = config.training.gradient_accumulation_steps

    best_val_loss = float("inf")

    for epoch in range(start_epoch, num_epochs):
        vae.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_percep_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()

        for batch_idx, (images, _, _) in enumerate(pbar):
            images = images.to(device)

            with autocast("cuda", enabled=config.training.use_amp):
                recon, mu, logvar = vae(images)

                # L1 reconstruction loss
                recon_loss = F.l1_loss(recon, images)

                # Perceptual loss
                percep_loss = compute_perceptual_loss(lpips_model, recon, images)

                # KL divergence
                kl_loss = VAE.kl_loss(mu, logvar)

                # Total loss
                loss = (
                    config.training.reconstruction_weight * recon_loss
                    + config.training.perceptual_weight * percep_loss
                    + config.training.kl_weight * kl_loss
                )
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                if config.training.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        vae.parameters(), config.training.grad_clip_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Update EMA
                update_ema(ema_vae, vae, config.training.ema_decay)

            epoch_loss += loss.item() * accum_steps
            epoch_recon_loss += recon_loss.item()
            epoch_percep_loss += percep_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item() * accum_steps:.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}",
            })

        lr_scheduler.step()

        # Log epoch metrics
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon_loss / num_batches
        avg_percep = epoch_percep_loss / num_batches
        avg_kl = epoch_kl_loss / num_batches

        logger.log_scalar("train/loss", avg_loss, epoch)
        logger.log_scalar("train/recon_loss", avg_recon, epoch)
        logger.log_scalar("train/perceptual_loss", avg_percep, epoch)
        logger.log_scalar("train/kl_loss", avg_kl, epoch)
        logger.log_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            f"Epoch {epoch+1}: loss={avg_loss:.4f} recon={avg_recon:.4f} "
            f"percep={avg_percep:.4f} kl={avg_kl:.4f}"
        )

        # Validation
        if (epoch + 1) % config.training.val_every_epochs == 0:
            val_loss = validate_vae(vae, val_loader, lpips_model, config, device)
            logger.log_scalar("val/loss", val_loss, epoch)
            logger.info(f"  Val loss: {val_loss:.4f}")

            # Log sample reconstructions
            with torch.no_grad():
                sample_images = next(iter(val_loader))[0][:8].to(device)
                sample_recon, _, _ = ema_vae(sample_images)
                comparison = torch.cat([sample_images, sample_recon], dim=0)
                logger.log_image("val/reconstructions", comparison, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    ema_vae, optimizer, epoch, val_loss,
                    ensure_dir(project_root / config.paths.checkpoint_dir) / "best.pt",
                )
                logger.info(f"  ✓ New best model saved (val_loss={val_loss:.4f})")

        # Periodic checkpointing
        if (epoch + 1) % config.training.save_every_epochs == 0:
            save_checkpoint(
                vae, optimizer, epoch, avg_loss,
                ensure_dir(project_root / config.paths.checkpoint_dir) / f"epoch_{epoch+1}.pt",
                ema_model=ema_vae,
            )

    # Save final model
    save_checkpoint(
        ema_vae, optimizer, num_epochs - 1, avg_loss,
        ensure_dir(project_root / config.paths.checkpoint_dir) / "final.pt",
    )

    logger.info("✓ VAE training complete!")
    logger.close()


@torch.no_grad()
def validate_vae(vae, val_loader, lpips_model, config, device):
    """Run validation and return average loss."""
    vae.eval()
    total_loss = 0.0
    num_batches = 0

    for images, _, _ in val_loader:
        images = images.to(device)

        with autocast("cuda", enabled=config.training.use_amp):
            recon, mu, logvar = vae(images)
            recon_loss = F.l1_loss(recon, images)
            percep_loss = compute_perceptual_loss(lpips_model, recon, images)
            kl_loss = VAE.kl_loss(mu, logvar)

            loss = (
                config.training.reconstruction_weight * recon_loss
                + config.training.perceptual_weight * percep_loss
                + config.training.kl_weight * kl_loss
            )

        total_loss += loss.item()
        num_batches += 1

    vae.train()
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train VAE for brain MRI compression")
    parser.add_argument(
        "--config", type=str, default="configs/vae_config.yaml",
        help="Path to VAE config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: train for 5 epochs only"
    )
    args = parser.parse_args()

    train_vae(
        config_path=str(project_root / args.config),
        resume=args.resume,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
