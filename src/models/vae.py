"""
VAE (Variational Autoencoder) wrapper for perceptual compression.

Uses MONAI's AutoencoderKL to compress 256×256 grayscale brain MRIs
into a 32×32×4 latent space for efficient latent diffusion.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from monai.networks.nets import AutoencoderKL


class VAE(nn.Module):
    """
    Variational Autoencoder for brain MRI compression.

    Maps images from pixel space (1, 256, 256) to latent space (4, 32, 32),
    providing an 8× spatial downsampling with 4 latent channels.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 4,
        channels: list = [64, 128, 256, 256],
        num_res_blocks: list = [2, 2, 2, 2],
        attention_levels: list = [False, False, True, True],
        norm_num_groups: int = 32,
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
    ):
        super().__init__()

        # Handle num_res_blocks: MONAI expects a list
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * len(channels)

        self.model = AutoencoderKL(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            channels=channels,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            norm_num_groups=norm_num_groups,
            with_encoder_nonlocal_attn=with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=with_decoder_nonlocal_attn,
        )

        self.latent_channels = latent_channels

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode images to latent space.

        Args:
            x: Input images (B, 1, 256, 256), values in [-1, 1]

        Returns:
            z: Sampled latent (B, 4, 32, 32)
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        z_mu, z_sigma = self.model.encode(x)
        z = self.model.sampling(z_mu, z_sigma)
        # Convert sigma to logvar for KL computation
        logvar = 2 * torch.log(z_sigma + 1e-8)
        return z, z_mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images.

        Args:
            z: Latent vectors (B, 4, 32, 32)

        Returns:
            Reconstructed images (B, 1, 256, 256), values in [-1, 1]
        """
        return self.model.decode(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → sample → decode.

        Returns:
            recon: Reconstructed images
            mu: Latent mean
            logvar: Latent log-variance
        """
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space (deterministic, using mean).
        Used for precomputing latents for DDPM training.
        """
        z_mu, z_sigma = self.model.encode(x)
        # Use mean for deterministic encoding
        return z_mu

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss: KL(q(z|x) || p(z)).

        Args:
            mu: Mean of approximate posterior (B, C, H, W)
            logvar: Log-variance of approximate posterior (B, C, H, W)

        Returns:
            KL divergence loss (scalar)
        """
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def build_vae(config) -> VAE:
    """Build a VAE from config object."""
    model_cfg = config.model

    # Handle num_res_blocks: could be int or list in config
    num_res_blocks = model_cfg.num_res_blocks
    if isinstance(num_res_blocks, int):
        num_res_blocks = [num_res_blocks] * len(model_cfg.num_channels)

    return VAE(
        spatial_dims=model_cfg.spatial_dims,
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        latent_channels=model_cfg.latent_channels,
        channels=model_cfg.num_channels,
        num_res_blocks=num_res_blocks,
        attention_levels=model_cfg.attention_levels,
        norm_num_groups=model_cfg.norm_num_groups,
        with_encoder_nonlocal_attn=model_cfg.with_encoder_nonlocal_attn,
        with_decoder_nonlocal_attn=model_cfg.with_decoder_nonlocal_attn,
    )
