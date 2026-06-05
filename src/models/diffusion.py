"""
Diffusion process: DDPM forward and reverse processes.

Handles the training objective (noise prediction loss) and
provides utilities for both DDPM and DDIM sampling.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.models.scheduler import NoiseScheduler


class GaussianDiffusion(nn.Module):
    """
    DDPM diffusion process wrapper.

    Encapsulates the forward (noising) and reverse (denoising) processes,
    the training loss computation, and sampling procedures.
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: NoiseScheduler,
        num_classes: int = 2,
    ):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.num_timesteps = scheduler.num_timesteps
        self.num_classes = num_classes

    def training_loss(
        self,
        x_0: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the simplified DDPM training loss.

        L = E_{t, ε} [ ||ε - ε_θ(x_t, t, c)||² ]

        Args:
            x_0: Clean latent data (B, C, H, W)
            class_labels: Optional class labels (B,)

        Returns:
            MSE loss (scalar)
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Forward process: add noise
        x_t, _ = self.scheduler.q_sample(x_0, t, noise)

        # Predict noise
        predicted_noise = self.model(x_t, t, class_labels)

        # MSE loss
        loss = nn.functional.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def ddpm_sample(
        self,
        shape: Tuple[int, ...],
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Full DDPM sampling (1000 steps).

        Args:
            shape: Output shape (B, C, H, W)
            class_labels: Class labels for conditional generation
            guidance_scale: CFG guidance scale (1.0 = no guidance)
            device: Device to sample on

        Returns:
            Sampled latent (B, C, H, W)
        """
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise (with optional CFG)
            predicted_noise = self._guided_prediction(
                x, t_batch, class_labels, guidance_scale
            )

            # Predict x_0
            x_0_pred = self.scheduler.predict_x0_from_noise(x, t_batch, predicted_noise)
            x_0_pred = torch.clamp(x_0_pred, -1, 1)

            # Compute posterior
            mean, var, log_var = self.scheduler.q_posterior(x_0_pred, x, t_batch)

            # Sample x_{t-1}
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.exp(0.5 * log_var) * noise
            else:
                x = mean

        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: Tuple[int, ...],
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        num_steps: int = 50,
        eta: float = 0.0,
        device: str = "cuda",
        x_start: Optional[torch.Tensor] = None,
        t_start: Optional[int] = None,
    ) -> torch.Tensor:
        """
        DDIM sampling (accelerated, optionally deterministic).

        Args:
            shape: Output shape (B, C, H, W)
            class_labels: Class labels for conditional generation
            guidance_scale: CFG guidance scale
            num_steps: Number of DDIM steps (< num_timesteps)
            eta: Stochasticity (0=deterministic, 1=DDPM)
            device: Device to sample on
            x_start: Optional starting point (for reconstruction)
            t_start: Starting timestep (for partial denoising in anomaly detection)

        Returns:
            Sampled/reconstructed latent (B, C, H, W)
        """
        # Create DDIM timestep subsequence
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        # If starting from a partially noised sample (anomaly detection)
        if x_start is not None and t_start is not None:
            x = x_start
            # Filter timesteps to only include those <= t_start
            timesteps = [t for t in timesteps if t <= t_start]
        else:
            x = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise with CFG
            predicted_noise = self._guided_prediction(
                x, t_batch, class_labels, guidance_scale
            )

            # Predict x_0
            x_0_pred = self.scheduler.predict_x0_from_noise(x, t_batch, predicted_noise)
            x_0_pred = torch.clamp(x_0_pred, -1, 1)

            if i < len(timesteps) - 1:
                # Next timestep
                t_next = timesteps[i + 1]

                alpha_t = self.scheduler.alphas_cumprod[t]
                alpha_t_next = self.scheduler.alphas_cumprod[t_next]

                # DDIM update rule
                sigma = eta * torch.sqrt(
                    (1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next)
                )

                # Direction pointing to x_t
                pred_dir = torch.sqrt(1 - alpha_t_next - sigma ** 2) * predicted_noise

                # DDIM step
                x = torch.sqrt(alpha_t_next) * x_0_pred + pred_dir

                if sigma > 0:
                    x = x + sigma * torch.randn_like(x)
            else:
                x = x_0_pred

        return x

    def _guided_prediction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor],
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Classifier-Free Guidance prediction.

        ε̃ = (1 + w) · ε_θ(x_t, t, c) - w · ε_θ(x_t, t, ∅)
        """
        if guidance_scale <= 1.0 or class_labels is None:
            return self.model(x, t, class_labels)

        # Conditional prediction
        noise_cond = self.model(x, t, class_labels)

        # Unconditional prediction (null class)
        null_labels = torch.full_like(class_labels, self.num_classes)
        noise_uncond = self.model(x, t, null_labels)

        # CFG interpolation
        return noise_uncond + guidance_scale * (noise_cond - noise_uncond)
