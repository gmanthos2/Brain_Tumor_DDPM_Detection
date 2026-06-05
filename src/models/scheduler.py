"""
Noise schedules for the diffusion process.

Provides linear and cosine beta schedules, plus precomputed alpha
quantities used throughout training and sampling.
"""

import math
import torch
from typing import Optional


class NoiseScheduler:
    """
    Manages the noise schedule for DDPM/DDIM.

    Precomputes and stores all α, β, and derived quantities needed
    for the forward (noising) and reverse (denoising) processes.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type

        # Compute beta schedule
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            betas = self._cosine_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.betas = betas.to(device)

        # Precompute derived quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device),
            self.alphas_cumprod[:-1]
        ])

        # Quantities for q(x_t | x_0) — forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Quantities for posterior q(x_{t-1} | x_t, x_0) — reverse process
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        # Quantities for predicting x_0 from noise
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def _cosine_schedule(self, num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine noise schedule as proposed in 'Improved DDPM'.

        Provides smoother noise progression, often better for medical images.
        """
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        f_t = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min=0.0001, max=0.9999).float()

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Forward diffusion: sample x_t given x_0 and timestep t.

        q(x_t | x_0) = N(√ᾱ_t * x_0, (1 - ᾱ_t) * I)

        Args:
            x_0: Clean data (B, C, H, W)
            t: Timesteps (B,)
            noise: Optional pre-sampled noise (B, C, H, W)

        Returns:
            x_t: Noised data
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.

        x_0 = (1/√ᾱ_t) * x_t - (√(1/ᾱ_t - 1)) * ε
        """
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * predicted_noise

    def q_posterior(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple:
        """
        Compute posterior mean and variance: q(x_{t-1} | x_t, x_0).
        """
        coef1 = self._extract(self.posterior_mean_coef1, t, x_0.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x_0 + coef2 * x_t
        var = self._extract(self.posterior_variance, t, x_t.shape)
        log_var = self._extract(self.posterior_log_variance, t, x_t.shape)
        return mean, var, log_var

    def to(self, device: torch.device) -> "NoiseScheduler":
        """Move all tensors to the specified device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance = self.posterior_log_variance.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        return self

    @staticmethod
    def _extract(tensor: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Extract values from tensor at indices t, reshape for broadcasting."""
        batch_size = t.shape[0]
        out = tensor.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))
