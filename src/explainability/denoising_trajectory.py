"""
Denoising trajectory visualization.

Shows the step-by-step denoising process, revealing how the model
progressively constructs anatomical structures and how it handles
anomalous regions differently from healthy ones.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.utils.helpers import ensure_dir


@torch.no_grad()
def capture_denoising_trajectory(
    diffusion, vae, shape, class_labels, guidance_scale,
    num_steps=50, capture_every=5, device="cuda",
    x_start=None, t_start=None,
):
    """
    Run DDIM sampling and capture intermediate denoised images.

    Returns list of (timestep, decoded_image) tuples.
    """
    scheduler = diffusion.scheduler
    step_size = scheduler.num_timesteps // num_steps
    timesteps = list(range(0, scheduler.num_timesteps, step_size))
    timesteps = list(reversed(timesteps))

    if x_start is not None and t_start is not None:
        x = x_start
        timesteps = [t for t in timesteps if t <= t_start]
    else:
        x = torch.randn(shape, device=device)

    trajectory = []
    for i, t in enumerate(timesteps):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        predicted_noise = diffusion._guided_prediction(x, t_batch, class_labels, guidance_scale)
        x_0_pred = scheduler.predict_x0_from_noise(x, t_batch, predicted_noise)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)

        if i % capture_every == 0 or i == len(timesteps) - 1:
            decoded = vae.decode(x_0_pred) if vae else x_0_pred
            trajectory.append((t, decoded.cpu()))

        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            alpha_t = scheduler.alphas_cumprod[t]
            alpha_next = scheduler.alphas_cumprod[t_next]
            pred_dir = torch.sqrt(1 - alpha_next) * predicted_noise
            x = torch.sqrt(alpha_next) * x_0_pred + pred_dir
        else:
            x = x_0_pred

    return trajectory


def plot_trajectory(trajectory, output_path, title="Denoising Trajectory"):
    """Plot the denoising trajectory as a horizontal strip."""
    n = len(trajectory)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, (t, img) in enumerate(trajectory):
        axes[i].imshow(img[0].squeeze().numpy(), cmap="gray")
        axes[i].set_title(f"t={t}", fontsize=12)
        axes[i].axis("off")
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compare_trajectories(
    traj_healthy, traj_anomalous, output_path,
):
    """Compare denoising trajectories for healthy vs anomalous input."""
    n = min(len(traj_healthy), len(traj_anomalous))
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    for i in range(n):
        axes[0, i].imshow(traj_healthy[i][1][0].squeeze().numpy(), cmap="gray")
        axes[0, i].set_title(f"t={traj_healthy[i][0]}")
        axes[0, i].axis("off")
        axes[1, i].imshow(traj_anomalous[i][1][0].squeeze().numpy(), cmap="gray")
        axes[1, i].set_title(f"t={traj_anomalous[i][0]}")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Healthy", fontsize=14)
    axes[1, 0].set_ylabel("Anomalous", fontsize=14)
    fig.suptitle("Denoising Trajectory Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
