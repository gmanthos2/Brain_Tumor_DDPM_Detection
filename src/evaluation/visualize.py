"""
Visualization utilities for anomaly detection and generation evaluation.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.utils.helpers import ensure_dir


def plot_anomaly_grid(results, output_path, max_samples=8):
    """Plot grid: original | reconstruction | anomaly heatmap."""
    n = min(len(results), max_samples)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i in range(n):
        r = results[i]
        axes[i, 0].imshow(r["original"].numpy(), cmap="gray")
        axes[i, 0].set_title(f"Original: {r.get('filename', '')}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(r["reconstruction"].numpy(), cmap="gray")
        axes[i, 1].set_title("Healthy Reconstruction")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(r["anomaly_map"], cmap="hot")
        axes[i, 2].set_title(f"Anomaly (score={r['anomaly_score']:.3f})")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(fpr, tpr, auroc, output_path):
    """Plot ROC curve with AUROC value."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("ROC Curve — Anomaly Detection", fontsize=16)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_distributions(healthy_scores, anomalous_scores, output_path):
    """Plot distribution of anomaly scores for healthy vs anomalous."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(healthy_scores, bins=30, alpha=0.6, label="Healthy", color="green", density=True)
    ax.hist(anomalous_scores, bins=30, alpha=0.6, label="Anomalous", color="red", density=True)
    ax.set_xlabel("Anomaly Score", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title("Anomaly Score Distribution", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_generation_grid(images, output_path, nrow=4):
    """Plot a grid of generated images."""
    from torchvision.utils import make_grid
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    fig, ax = plt.subplots(figsize=(2 * nrow, 2 * ((len(images) + nrow - 1) // nrow)))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    ax.set_title("Generated Synthetic Brain MRIs", fontsize=16)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
