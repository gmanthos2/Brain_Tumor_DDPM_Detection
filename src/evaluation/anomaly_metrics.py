"""
Anomaly detection evaluation metrics.

Computes AUROC, AUPRC, Dice, and IoU for anomaly detection performance
using reconstruction-error-based anomaly scores.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
)
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.inference.reconstruct import AnomalyDetector
from src.utils.config import load_config
from src.utils.helpers import ensure_dir


def compute_image_level_metrics(
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Compute image-level anomaly detection metrics.

    Args:
        anomaly_scores: Per-image anomaly scores (N,)
        labels: Ground truth labels (N,), 0=healthy, 1=anomalous

    Returns:
        Dictionary of metrics
    """
    auroc = roc_auc_score(labels, anomaly_scores)
    auprc = average_precision_score(labels, anomaly_scores)

    # Find optimal threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(labels, anomaly_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Binary predictions at optimal threshold
    predictions = (anomaly_scores >= optimal_threshold).astype(int)
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    accuracy = (tp + tn) / len(labels)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "optimal_threshold": optimal_threshold,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": thresholds,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "pr_thresholds": pr_thresholds,
    }


def compute_pixel_level_metrics(
    anomaly_maps: list,
    ground_truth_masks: list,
    threshold_percentile: float = 95,
) -> dict:
    """
    Compute pixel-level anomaly detection metrics.

    Args:
        anomaly_maps: List of anomaly maps (H, W)
        ground_truth_masks: List of binary masks (H, W)
        threshold_percentile: Percentile for binarizing anomaly maps

    Returns:
        Dictionary of pixel-level metrics
    """
    all_scores = np.concatenate([m.flatten() for m in anomaly_maps])
    all_labels = np.concatenate([m.flatten() for m in ground_truth_masks])

    # Pixel-level AUROC
    auroc = roc_auc_score(all_labels, all_scores)
    auprc = average_precision_score(all_labels, all_scores)

    # Dice at optimal threshold
    threshold = np.percentile(all_scores, threshold_percentile)
    predictions = (all_scores >= threshold).astype(int)

    intersection = np.sum(predictions * all_labels)
    dice = 2 * intersection / (np.sum(predictions) + np.sum(all_labels) + 1e-8)

    # IoU
    union = np.sum((predictions + all_labels) > 0)
    iou = intersection / (union + 1e-8)

    return {
        "pixel_auroc": auroc,
        "pixel_auprc": auprc,
        "pixel_dice": dice,
        "pixel_iou": iou,
        "threshold": threshold,
    }


def evaluate_anomaly_detection(
    detector: AnomalyDetector,
    healthy_dir: str,
    anomalous_dir: str,
    t_start: int = 300,
    guidance_scale: float = 3.0,
    ddim_steps: int = 50,
) -> dict:
    """
    Run full anomaly detection evaluation.

    Processes both healthy and anomalous test images and computes metrics.
    """
    print("Processing healthy test images...")
    healthy_results = detector.detect_batch(
        healthy_dir,
        t_start=t_start,
        guidance_scale=guidance_scale,
        ddim_steps=ddim_steps,
    )

    print("Processing anomalous test images...")
    anomalous_results = detector.detect_batch(
        anomalous_dir,
        t_start=t_start,
        guidance_scale=guidance_scale,
        ddim_steps=ddim_steps,
    )

    # Image-level evaluation
    scores = np.array(
        [r["anomaly_score"] for r in healthy_results]
        + [r["anomaly_score"] for r in anomalous_results]
    )
    labels = np.array(
        [0] * len(healthy_results) + [1] * len(anomalous_results)
    )

    metrics = compute_image_level_metrics(scores, labels)

    print("\n" + "=" * 50)
    print("ANOMALY DETECTION RESULTS")
    print("=" * 50)
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  AUPRC:       {metrics['auprc']:.4f}")
    print(f"  F1 Score:    {metrics['f1_score']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Threshold:   {metrics['optimal_threshold']:.4f}")
    print("=" * 50)

    return {
        "metrics": metrics,
        "healthy_results": healthy_results,
        "anomalous_results": anomalous_results,
        "scores": scores,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection performance")
    parser.add_argument("--healthy-dir", type=str, default="data/processed/test/healthy")
    parser.add_argument("--anomalous-dir", type=str, default="data/processed/test/anomalous")
    parser.add_argument("--output", type=str, default="results/evaluation")
    parser.add_argument("--vae-config", type=str, default="configs/vae_config.yaml")
    parser.add_argument("--ddpm-config", type=str, default="configs/ddpm_config.yaml")
    parser.add_argument("--vae-checkpoint", type=str, default="checkpoints/vae/best.pt")
    parser.add_argument("--ddpm-checkpoint", type=str, default="checkpoints/ddpm/best.pt")
    parser.add_argument("--t-start", type=int, default=300)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--ddim-steps", type=int, default=50)
    args = parser.parse_args()

    detector = AnomalyDetector(
        vae_config_path=str(project_root / args.vae_config),
        ddpm_config_path=str(project_root / args.ddpm_config),
        vae_checkpoint_path=str(project_root / args.vae_checkpoint),
        ddpm_checkpoint_path=str(project_root / args.ddpm_checkpoint),
    )

    output_dir = ensure_dir(project_root / args.output)

    results = evaluate_anomaly_detection(
        detector,
        healthy_dir=str(project_root / args.healthy_dir),
        anomalous_dir=str(project_root / args.anomalous_dir),
        t_start=args.t_start,
        guidance_scale=args.guidance_scale,
        ddim_steps=args.ddim_steps,
    )

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(results["metrics"]["fpr"], results["metrics"]["tpr"], color='darkorange', lw=2, label=f'ROC curve (AUROC = {results["metrics"]["auroc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Image-Level Anomaly Detection ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_path = output_dir / "roc_curve_image_level.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(results["metrics"]["recall_curve"], results["metrics"]["precision_curve"], color='blue', lw=2, label=f'PR curve (AUPRC = {results["metrics"]["auprc"]:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Image-Level Anomaly Detection Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    pr_path = output_dir / "pr_curve_image_level.png"
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics
    import json
    metrics_to_save = {
        k: v for k, v in results["metrics"].items()
        if not isinstance(v, np.ndarray)
    }
    with open(output_dir / "anomaly_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    print(f"\nMetrics saved to {output_dir / 'anomaly_metrics.json'}")
    print(f"ROC curve saved to {roc_path}")
    print(f"PR curve saved to {pr_path}")


if __name__ == "__main__":
    main()
