"""
scripts/compare_anomaly_detection.py

Head-to-head comparison between Our Latent DDPM Anomaly Detector
and Rad-DiT's Flow Matching Anomaly Detector.

Computes:
1. Image-level metrics (AUROC, AUPRC, Accuracy, Sensitivity, Specificity, F1)
2. Pixel-level metrics against ground truth tumor masks (if available)
3. Side-by-side visual comparison grids
4. Comprehensive JSON summary report
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.reconstruct import AnomalyDetector as OurAnomalyDetector
from baselines.raddit_anomaly_detector import RadDiTAnomalyDetector
from src.utils.helpers import ensure_dir


def create_mask_from_regions(regions, original_size, target_size=(256, 256)):
    """Generate a binary ground-truth mask from VIA JSON regions."""
    mask = Image.new("L", original_size, 0)
    draw = ImageDraw.Draw(mask)

    for region in regions:
        shape_attr = region.get("shape_attributes", {})
        shape_name = shape_attr.get("name")

        if shape_name == "polygon":
            xs = shape_attr.get("all_points_x", [])
            ys = shape_attr.get("all_points_y", [])
            if xs and ys:
                draw.polygon(list(zip(xs, ys)), fill=1)
        elif shape_name == "ellipse":
            cx, cy = shape_attr.get("cx", 0), shape_attr.get("cy", 0)
            rx, ry = shape_attr.get("rx", 0), shape_attr.get("ry", 0)
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=1)
        elif shape_name == "circle":
            cx, cy = shape_attr.get("cx", 0), shape_attr.get("cy", 0)
            r = shape_attr.get("r", 0)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=1)

    mask = mask.resize(target_size, Image.NEAREST)
    return np.array(mask)


def compute_binary_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive binary classification metrics."""
    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))

    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    threshold = float(thresholds[best_idx])

    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    accuracy = float((tp + tn) / len(labels))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "optimal_threshold": threshold,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_image_level(detector, healthy_dir, anomalous_dir, detect_kwargs, name="Model"):
    """Evaluates image-level anomaly detection performance on test directories."""
    print(f"\n--- Running Image-Level Evaluation: {name} ---")
    h_files = sorted(list(Path(healthy_dir).glob("*.png")))
    a_files = sorted(list(Path(anomalous_dir).glob("*.png")))

    scores = []
    labels = []

    for f in tqdm(h_files, desc=f"[{name}] Healthy scans"):
        res = detector.detect(str(f), **detect_kwargs)
        scores.append(res["anomaly_score"])
        labels.append(0)

    for f in tqdm(a_files, desc=f"[{name}] Anomalous scans"):
        res = detector.detect(str(f), **detect_kwargs)
        scores.append(res["anomaly_score"])
        labels.append(1)

    scores = np.array(scores)
    labels = np.array(labels)
    metrics = compute_binary_metrics(scores, labels)
    return metrics, scores, labels


def evaluate_pixel_level(
    our_detector,
    raddit_detector,
    annotations_path,
    orig_dir,
    our_kwargs,
    raddit_kwargs,
    max_samples=50,
    output_dir=None,
):
    """Evaluates pixel-level AUROC and saves qualitative side-by-side grids."""
    if not Path(annotations_path).exists():
        print(f"Annotations not found at {annotations_path}. Skipping pixel-level evaluation.")
        return None

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    gt_masks = []
    our_maps = []
    raddit_maps = []
    saved_qual_count = 0

    print("\n--- Running Pixel-Level Ground-Truth Evaluation ---")
    for key, item in tqdm(list(annotations.items())[:max_samples], desc="Pixel evaluation"):
        filename = item.get("filename")
        regions = item.get("regions", [])
        if not regions:
            continue

        img_path = Path(orig_dir) / filename
        if not img_path.exists():
            continue

        with Image.open(img_path) as img:
            orig_size = img.size

        gt_mask = create_mask_from_regions(regions, orig_size, target_size=(256, 256))

        # Run our detector
        res_our = our_detector.detect(str(img_path), **our_kwargs)
        # Run Rad-DiT detector
        res_raddit = raddit_detector.detect(str(img_path), **raddit_kwargs)

        gt_masks.append(gt_mask.flatten())
        our_maps.append(res_our["anomaly_map_raw"].flatten())
        raddit_maps.append(res_raddit["anomaly_map_raw"].flatten())

        # Save side-by-side qualitative comparisons for the first few samples
        if output_dir and saved_qual_count < 5:
            fig, axes = plt.subplots(1, 6, figsize=(22, 4))
            axes[0].imshow(res_our["original"].squeeze().numpy(), cmap="gray")
            axes[0].set_title("Input Image")

            axes[1].imshow(gt_mask, cmap="gray")
            axes[1].set_title("Ground Truth Mask")

            axes[2].imshow(res_our["reconstruction"].squeeze().numpy(), cmap="gray")
            axes[2].set_title("Ours: Recon (Healthy)")

            axes[3].imshow(res_our["anomaly_map"], cmap="hot")
            axes[3].set_title(f"Ours: Map (Score={res_our['anomaly_score']:.3f})")

            axes[4].imshow(res_raddit["reconstruction"].squeeze().numpy(), cmap="gray")
            axes[4].set_title("Rad-DiT: Recon")

            axes[5].imshow(res_raddit["anomaly_map"], cmap="hot")
            axes[5].set_title(f"Rad-DiT: Map (Score={res_raddit['anomaly_score']:.3f})")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            out_img = Path(output_dir) / f"comparison_{Path(filename).stem}.png"
            plt.savefig(out_img, dpi=200, bbox_inches="tight")
            plt.close()
            saved_qual_count += 1

    if not gt_masks:
        return None

    y_true = np.concatenate(gt_masks)
    y_our = np.concatenate(our_maps)
    y_raddit = np.concatenate(raddit_maps)

    our_pixel_auroc = float(roc_auc_score(y_true, y_our))
    raddit_pixel_auroc = float(roc_auc_score(y_true, y_raddit))

    return {
        "our_pixel_auroc": our_pixel_auroc,
        "raddit_pixel_auroc": raddit_pixel_auroc,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Anomaly Detection: Ours vs Rad-DiT")
    parser.add_argument("--healthy-dir", default="data/processed/test/healthy")
    parser.add_argument("--anomalous-dir", default="data/processed/test/anomalous")
    parser.add_argument("--output-dir", default="results/comparison")

    # Our model configs
    parser.add_argument("--vae-config", default="configs/vae_config.yaml")
    parser.add_argument("--ddpm-config", default="configs/ddpm_config.yaml")
    parser.add_argument("--vae-checkpoint", default="checkpoints/vae/best.pt")
    parser.add_argument("--ddpm-checkpoint", default="checkpoints/ddpm/step_100000.pt")
    parser.add_argument("--our-t-start", type=int, default=150)
    parser.add_argument("--our-guidance-scale", type=float, default=7.5)
    parser.add_argument("--our-ddim-steps", type=int, default=50)

    # Rad-DiT configs
    parser.add_argument(
        "--raddit-checkpoint",
        default="baselines/rad-dit/checkpoints/checkpoint-last.pth",
        help="Path to Rad-DiT checkpoint file (e.g. JiT-L/16)",
    )
    parser.add_argument("--raddit-model", default="JiT-L/16")
    parser.add_argument("--raddit-t-start", type=float, default=0.75)
    parser.add_argument("--raddit-steps", type=int, default=50)

    # Ground truth annotations for pixel evaluation
    parser.add_argument("--annotations", default="dataset/Br35H-Mask-RCNN/annotations_all.json")
    parser.add_argument("--orig-dir", default="dataset/yes")
    parser.add_argument("--max-pixel-samples", type=int, default=50)

    args = parser.parse_args()

    out_dir = ensure_dir(PROJECT_ROOT / args.output_dir)

    print("=" * 65)
    print(" ANOMALY DETECTION HEAD-TO-HEAD COMPARISON ")
    print("=" * 65)

    # 1. Initialize Our Detector
    print("\n[1/2] Loading Our Model (Latent DDPM)...")
    our_detector = OurAnomalyDetector(
        vae_config_path=str(PROJECT_ROOT / args.vae_config),
        ddpm_config_path=str(PROJECT_ROOT / args.ddpm_config),
        vae_checkpoint_path=str(PROJECT_ROOT / args.vae_checkpoint),
        ddpm_checkpoint_path=str(PROJECT_ROOT / args.ddpm_checkpoint),
    )

    our_detect_kwargs = {
        "t_start": args.our_t_start,
        "guidance_scale": args.our_guidance_scale,
        "ddim_steps": args.our_ddim_steps,
    }

    # 2. Initialize Rad-DiT Detector
    print("\n[2/2] Loading Rad-DiT Model (Pixel Flow Matching)...")
    raddit_ckpt_path = PROJECT_ROOT / args.raddit_checkpoint
    if not raddit_ckpt_path.exists():
        print(f"\n[!] Rad-DiT checkpoint not found at: {raddit_ckpt_path}")
        print("    Please download the pretrained checkpoint into baselines/rad-dit/checkpoints/.")
        print("    Running benchmark on Our Model only for baseline recording...\n")
        raddit_detector = None
    else:
        raddit_detector = RadDiTAnomalyDetector(
            checkpoint_path=str(raddit_ckpt_path),
            model_variant=args.raddit_model,
        )

    raddit_detect_kwargs = {
        "t_start": args.raddit_t_start,
        "num_steps": args.raddit_steps,
    }

    # 3. Image-Level Evaluation
    h_dir = str(PROJECT_ROOT / args.healthy_dir)
    a_dir = str(PROJECT_ROOT / args.anomalous_dir)

    our_metrics, _, _ = evaluate_image_level(
        our_detector, h_dir, a_dir, our_detect_kwargs, name="Our Model (Latent DDPM)"
    )

    if raddit_detector is not None:
        raddit_metrics, _, _ = evaluate_image_level(
            raddit_detector, h_dir, a_dir, raddit_detect_kwargs, name="Rad-DiT (Flow Matching)"
        )
    else:
        raddit_metrics = None

    # 4. Pixel-Level Evaluation
    pixel_metrics = None
    if raddit_detector is not None:
        pixel_metrics = evaluate_pixel_level(
            our_detector=our_detector,
            raddit_detector=raddit_detector,
            annotations_path=str(PROJECT_ROOT / args.annotations),
            orig_dir=str(PROJECT_ROOT / args.orig_dir),
            our_kwargs=our_detect_kwargs,
            raddit_kwargs=raddit_detect_kwargs,
            max_samples=args.max_pixel_samples,
            output_dir=str(out_dir),
        )

    # 5. Print Comparison Table
    print("\n" + "=" * 65)
    print(f"{'METRIC':<25} | {'OUR MODEL':<16} | {'RAD-DIT':<16}")
    print("-" * 65)
    for m in ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1_score"]:
        our_val = f"{our_metrics[m]:.4f}"
        rad_val = f"{raddit_metrics[m]:.4f}" if raddit_metrics else "N/A"
        print(f"{m.upper():<25} | {our_val:<16} | {rad_val:<16}")

    if pixel_metrics:
        print("-" * 65)
        print(
            f"{'PIXEL AUROC':<25} | {pixel_metrics['our_pixel_auroc']:<16.4f} | "
            f"{pixel_metrics['raddit_pixel_auroc']:<16.4f}"
        )
    print("=" * 65)

    # 6. Save JSON report
    report = {
        "our_model": {
            "config": our_detect_kwargs,
            "image_level": our_metrics,
        },
        "raddit_model": {
            "config": raddit_detect_kwargs if raddit_detector else None,
            "image_level": raddit_metrics,
        },
        "pixel_level": pixel_metrics,
    }

    report_path = out_dir / "anomaly_comparison_metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[✓] Results saved to: {report_path}")


if __name__ == "__main__":
    main()
