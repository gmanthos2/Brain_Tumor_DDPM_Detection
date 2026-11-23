"""
scripts/compare_anomaly_detection.py

Head-to-head comparison between Our Latent DDPM (U-Net) Anomaly Detector
and Rad-DiT's Flow Matching (JiT) Anomaly Detector.

Computes:
1. Image-level metrics (AUROC, AUPRC, Accuracy, Sensitivity, Specificity, Precision, F1)
2. Pixel-level metrics against ground truth tumor masks (AUROC, AUPRC)
3. Inference latency comparison (seconds per image)
4. Combined ROC & PR curves and side-by-side qualitative visual comparison grids
5. Comprehensive JSON summary report
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
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
from baselines.ldit_brats_anomaly_detector import LDiTBratsAnomalyDetector
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

    healthy_scores = scores[labels == 0]
    anomalous_scores = scores[labels == 1]

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
        "healthy_score_mean": float(healthy_scores.mean()),
        "healthy_score_std": float(healthy_scores.std()),
        "anomalous_score_mean": float(anomalous_scores.mean()),
        "anomalous_score_std": float(anomalous_scores.std()),
        "num_healthy": int(len(healthy_scores)),
        "num_anomalous": int(len(anomalous_scores)),
    }


def _build_detector(model_kind: str, init_kwargs: Dict[str, Any]):
    if model_kind == "unet":
        return OurAnomalyDetector(**init_kwargs)
    if model_kind == "ldit-brats":
        return LDiTBratsAnomalyDetector(**init_kwargs)
    return RadDiTAnomalyDetector(**init_kwargs)


def _worker_run_shard(task: Tuple):
    """Worker process for CPU-parallel inference on a shard of images."""
    (
        worker_idx,
        model_kind,
        init_kwargs,
        detect_kwargs,
        image_tasks,
        pixel_tasks,
        threads_per_worker,
        base_seed,
    ) = task

    # Stagger checkpoint loading slightly to avoid simultaneous peak RAM spikes
    if worker_idx > 0:
        time.sleep(worker_idx * 1.5)

    torch.set_num_threads(threads_per_worker)
    detector = _build_detector(model_kind, init_kwargs)
    variant = getattr(detector, "model_variant", None)

    img_out = []
    t0 = time.time()
    for idx, img_path, label in image_tasks:
        torch.manual_seed(base_seed + idx)
        np.random.seed(base_seed + idx)
        res = detector.detect(str(img_path), **detect_kwargs)
        img_out.append((idx, float(res["anomaly_score"]), int(label)))
    img_elapsed = time.time() - t0

    pix_out = []
    qual_out = []
    for idx, img_path, filename, save_qual in pixel_tasks:
        torch.manual_seed(base_seed + 10000 + idx)
        np.random.seed(base_seed + 10000 + idx)
        res = detector.detect(str(img_path), **detect_kwargs)
        pix_out.append((idx, res["anomaly_map_raw"].astype(np.float32).flatten()))
        if save_qual:
            qual_out.append((
                idx,
                {
                    "filename": filename,
                    "original": res["original"].squeeze().numpy(),
                    "reconstruction": res["reconstruction"].squeeze().numpy(),
                    "anomaly_map": res["anomaly_map"],
                    "anomaly_score": float(res["anomaly_score"]),
                },
            ))

    del detector
    gc.collect()
    return img_out, img_elapsed, pix_out, qual_out, variant


def run_model_evaluation(
    model_kind: str,
    init_kwargs: Dict[str, Any],
    detect_kwargs: Dict[str, Any],
    healthy_dir: str,
    anomalous_dir: str,
    pixel_items: List,
    name: str,
    max_images: int = 0,
    seed: int = 42,
    num_workers: int = 1,
    num_qual: int = 5,
):
    """Run image-level and pixel-level evaluation for a single detector."""
    h_files = sorted(list(Path(healthy_dir).glob("*.png")))
    a_files = sorted(list(Path(anomalous_dir).glob("*.png")))
    if max_images and max_images > 0:
        h_files = h_files[:max_images]
        a_files = a_files[:max_images]

    image_tasks = []
    idx_counter = 0
    for f in h_files:
        image_tasks.append((idx_counter, str(f), 0))
        idx_counter += 1
    for f in a_files:
        image_tasks.append((idx_counter, str(f), 1))
        idx_counter += 1

    pixel_tasks = [
        (i, str(img_path), filename, i < num_qual)
        for i, (img_path, filename, _) in enumerate(pixel_items)
    ]

    use_parallel = (num_workers > 1) and (not torch.cuda.is_available())
    if use_parallel:
        phys_cores = max(1, (os.cpu_count() or 8) // 2)
        threads_per_worker = max(1, phys_cores // num_workers)
        print(
            f"\n--- Running {name} across {num_workers} CPU workers "
            f"({threads_per_worker} threads/worker) on {len(image_tasks)} test + "
            f"{len(pixel_tasks)} pixel images ---"
        )
        img_shards = [image_tasks[w::num_workers] for w in range(num_workers)]
        pix_shards = [pixel_tasks[w::num_workers] for w in range(num_workers)]

        wall_t0 = time.time()
        worker_args = [
            (
                w,
                model_kind,
                init_kwargs,
                detect_kwargs,
                img_shards[w],
                pix_shards[w],
                threads_per_worker,
                seed,
            )
            for w in range(num_workers)
        ]

        all_img = []
        all_pix = []
        all_qual = []
        worker_img_times = []
        variant_used = None

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for img_out, img_elapsed, pix_out, qual_out, variant in pool.map(
                _worker_run_shard, worker_args
            ):
                all_img.extend(img_out)
                worker_img_times.append(img_elapsed)
                all_pix.extend(pix_out)
                all_qual.extend(qual_out)
                if variant:
                    variant_used = variant

        wall_elapsed = time.time() - wall_t0
        all_img.sort(key=lambda x: x[0])
        all_pix.sort(key=lambda x: x[0])
        all_qual.sort(key=lambda x: x[0])

        scores = np.array([x[1] for x in all_img], dtype=np.float64)
        labels = np.array([x[2] for x in all_img], dtype=np.int64)
        metrics = compute_binary_metrics(scores, labels)
        metrics["wall_time_sec"] = float(wall_elapsed)
        metrics["total_time_sec"] = float(max(worker_img_times))
        metrics["avg_time_per_image_sec"] = float(max(worker_img_times) / max(len(scores), 1))
        metrics["effective_single_worker_sec_per_img"] = float(
            sum(worker_img_times) / max(len(scores), 1)
        )
        raw_maps = [x[1] for x in all_pix]
        qual_results = [x[1] for x in all_qual]
        print(f"Completed {name} in {wall_elapsed:.1f}s wall time.")
        return metrics, scores, labels, raw_maps, qual_results, variant_used

    # Single-process path (GPU or num_workers == 1)
    detector = _build_detector(model_kind, init_kwargs)
    variant_used = getattr(detector, "model_variant", None)

    print(f"\n--- Running Image-Level Evaluation: {name} ---")
    scores = []
    labels = []
    t0 = time.time()
    for idx, img_path, label in tqdm(image_tasks, desc=f"[{name}] Test scans"):
        torch.manual_seed(seed + idx)
        np.random.seed(seed + idx)
        res = detector.detect(img_path, **detect_kwargs)
        scores.append(float(res["anomaly_score"]))
        labels.append(int(label))
    elapsed = time.time() - t0

    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.int64)
    metrics = compute_binary_metrics(scores, labels)
    metrics["total_time_sec"] = float(elapsed)
    metrics["avg_time_per_image_sec"] = float(elapsed / max(len(scores), 1))

    raw_maps = []
    qual_results = []
    if pixel_tasks:
        for idx, img_path, filename, save_qual in tqdm(
            pixel_tasks, desc=f"[{name}] Pixel evaluation"
        ):
            torch.manual_seed(seed + 10000 + idx)
            np.random.seed(seed + 10000 + idx)
            res = detector.detect(img_path, **detect_kwargs)
            raw_maps.append(res["anomaly_map_raw"].astype(np.float32).flatten())
            if save_qual:
                qual_results.append({
                    "filename": filename,
                    "original": res["original"].squeeze().numpy(),
                    "reconstruction": res["reconstruction"].squeeze().numpy(),
                    "anomaly_map": res["anomaly_map"],
                    "anomaly_score": float(res["anomaly_score"]),
                })

    del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, scores, labels, raw_maps, qual_results, variant_used


def collect_pixel_items(annotations_path: str, orig_dir: str, max_samples: int = 50):
    """Collect valid annotated (img_path, filename, gt_mask) entries."""
    if not Path(annotations_path).exists():
        print(f"Annotations not found at {annotations_path}. Skipping pixel-level evaluation.")
        return []

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    items = []
    for _, item in annotations.items():
        if max_samples and len(items) >= max_samples:
            break
        filename = item.get("filename")
        regions = item.get("regions", [])
        if not regions or not filename:
            continue

        img_path = Path(orig_dir) / filename
        if not img_path.exists():
            continue

        with Image.open(img_path) as img:
            orig_size = img.size

        gt_mask = create_mask_from_regions(regions, orig_size, target_size=(256, 256))
        if gt_mask.sum() == 0:
            continue
        items.append((img_path, filename, gt_mask))
    return items


def save_roc_pr_curves(
    our_scores: np.ndarray,
    our_labels: np.ndarray,
    raddit_scores: Optional[np.ndarray],
    raddit_labels: Optional[np.ndarray],
    out_dir: Path,
    raddit_label_name: str = "Rad-DiT",
):
    """Save combined ROC and Precision-Recall comparison curves."""
    plt.figure(figsize=(8, 6))
    fpr_o, tpr_o, _ = roc_curve(our_labels, our_scores)
    auc_o = roc_auc_score(our_labels, our_scores)
    plt.plot(fpr_o, tpr_o, lw=2, label=f"Latent DDPM U-Net (AUROC = {auc_o:.4f})")

    if raddit_scores is not None and raddit_labels is not None:
        fpr_r, tpr_r, _ = roc_curve(raddit_labels, raddit_scores)
        auc_r = roc_auc_score(raddit_labels, raddit_scores)
        plt.plot(fpr_r, tpr_r, lw=2, label=f"{raddit_label_name} (AUROC = {auc_r:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Image-Level Anomaly Detection ROC Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "roc_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    prec_o, rec_o, _ = precision_recall_curve(our_labels, our_scores)
    ap_o = average_precision_score(our_labels, our_scores)
    plt.plot(rec_o, prec_o, lw=2, label=f"Latent DDPM U-Net (AUPRC = {ap_o:.4f})")

    if raddit_scores is not None and raddit_labels is not None:
        prec_r, rec_r, _ = precision_recall_curve(raddit_labels, raddit_scores)
        ap_r = average_precision_score(raddit_labels, raddit_scores)
        plt.plot(rec_r, prec_r, lw=2, label=f"{raddit_label_name} (AUPRC = {ap_r:.4f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Image-Level Anomaly Detection PR Comparison")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "pr_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_qualitative_grids(items, our_qual, raddit_qual, out_dir: Path, raddit_label_name: str):
    """Save side-by-side qualitative visual comparisons."""
    for idx in range(min(len(our_qual), len(raddit_qual))):
        _, filename, gt_mask = items[idx]
        oq = our_qual[idx]
        rq = raddit_qual[idx]

        fig, axes = plt.subplots(1, 6, figsize=(22, 4))
        axes[0].imshow(oq["original"], cmap="gray")
        axes[0].set_title("Input MRI")

        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")

        axes[2].imshow(oq["reconstruction"], cmap="gray")
        axes[2].set_title("U-Net: Recon (Healthy)")

        axes[3].imshow(oq["anomaly_map"], cmap="hot")
        axes[3].set_title(f"U-Net: Map (Score={oq['anomaly_score']:.3f})")

        axes[4].imshow(rq["reconstruction"], cmap="gray")
        axes[4].set_title(f"{raddit_label_name}: Recon")

        axes[5].imshow(rq["anomaly_map"], cmap="hot")
        axes[5].set_title(f"{raddit_label_name}: Map (Score={rq['anomaly_score']:.3f})")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        out_img = out_dir / f"comparison_{Path(filename).stem}.png"
        plt.savefig(out_img, dpi=200, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare Anomaly Detection: Ours (U-Net) vs Rad-DiT")
    parser.add_argument("--healthy-dir", default="data/processed/test/healthy")
    parser.add_argument("--anomalous-dir", default="data/processed/test/anomalous")
    parser.add_argument("--output-dir", default="results/comparison")
    parser.add_argument("--max-images", type=int, default=0, help="Limit per split (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1, help="CPU worker processes (ignored on CUDA)")

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
        "--raddit-type",
        default="jit",
        choices=["jit", "ldit-brats"],
        help="Rad-DiT architecture: pixel-space JiT or BraTS latent LightningDiT-XL/1",
    )
    parser.add_argument(
        "--raddit-checkpoint",
        default="baselines/rad-dit/checkpoints/checkpoint-last.pth",
        help="Path to Rad-DiT checkpoint file (e.g. JiT-B/16 or ldit-xl1-vavae.pt)",
    )
    parser.add_argument(
        "--raddit-vae-checkpoint",
        default="baselines/rad-dit/ldit/pretrained/vavae-imagenet256-f16d32-dinov2/vavae-imagenet256-f16d32-dinov2.pt",
        help="VA-VAE checkpoint (used with --raddit-type ldit-brats)",
    )
    parser.add_argument(
        "--raddit-latent-stats",
        default="baselines/rad-dit/ldit/pretrained/vavae-imagenet256-f16d32-dinov2/latents_stats.pt",
        help="VA-VAE latent stats (used with --raddit-type ldit-brats)",
    )
    parser.add_argument("--raddit-model", default="JiT-B/16")
    parser.add_argument("--raddit-t-start", type=float, default=0.75)
    parser.add_argument("--raddit-steps", type=int, default=50)
    parser.add_argument("--raddit-method", default="heun", choices=["heun", "euler"])
    parser.add_argument("--raddit-cfg", type=float, default=1.0)
    parser.add_argument("--raddit-class-label", type=int, default=None)
    parser.add_argument("--raddit-negative-label", type=int, default=1003)

    # Ground truth annotations for pixel evaluation
    parser.add_argument("--annotations", default="dataset/Br35H-Mask-RCNN/annotations_all.json")
    parser.add_argument("--orig-dir", default="dataset/yes")
    parser.add_argument("--max-pixel-samples", type=int, default=50)

    args = parser.parse_args()

    out_dir = ensure_dir(PROJECT_ROOT / args.output_dir)
    h_dir = str(PROJECT_ROOT / args.healthy_dir)
    a_dir = str(PROJECT_ROOT / args.anomalous_dir)

    print("=" * 70)
    print(" ANOMALY DETECTION HEAD-TO-HEAD COMPARISON: U-NET vs RAD-DIT ")
    print("=" * 70)

    pixel_items = collect_pixel_items(
        annotations_path=str(PROJECT_ROOT / args.annotations),
        orig_dir=str(PROJECT_ROOT / args.orig_dir),
        max_samples=args.max_pixel_samples,
    )

    # 1. Evaluate Our Model (Latent DDPM U-Net)
    print("\n[1/2] Evaluating Our Model (Latent DDPM U-Net)...")
    our_init_kwargs = {
        "vae_config_path": str(PROJECT_ROOT / args.vae_config),
        "ddpm_config_path": str(PROJECT_ROOT / args.ddpm_config),
        "vae_checkpoint_path": str(PROJECT_ROOT / args.vae_checkpoint),
        "ddpm_checkpoint_path": str(PROJECT_ROOT / args.ddpm_checkpoint),
    }
    our_detect_kwargs = {
        "t_start": args.our_t_start,
        "guidance_scale": args.our_guidance_scale,
        "ddim_steps": args.our_ddim_steps,
    }
    our_metrics, our_scores, our_labels, our_pixel_maps, our_qual, _ = run_model_evaluation(
        model_kind="unet",
        init_kwargs=our_init_kwargs,
        detect_kwargs=our_detect_kwargs,
        healthy_dir=h_dir,
        anomalous_dir=a_dir,
        pixel_items=pixel_items,
        name="Our Model (Latent DDPM U-Net)",
        max_images=args.max_images,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # 2. Evaluate Rad-DiT Model (JiT or LDiT-BraTS)
    raddit_desc = (
        "Latent Flow Matching (LDiT-XL/1-BraTS)"
        if args.raddit_type == "ldit-brats"
        else "Pixel Flow Matching (JiT)"
    )
    print(f"\n[2/2] Evaluating Rad-DiT Model ({raddit_desc})...")
    raddit_ckpt_path = PROJECT_ROOT / args.raddit_checkpoint
    raddit_metrics = None
    raddit_scores = None
    raddit_labels = None
    raddit_pixel_maps = []
    raddit_qual = []
    raddit_variant_used = "LDiT-XL/1-BraTS" if args.raddit_type == "ldit-brats" else args.raddit_model

    raddit_detect_kwargs = {
        "t_start": args.raddit_t_start,
        "num_steps": args.raddit_steps,
        "guidance_scale": args.raddit_cfg,
        "class_label": args.raddit_class_label,
    }
    if args.raddit_type == "ldit-brats":
        raddit_detect_kwargs["negative_label"] = args.raddit_negative_label

    if not raddit_ckpt_path.exists():
        print(f"\n[!] Rad-DiT checkpoint not found at: {raddit_ckpt_path}")
        print("    Please place the pretrained checkpoint at the specified path.")
    else:
        if args.raddit_type == "ldit-brats":
            raddit_init_kwargs = {
                "checkpoint_path": str(raddit_ckpt_path),
                "vae_checkpoint_path": str(PROJECT_ROOT / args.raddit_vae_checkpoint),
                "latent_stats_path": str(PROJECT_ROOT / args.raddit_latent_stats),
                "cfg_scale": args.raddit_cfg,
                "sampling_method": args.raddit_method,
                "num_sampling_steps": args.raddit_steps,
                "cond_label": 1001 if args.raddit_class_label is None else args.raddit_class_label,
                "negative_label": args.raddit_negative_label,
            }
            model_kind = "ldit-brats"
        else:
            raddit_init_kwargs = {
                "checkpoint_path": str(raddit_ckpt_path),
                "model_variant": args.raddit_model,
                "cfg_scale": args.raddit_cfg,
                "sampling_method": args.raddit_method,
                "num_sampling_steps": args.raddit_steps,
            }
            model_kind = "raddit"

        (
            raddit_metrics,
            raddit_scores,
            raddit_labels,
            raddit_pixel_maps,
            raddit_qual,
            detected_variant,
        ) = run_model_evaluation(
            model_kind=model_kind,
            init_kwargs=raddit_init_kwargs,
            detect_kwargs=raddit_detect_kwargs,
            healthy_dir=h_dir,
            anomalous_dir=a_dir,
            pixel_items=pixel_items,
            name=f"Rad-DiT ({raddit_variant_used})",
            max_images=args.max_images,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        if detected_variant:
            raddit_variant_used = detected_variant

    # 3. Compute Pixel-Level Metrics & Save Plots
    pixel_metrics = None
    if pixel_items and our_pixel_maps:
        y_true_pix = np.concatenate([gt.flatten() for _, _, gt in pixel_items])
        y_our_pix = np.concatenate(our_pixel_maps)
        pixel_metrics = {
            "num_samples": len(pixel_items),
            "our_pixel_auroc": float(roc_auc_score(y_true_pix, y_our_pix)),
            "our_pixel_auprc": float(average_precision_score(y_true_pix, y_our_pix)),
        }
        if raddit_pixel_maps:
            y_rad_pix = np.concatenate(raddit_pixel_maps)
            pixel_metrics["raddit_pixel_auroc"] = float(roc_auc_score(y_true_pix, y_rad_pix))
            pixel_metrics["raddit_pixel_auprc"] = float(average_precision_score(y_true_pix, y_rad_pix))
            save_qualitative_grids(
                pixel_items,
                our_qual,
                raddit_qual,
                out_dir,
                raddit_label_name=f"Rad-DiT ({raddit_variant_used})",
            )

    save_roc_pr_curves(
        our_scores,
        our_labels,
        raddit_scores,
        raddit_labels,
        out_dir,
        raddit_label_name=f"Rad-DiT ({raddit_variant_used})",
    )

    # 4. Print Comparison Table
    print("\n" + "=" * 70)
    print(f"{'METRIC':<26} | {'U-NET (OURS)':<18} | {f'RAD-DIT ({raddit_variant_used})':<18}")
    print("-" * 70)
    for m in ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "precision", "f1_score"]:
        our_val = f"{our_metrics[m]:.4f}"
        rad_val = f"{raddit_metrics[m]:.4f}" if raddit_metrics else "N/A"
        print(f"{m.upper():<26} | {our_val:<18} | {rad_val:<18}")

    print("-" * 70)
    our_lat = f"{our_metrics['avg_time_per_image_sec']:.3f}s"
    rad_lat = f"{raddit_metrics['avg_time_per_image_sec']:.3f}s" if raddit_metrics else "N/A"
    print(f"{'AVG TIME / IMAGE':<26} | {our_lat:<18} | {rad_lat:<18}")

    if pixel_metrics:
        print("-" * 70)
        our_p_auc = f"{pixel_metrics['our_pixel_auroc']:.4f}"
        rad_p_auc = (
            f"{pixel_metrics['raddit_pixel_auroc']:.4f}"
            if "raddit_pixel_auroc" in pixel_metrics
            else "N/A"
        )
        print(f"{'PIXEL AUROC':<26} | {our_p_auc:<18} | {rad_p_auc:<18}")
        our_p_ap = f"{pixel_metrics['our_pixel_auprc']:.4f}"
        rad_p_ap = (
            f"{pixel_metrics['raddit_pixel_auprc']:.4f}"
            if "raddit_pixel_auprc" in pixel_metrics
            else "N/A"
        )
        print(f"{'PIXEL AUPRC':<26} | {our_p_ap:<18} | {rad_p_ap:<18}")
    print("=" * 70)

    # 5. Save JSON report
    report = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": args.seed,
        "num_workers": args.num_workers,
        "max_images_per_split": args.max_images,
        "our_model": {
            "checkpoint": args.ddpm_checkpoint,
            "config": our_detect_kwargs,
            "image_level": our_metrics,
        },
        "raddit_model": {
            "variant": raddit_variant_used,
            "checkpoint": args.raddit_checkpoint,
            "method": args.raddit_method,
            "config": raddit_detect_kwargs if raddit_metrics else None,
            "image_level": raddit_metrics,
        },
        "pixel_level": pixel_metrics,
    }

    report_path = out_dir / "anomaly_comparison_metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[✓] Results saved to: {report_path}")
    print(f"[✓] Plots saved to:   {out_dir}")


if __name__ == "__main__":
    main()
