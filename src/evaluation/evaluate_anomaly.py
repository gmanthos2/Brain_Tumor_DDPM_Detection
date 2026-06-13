import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from src.inference.reconstruct import AnomalyDetector
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("evaluate_anomaly")

def create_mask_from_regions(regions, original_size, target_size=(256, 256)):
    """
    Generate a binary mask from VIA JSON regions.
    original_size: (width, height)
    """
    mask = Image.new('L', original_size, 0)
    draw = ImageDraw.Draw(mask)
    
    for region in regions:
        shape_attr = region.get("shape_attributes", {})
        shape_name = shape_attr.get("name")
        
        if shape_name == "polygon":
            x_coords = shape_attr.get("all_points_x", [])
            y_coords = shape_attr.get("all_points_y", [])
            if x_coords and y_coords:
                xy = list(zip(x_coords, y_coords))
                draw.polygon(xy, fill=1)
                
        elif shape_name == "ellipse":
            cx, cy = shape_attr.get("cx", 0), shape_attr.get("cy", 0)
            rx, ry = shape_attr.get("rx", 0), shape_attr.get("ry", 0)
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=1)
            
        elif shape_name == "circle":
            cx, cy = shape_attr.get("cx", 0), shape_attr.get("cy", 0)
            r = shape_attr.get("r", 0)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=1)
            
    # Resize to match anomaly map dimensions
    mask = mask.resize(target_size, Image.NEAREST)
    return np.array(mask)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Anomaly Detection using pixel-level AUROC")
    parser.add_argument("--annotations", type=str, default="dataset/Br35H-Mask-RCNN/annotations_all.json")
    parser.add_argument("--image-dir", type=str, default="dataset/yes")
    parser.add_argument("--output-dir", type=str, default="results/evaluation")
    parser.add_argument("--max-samples", type=int, default=100, help="Max images to evaluate to save time")
    parser.add_argument("--t-start", type=int, default=300, help="Noise level for DDIM inversion/reconstruction")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddpm-checkpoint", type=str, default="checkpoints/ddpm/step_130000.pt")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[2]
    annotations_path = project_root / args.annotations
    image_dir = project_root / args.image_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Annotations
    logger.info(f"Loading annotations from {annotations_path}")
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
        
    # 2. Initialize Detector
    logger.info("Initializing Anomaly Detector...")
    detector = AnomalyDetector(
        vae_config_path=str(project_root / "configs/vae_config.yaml"),
        ddpm_config_path=str(project_root / "configs/ddpm_config.yaml"),
        vae_checkpoint_path=str(project_root / "checkpoints/vae/best.pt"),
        ddpm_checkpoint_path=str(project_root / args.ddpm_checkpoint),
    )
    
    all_masks = []
    all_anomaly_maps = []
    
    evaluated_count = 0
    
    # 3. Evaluate Images
    logger.info("Starting evaluation...")
    for key, data in tqdm(annotations.items(), desc="Evaluating Images"):
        filename = data.get("filename")
        regions = data.get("regions", [])
        
        if not regions:
            continue
            
        img_path = image_dir / filename
        if not img_path.exists():
            continue
            
        # Get original image size
        try:
            with Image.open(img_path) as img:
                orig_size = img.size # (width, height)
        except Exception as e:
            logger.warning(f"Failed to open {img_path}: {e}")
            continue
            
        # Generate Ground Truth Mask (1 for tumor, 0 for background)
        gt_mask = create_mask_from_regions(regions, orig_size, target_size=(256, 256))
        
        # Generate Anomaly Map from DDPM
        try:
            result = detector.detect(
                str(img_path),
                t_start=args.t_start,
                ddim_steps=args.ddim_steps
            )
            anomaly_map = result["anomaly_map"] # (256, 256) numpy array
        except Exception as e:
            logger.warning(f"Detection failed for {img_path}: {e}")
            continue
            
        all_masks.append(gt_mask.flatten())
        all_anomaly_maps.append(anomaly_map.flatten())
        
        evaluated_count += 1
        if args.max_samples and evaluated_count >= args.max_samples:
            break
            
    if not all_masks:
        logger.error("No images were successfully evaluated.")
        return
        
    # Concatenate all pixels
    logger.info("Computing AUROC...")
    y_true = np.concatenate(all_masks)
    y_scores = np.concatenate(all_anomaly_maps)
    
    # Calculate AUROC
    auroc = roc_auc_score(y_true, y_scores)
    logger.info(f"Pixel-level AUROC: {auroc:.4f}")
    
    # Calculate ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Pixel-Level Anomaly Detection ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plot_path = output_dir / "roc_curve_pixel_level.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved ROC curve to {plot_path}")
    
    # Save a qualitative example
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(result["original"].numpy(), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")
    
    axes[2].imshow(anomaly_map, cmap="hot")
    axes[2].set_title("Anomaly Heatmap")
    axes[2].axis("off")
    
    qual_path = output_dir / "qualitative_example.png"
    plt.savefig(qual_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved qualitative example to {qual_path}")

if __name__ == "__main__":
    main()
