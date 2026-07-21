import json
import os
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from src.inference.reconstruct import AnomalyDetector
from src.evaluation.evaluate_anomaly import create_mask_from_regions

def main():
    project_root = Path(__file__).resolve().parent
    annotations_path = project_root / "dataset/Br35H-Mask-RCNN/annotations_all.json"
    original_dir = project_root / "dataset/yes"
    processed_dir = project_root / "data/processed/test/anomalous"
    output_dir = project_root / "results/evaluation"
    
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
        
    detector = AnomalyDetector(
        vae_config_path=str(project_root / "configs/vae_config.yaml"),
        ddpm_config_path=str(project_root / "configs/ddpm_config.yaml"),
        vae_checkpoint_path=str(project_root / "checkpoints/vae/best.pt"),
        ddpm_checkpoint_path=str(project_root / "checkpoints/ddpm/step_100000.pt"),
    )
    
    count = 0
    # skip the first one to get new examples
    skip = 1
    
    for key, data in annotations.items():
        filename = data.get("filename")
        regions = data.get("regions", [])
        if not regions: continue
            
        orig_img_path = original_dir / filename
        if not orig_img_path.exists(): continue
            
        with Image.open(orig_img_path) as img:
            orig_size = img.size
            
        png_filename = Path(filename).with_suffix('.png').name
        processed_path = processed_dir / png_filename
        if not processed_path.exists(): continue
            
        if skip > 0:
            skip -= 1
            continue
            
        gt_mask = create_mask_from_regions(regions, orig_size, target_size=(256, 256))
        
        result = detector.detect(
            str(processed_path),
            t_start=150,
            guidance_scale=7.5,
            ddim_steps=50
        )
        anomaly_map = result["anomaly_map"]
        
        count += 1
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(result["original"].numpy(), cmap="gray")
        axes[0].set_title(f"Original ({filename})")
        axes[0].axis("off")
        
        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")
        
        axes[2].imshow(anomaly_map, cmap="hot")
        axes[2].set_title("Anomaly Heatmap")
        axes[2].axis("off")
        
        qual_path = output_dir / f"qualitative_example_{count}.png"
        plt.savefig(qual_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {qual_path}")
        
        if count >= 3:
            break

if __name__ == "__main__":
    main()
