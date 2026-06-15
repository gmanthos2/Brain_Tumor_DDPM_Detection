import json
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw

def create_mask_from_regions(regions, original_size, target_size):
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
            
    # Resize to match target size
    mask = mask.resize(target_size, Image.NEAREST)
    return mask

def main():
    project_root = Path(__file__).resolve().parent
    annotations_path = project_root / "dataset/Br35H-Mask-RCNN/annotations_all.json"
    original_dir = project_root / "dataset/yes"
    processed_dir = project_root / "data/processed"
    output_dir = processed_dir / "mask_overlays"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
        
    print(f"Loaded {len(annotations)} annotations.")
    
    # Pre-index processed images for fast lookup
    processed_files = list(processed_dir.rglob("*.png"))
    processed_map = {f.name: f for f in processed_files if "mask_overlays" not in str(f)}
    
    count = 0
    
    for key, data in tqdm(annotations.items(), desc="Creating overlays"):
        filename = data.get("filename")
        regions = data.get("regions", [])
        
        if not regions:
            continue
            
        orig_img_path = original_dir / filename
        if not orig_img_path.exists():
            continue
            
        # Get original image size
        try:
            with Image.open(orig_img_path) as img:
                orig_size = img.size # (width, height)
        except Exception as e:
            continue
            
        # Find processed image
        png_filename = Path(filename).with_suffix('.png').name
        if png_filename not in processed_map:
            continue
            
        processed_path = processed_map[png_filename]
        
        try:
            with Image.open(processed_path) as p_img:
                p_img = p_img.convert("RGBA")
                target_size = p_img.size
                
                mask = create_mask_from_regions(regions, orig_size, target_size)
                
                # Create a red overlay with alpha=128 only where mask is 1
                red_overlay = Image.new('RGBA', target_size, (255, 0, 0, 0))
                mask_128 = mask.point(lambda p: p * 128)
                red_overlay.putalpha(mask_128)
                
                # Alpha composite the overlay onto the original image
                result = Image.alpha_composite(p_img, red_overlay)
                
                output_path = output_dir / png_filename
                result.convert("RGB").save(output_path)
                count += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
            
    print(f"Successfully created {count} mask overlays in {output_dir}")

if __name__ == "__main__":
    main()
