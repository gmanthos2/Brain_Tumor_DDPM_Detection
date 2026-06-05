"""
Data preprocessing pipeline for brain MRI images.

Takes raw dataset (dataset/no/, dataset/yes/) and produces:
- data/processed/train/healthy/     (training: healthy only)
- data/processed/val/healthy/       (validation: healthy only)
- data/processed/test/healthy/      (test: healthy subset)
- data/processed/test/anomalous/    (test: tumor subset)
- data/processed/metadata.csv       (image manifest with labels)
"""

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def preprocess_image(
    image_path: Path,
    target_size: int = 256,
) -> np.ndarray:
    """
    Load, convert to grayscale, resize, and normalize a brain MRI image.

    Returns:
        np.ndarray: Preprocessed image as uint8 (0–255), shape (H, W).
    """
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(img)


def split_dataset(
    healthy_dir: Path,
    anomalous_dir: Path,
    output_base: Path,
    target_size: int = 256,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Split and preprocess the dataset into train/val/test folders.

    Training and validation contain ONLY healthy images.
    Test contains both healthy and anomalous images.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    rng = np.random.RandomState(seed)

    # Gather all image paths
    healthy_files = sorted([
        f for f in healthy_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    ])
    anomalous_files = sorted([
        f for f in anomalous_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    ])

    print(f"Found {len(healthy_files)} healthy images")
    print(f"Found {len(anomalous_files)} anomalous images")

    # Shuffle healthy images
    rng.shuffle(healthy_files)

    # Split healthy images
    n_healthy = len(healthy_files)
    n_train = int(n_healthy * train_ratio)
    n_val = int(n_healthy * val_ratio)

    train_files = healthy_files[:n_train]
    val_files = healthy_files[n_train:n_train + n_val]
    test_healthy_files = healthy_files[n_train + n_val:]

    # Shuffle anomalous images and take a subset for testing
    rng.shuffle(anomalous_files)
    # Use same number of anomalous as test healthy for balanced evaluation
    n_test_anomalous = min(len(anomalous_files), len(test_healthy_files) * 2)
    test_anomalous_files = anomalous_files[:n_test_anomalous]

    print(f"\nSplit:")
    print(f"  Train (healthy):        {len(train_files)}")
    print(f"  Validation (healthy):   {len(val_files)}")
    print(f"  Test (healthy):         {len(test_healthy_files)}")
    print(f"  Test (anomalous):       {len(test_anomalous_files)}")

    # Create output directories
    dirs = {
        "train_healthy": output_base / "train" / "healthy",
        "val_healthy": output_base / "val" / "healthy",
        "test_healthy": output_base / "test" / "healthy",
        "test_anomalous": output_base / "test" / "anomalous",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Process and save images
    metadata = []

    splits = [
        ("train_healthy", train_files, 0),
        ("val_healthy", val_files, 0),
        ("test_healthy", test_healthy_files, 0),
        ("test_anomalous", test_anomalous_files, 1),
    ]

    for split_name, files, label in splits:
        split_type = split_name.split("_")[0]  # train, val, or test
        print(f"\nProcessing {split_name}...")
        for f in tqdm(files, desc=split_name):
            img_array = preprocess_image(f, target_size=target_size)
            out_name = f"{f.stem}.png"
            out_path = dirs[split_name] / out_name

            # Save as lossless PNG
            Image.fromarray(img_array).save(out_path)

            metadata.append({
                "filename": out_name,
                "split": split_type,
                "label": label,
                "label_name": "healthy" if label == 0 else "anomalous",
                "original_path": str(f),
                "processed_path": str(out_path),
            })

    # Save metadata CSV
    csv_path = output_base / "metadata.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\nMetadata saved to {csv_path}")
    print(f"Total processed: {len(metadata)} images")

    return {
        "train": len(train_files),
        "val": len(val_files),
        "test_healthy": len(test_healthy_files),
        "test_anomalous": len(test_anomalous_files),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess brain MRI dataset")
    parser.add_argument(
        "--healthy-dir", type=str, default="dataset/no",
        help="Path to healthy images directory"
    )
    parser.add_argument(
        "--anomalous-dir", type=str, default="dataset/yes",
        help="Path to anomalous (tumor) images directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--image-size", type=int, default=256,
        help="Target image size (square)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits"
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parents[2]
    healthy_dir = project_root / args.healthy_dir
    anomalous_dir = project_root / args.anomalous_dir
    output_dir = project_root / args.output_dir

    if not healthy_dir.exists():
        raise FileNotFoundError(f"Healthy directory not found: {healthy_dir}")
    if not anomalous_dir.exists():
        raise FileNotFoundError(f"Anomalous directory not found: {anomalous_dir}")

    stats = split_dataset(
        healthy_dir=healthy_dir,
        anomalous_dir=anomalous_dir,
        output_base=output_dir,
        target_size=args.image_size,
        seed=args.seed,
    )

    print("\n✓ Preprocessing complete!")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
