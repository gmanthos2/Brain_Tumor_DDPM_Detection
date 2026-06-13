"""
Generation quality metrics: FID, SSIM, LPIPS, diversity.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.utils.helpers import ensure_dir


def load_images_as_tensors(image_dir, image_size=256, max_images=None):
    image_dir = Path(image_dir)
    files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if max_images:
        files = files[:max_images]
    transform = transforms.Compose([
        transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])
    images = [transform(Image.open(f).convert("L")) for f in tqdm(files, desc="Loading")]
    return torch.stack(images)


def compute_ssim_batch(real, fake, num_pairs=200):
    """Compute SSIM between randomly sampled real-fake pairs (distribution-level)."""
    from skimage.metrics import structural_similarity as ssim
    n_real, n_fake = len(real), len(fake)
    real_idx = np.random.choice(n_real, size=num_pairs, replace=True)
    fake_idx = np.random.choice(n_fake, size=num_pairs, replace=True)
    scores = [
        ssim(
            ((real[ri].squeeze().numpy() + 1) / 2),
            ((fake[fi].squeeze().numpy() + 1) / 2),
            data_range=1.0,
        )
        for ri, fi in zip(real_idx, fake_idx)
    ]
    return {"ssim_mean": float(np.mean(scores)), "ssim_std": float(np.std(scores))}


def compute_lpips_score(real, fake, device="cuda", num_pairs=200):
    """Compute LPIPS between randomly sampled real-fake pairs (distribution-level)."""
    import lpips
    model = lpips.LPIPS(net="vgg").to(device)
    n_real, n_fake = len(real), len(fake)
    real_idx = np.random.choice(n_real, size=num_pairs, replace=True)
    fake_idx = np.random.choice(n_fake, size=num_pairs, replace=True)
    scores = []
    with torch.no_grad():
        for ri, fi in zip(real_idx, fake_idx):
            s = model(
                real[ri:ri+1].repeat(1, 3, 1, 1).to(device),
                fake[fi:fi+1].repeat(1, 3, 1, 1).to(device),
            )
            scores.append(s.item())
    return {"lpips_mean": float(np.mean(scores)), "lpips_std": float(np.std(scores))}


def compute_diversity(fake, num_pairs=100, device="cuda"):
    import lpips
    model = lpips.LPIPS(net="vgg").to(device)
    n = len(fake)
    indices = np.random.choice(n, size=(num_pairs, 2), replace=True)
    scores = []
    with torch.no_grad():
        for i, j in indices:
            if i != j:
                s = model(fake[i:i+1].repeat(1,3,1,1).to(device), fake[j:j+1].repeat(1,3,1,1).to(device))
                scores.append(s.item())
    return {"diversity_lpips_mean": float(np.mean(scores)), "diversity_lpips_std": float(np.std(scores))}


def compute_fid_manual(real_dir, fake_dir, device="cuda"):
    from torchvision.models import inception_v3
    from scipy import linalg
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity(); model.eval()
    transform = transforms.Compose([transforms.Resize(299), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def get_feats(d):
        feats = []
        for f in sorted(Path(d).iterdir()):
            if f.suffix.lower() in ('.png','.jpg','.jpeg'):
                with torch.no_grad():
                    feats.append(model(transform(Image.open(f).convert("RGB")).unsqueeze(0).to(device)).squeeze().cpu().numpy())
        return np.array(feats)
    rf, ff = get_feats(real_dir), get_feats(fake_dir)
    mu1, s1 = rf.mean(0), np.cov(rf, rowvar=False)
    mu2, s2 = ff.mean(0), np.cov(ff, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(diff @ diff + np.trace(s1 + s2 - 2*covmean))


def evaluate_generation(real_dir, fake_dir, output_dir=None, device="cuda"):
    real = load_images_as_tensors(real_dir)
    fake = load_images_as_tensors(fake_dir)
    metrics = {}
    metrics.update(compute_ssim_batch(real, fake))
    metrics.update(compute_lpips_score(real, fake, device))
    metrics.update(compute_diversity(fake, device=device))
    metrics["fid"] = compute_fid_manual(real_dir, fake_dir, device)
    print("\n=== GENERATION QUALITY ===")
    for k, v in metrics.items(): print(f"  {k}: {v:.4f}")
    if output_dir:
        import json
        p = ensure_dir(output_dir)
        with open(p / "generation_metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--fake-dir", required=True)
    parser.add_argument("--output", default="results/evaluation")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate_generation(args.real_dir, args.fake_dir, str(project_root / args.output), args.device)

if __name__ == "__main__":
    main()
