# Diffusion-Based Brain MRI Anomaly Detection & Synthetic Generation

A Latent Diffusion Model (LDM) pipeline for detecting anomalies in brain MRI scans and generating high-quality synthetic brain images.

## Architecture

```
Brain MRI (256×256) → VAE Encoder → Latent (4×32×32) → DDPM U-Net → Denoised Latent → VAE Decoder → Output
```

**Stage 1 — VAE**: Compresses brain MRI images into a compact latent space (8× spatial downsampling)  
**Stage 2 — DDPM**: Learns the distribution of healthy brain anatomy in latent space with Classifier-Free Guidance

### Anomaly Detection
1. Encode input image to latent space
2. Add noise up to timestep `t_start`
3. Denoise conditioned on "healthy" class using DDIM
4. Compare original vs reconstruction → anomaly heatmap

### Synthetic Generation
1. Sample random noise in latent space
2. Iteratively denoise using DDIM with CFG
3. Decode to image space via VAE decoder

## Quick Start

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Preprocess dataset
python -m src.data.preprocessing --healthy-dir dataset/no --anomalous-dir dataset/yes

# 3. Train VAE (Stage 1)
python -m src.training.train_vae --config configs/vae_config.yaml

# 4. Precompute latents
python -m src.training.precompute_latents --vae-checkpoint checkpoints/vae/best.pt

# 5. Train DDPM (Stage 2)
python -m src.training.train_ddpm --config configs/ddpm_config.yaml

# 6. Anomaly detection
python -m src.inference.reconstruct --input data/processed/test/anomalous --output results/anomaly_maps

# 7. Generate synthetic images
python -m src.inference.generate --num-samples 100 --save-grid

# 8. Evaluate
python -m src.evaluation.anomaly_metrics --healthy-dir data/processed/test/healthy --anomalous-dir data/processed/test/anomalous
python -m src.evaluation.generation_metrics --real-dir data/processed/test/healthy --fake-dir results/synthetic/individual
```

## Project Structure

```
├── configs/                    # YAML configuration files
├── src/
│   ├── data/                   # Dataset & preprocessing
│   ├── models/                 # VAE, U-Net, Diffusion, Scheduler
│   ├── training/               # Training scripts (VAE, DDPM, latent precomputation)
│   ├── inference/              # Anomaly detection & generation pipelines
│   ├── evaluation/             # Metrics (AUROC, FID, SSIM, LPIPS)
│   ├── explainability/         # Attention maps, denoising trajectories, Grad-CAM
│   └── utils/                  # Config loader, logging, helpers
├── docs/                       # Ethics discussion, LaTeX report skeleton
├── dataset/                    # Raw dataset (not tracked by git)
├── data/processed/             # Preprocessed images
├── checkpoints/                # Model weights
└── results/                    # Outputs, metrics, visualizations
```

## Hardware Requirements
- **GPU**: NVIDIA RTX 4060 (8 GB VRAM) or equivalent
- **RAM**: 16 GB recommended
- Memory optimizations: mixed precision (FP16), gradient checkpointing, gradient accumulation

## Key Dependencies
- PyTorch ≥ 2.1, MONAI ≥ 1.3, torchvision, LPIPS, scikit-learn, matplotlib
