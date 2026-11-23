"""
baselines/ldit_brats_anomaly_detector.py

Anomaly detection pipeline using Rad-DiT's BraTS-trained Latent Diffusion
Transformer (LightningDiT-XL/1 + VA-VAE, flow matching in 32x16x16 latent space).

Conditioned on BraTS class `1001` (`no_tumor`) with Classifier-Free Guidance
against `1003` (CFG-null) or `1002` (`has_tumor`, contrastive guidance) so that
tumorous regions are reconstructed as healthy brain tissue.

Implements the same interface as src.inference.reconstruct.AnomalyDetector.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RADDIT_LDIT_DIR = PROJECT_ROOT / "baselines" / "rad-dit" / "ldit"
if str(RADDIT_LDIT_DIR) not in sys.path:
    sys.path.insert(0, str(RADDIT_LDIT_DIR))

from models.lightningdit import LightningDiT_models
from tokenizer.autoencoder import AutoencoderKL


class LDiTBratsAnomalyDetector:
    """
    Latent flow-matching anomaly detector using Rad-DiT's BraTS-trained
    LightningDiT-XL/1 + VA-VAE (f16d32).

    Pipeline:
    1. Encode 256x256 MRI -> 32x16x16 VA-VAE latent z_raw
    2. Normalize latent with VA-VAE ImageNet stats
    3. Partial linear flow noising: z_t = t_start * z_1 + (1 - t_start) * eps
    4. Integrate velocity ODE from t_start -> 1.0 conditioned on `no_tumor` (1001)
    5. Denormalize and decode via VA-VAE -> healthy reconstruction x_hat
    6. Compute smoothed pixel residual |x - x_hat|
    """

    NO_TUMOR_LABEL = 1001
    HAS_TUMOR_LABEL = 1002
    UNCOND_LABEL = 1003

    def __init__(
        self,
        checkpoint_path: str = "baselines/rad-dit/ldit/checkpoints/ldit-xl1-vavae.pt",
        vae_checkpoint_path: str = "baselines/rad-dit/ldit/pretrained/vavae-imagenet256-f16d32-dinov2/vavae-imagenet256-f16d32-dinov2.pt",
        latent_stats_path: str = "baselines/rad-dit/ldit/pretrained/vavae-imagenet256-f16d32-dinov2/latents_stats.pt",
        img_size: int = 256,
        cfg_scale: float = 3.0,
        sampling_method: str = "heun",
        num_sampling_steps: int = 50,
        cond_label: int = 1001,
        negative_label: int = 1003,
        cfg_interval_start: float = 0.11,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.cfg_scale = cfg_scale
        self.sampling_method = sampling_method
        self.num_sampling_steps = num_sampling_steps
        self.cond_label = cond_label
        self.negative_label = negative_label
        self.cfg_interval_start = cfg_interval_start
        self.model_variant = "LDiT-XL/1-BraTS"

        # Resolve paths relative to project root if needed
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.is_absolute():
            ckpt_p = PROJECT_ROOT / ckpt_p
        vae_p = Path(vae_checkpoint_path)
        if not vae_p.is_absolute():
            vae_p = PROJECT_ROOT / vae_p
        stats_p = Path(latent_stats_path)
        if not stats_p.is_absolute():
            stats_p = PROJECT_ROOT / stats_p

        if not ckpt_p.exists():
            raise FileNotFoundError(f"LDiT checkpoint not found: {ckpt_p}")
        if not vae_p.exists():
            raise FileNotFoundError(f"VA-VAE checkpoint not found: {vae_p}")
        if not stats_p.exists():
            raise FileNotFoundError(f"VA-VAE latent stats not found: {stats_p}")

        # 1. Load VA-VAE (f16d32)
        self.vae = AutoencoderKL(
            embed_dim=32,
            ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=str(vae_p),
            model_type="vavae",
            use_ema=False,
        ).eval().to(self.device)

        # 2. Load latent normalization stats
        stats = torch.load(stats_p, map_location="cpu", weights_only=False)
        mean = stats["mean"]
        std = stats["std"]
        if mean.ndim == 1:
            mean = mean.view(1, -1, 1, 1)
        elif mean.ndim == 3:
            mean = mean.unsqueeze(0)
        if std.ndim == 1:
            std = std.view(1, -1, 1, 1)
        elif std.ndim == 3:
            std = std.unsqueeze(0)
        self.latent_mean = mean.float().to(self.device)
        self.latent_std = std.float().to(self.device)
        self.latent_multiplier = 1.0

        # 3. Load LightningDiT-XL/1 checkpoint (prefer EMA weights)
        ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})

        model_type = model_cfg.get("model_type", "LightningDiT-XL/1")
        in_chans = model_cfg.get("in_chans", 32)
        num_classes = data_cfg.get("num_classes", 1003)
        self.latent_multiplier = float(data_cfg.get("latent_multiplier", 1.0))

        self.dit = LightningDiT_models[model_type](
            input_size=img_size // 16,
            num_classes=num_classes,
            use_qknorm=model_cfg.get("use_qknorm", True),
            use_swiglu=model_cfg.get("use_swiglu", True),
            use_rope=model_cfg.get("use_rope", True),
            use_rmsnorm=model_cfg.get("use_rmsnorm", True),
            wo_shift=model_cfg.get("wo_shift", False),
            in_channels=in_chans,
            learn_sigma=model_cfg.get("learn_sigma", False),
        )

        if isinstance(ckpt, dict):
            if "ema" in ckpt:
                state_dict = ckpt["ema"]
            elif "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
        del ckpt

        # Strip DDP 'module.' prefix if present
        state_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        missing, unexpected = self.dit.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"LDiT load warnings — missing: {len(missing)}, unexpected: {len(unexpected)}")

        # Keep DiT in bfloat16 on CUDA to fit comfortably within 8 GB VRAM
        self.use_bf16 = self.device.type == "cuda"
        if self.use_bf16:
            self.dit = self.dit.to(device=self.device, dtype=torch.bfloat16)
        else:
            self.dit = self.dit.to(self.device)
        self.dit.eval()

        print(
            f"Loaded {model_type} (BraTS) + VA-VAE from {ckpt_p.name} on {self.device} "
            f"(cond={self.cond_label}, neg={self.negative_label}, cfg={self.cfg_scale})"
        )

        # Preprocessing: grayscale -> 256x256 -> [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def normalize_latents(self, z_raw: torch.Tensor) -> torch.Tensor:
        return (z_raw - self.latent_mean) / self.latent_std * self.latent_multiplier

    def denormalize_latents(self, z_norm: torch.Tensor) -> torch.Tensor:
        return (z_norm * self.latent_std) / self.latent_multiplier + self.latent_mean

    @torch.no_grad()
    def _predict_velocity(
        self,
        z: torch.Tensor,
        t_scalar: float,
        y_cond: torch.Tensor,
        y_neg: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        bsz = z.size(0)
        t_vec = torch.full((bsz,), t_scalar, device=self.device, dtype=torch.float32)

        apply_cfg = (
            cfg_scale != 1.0
            and t_scalar >= self.cfg_interval_start
            and not bool(torch.all(y_cond == y_neg).item())
        )

        dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        with torch.amp.autocast(device_type=self.device.type, dtype=dtype, enabled=self.use_bf16):
            if apply_cfg:
                z_in = torch.cat([z, z], dim=0).to(dtype)
                t_in = torch.cat([t_vec, t_vec], dim=0).to(dtype)
                y_in = torch.cat([y_cond, y_neg], dim=0)
                v_out = self.dit(z_in, t_in, y_in).float()
                v_cond, v_neg = v_out.chunk(2, dim=0)
                return v_neg + cfg_scale * (v_cond - v_neg)
            else:
                return self.dit(z.to(dtype), t_vec.to(dtype), y_cond).float()

    @torch.no_grad()
    def _partial_ode_solve(
        self,
        z_start: torch.Tensor,
        t_start: float,
        y_cond: torch.Tensor,
        y_neg: torch.Tensor,
        cfg_scale: float,
        num_steps: int = 50,
        method: str = "heun",
    ) -> torch.Tensor:
        steps = max(int(num_steps * (1.0 - t_start)), 2)
        ts = torch.linspace(t_start, 1.0, steps + 1).tolist()

        z = z_start.float()
        for i in range(steps - 1):
            t = ts[i]
            t_next = ts[i + 1]
            dt = t_next - t
            v_t = self._predict_velocity(z, t, y_cond, y_neg, cfg_scale)
            if method == "heun":
                z_euler = z + dt * v_t
                v_next = self._predict_velocity(z_euler, t_next, y_cond, y_neg, cfg_scale)
                z = z + dt * 0.5 * (v_t + v_next)
            else:
                z = z + dt * v_t

        # Final Euler step to t = 1.0
        t = ts[-2]
        dt = ts[-1] - t
        v_t = self._predict_velocity(z, t, y_cond, y_neg, cfg_scale)
        z = z + dt * v_t
        return z

    @torch.no_grad()
    def detect(
        self,
        image_path: str,
        t_start: float = 0.55,
        guidance_scale: Optional[float] = None,
        num_steps: Optional[int] = None,
        gaussian_sigma: float = 2.0,
        class_label: Optional[int] = None,
        negative_label: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Detect anomalies in a single brain MRI image by encoding to VA-VAE latent space,
        partially noising to t_start, and integrating toward t=1.0 conditioned on no_tumor (1001).
        """
        img = Image.open(image_path).convert("L")
        x_orig = self.transform(img).unsqueeze(0).to(self.device)  # (1, 1, 256, 256)
        x_rgb = x_orig.repeat(1, 3, 1, 1)

        # 1. Encode to VA-VAE latent and normalize
        z_raw = self.vae.encode(x_rgb).mode()  # (1, 32, 16, 16)
        z_1 = self.normalize_latents(z_raw)

        # 2. Linear flow forward noising: z_t = t * z_1 + (1 - t) * eps
        eps = torch.randn_like(z_1)
        z_t = t_start * z_1 + (1.0 - t_start) * eps

        # 3. Integrate ODE from t_start -> 1.0 conditioned on healthy (no_tumor = 1001)
        cond_val = self.cond_label if class_label is None else class_label
        neg_val = self.negative_label if negative_label is None else negative_label
        cfg = self.cfg_scale if guidance_scale is None else guidance_scale
        steps = num_steps or self.num_sampling_steps

        y_cond = torch.full((1,), cond_val, device=self.device, dtype=torch.long)
        y_neg = torch.full((1,), neg_val, device=self.device, dtype=torch.long)

        z_recon_norm = self._partial_ode_solve(
            z_t,
            t_start=t_start,
            y_cond=y_cond,
            y_neg=y_neg,
            cfg_scale=cfg,
            num_steps=steps,
            method=self.sampling_method,
        )

        # 4. Denormalize & decode via VA-VAE
        z_recon_raw = self.denormalize_latents(z_recon_norm)
        x_recon_rgb = self.vae.decode(z_recon_raw).clamp(-1.0, 1.0)
        x_recon_gray = x_recon_rgb.mean(dim=1, keepdim=True)

        # 5. Compute anomaly residual map
        diff = torch.abs(x_orig - x_recon_gray)
        anomaly_map_raw = diff.squeeze().cpu().numpy()

        if gaussian_sigma > 0:
            anomaly_map_raw = gaussian_filter(anomaly_map_raw, sigma=gaussian_sigma)

        anomaly_score = float(anomaly_map_raw.mean())

        map_min, map_max = anomaly_map_raw.min(), anomaly_map_raw.max()
        if map_max > map_min:
            anomaly_map_vis = (anomaly_map_raw - map_min) / (map_max - map_min)
        else:
            anomaly_map_vis = anomaly_map_raw

        return {
            "original": x_orig.squeeze().cpu(),
            "reconstruction": x_recon_gray.squeeze().cpu(),
            "anomaly_map": anomaly_map_vis,
            "anomaly_map_raw": anomaly_map_raw,
            "anomaly_score": anomaly_score,
        }

    @torch.no_grad()
    def detect_batch(
        self,
        image_dir: str,
        t_start: float = 0.55,
        guidance_scale: Optional[float] = None,
        num_steps: Optional[int] = None,
        gaussian_sigma: float = 2.0,
        class_label: Optional[int] = None,
        negative_label: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        from tqdm import tqdm
        dir_p = Path(image_dir)
        files = sorted([
            f for f in dir_p.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ])
        results = []
        for f in tqdm(files, desc=f"LDiT-BraTS detecting in {dir_p.name}"):
            res = self.detect(
                str(f),
                t_start=t_start,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                gaussian_sigma=gaussian_sigma,
                class_label=class_label,
                negative_label=negative_label,
            )
            res["filename"] = f.name
            results.append(res)
        return results
