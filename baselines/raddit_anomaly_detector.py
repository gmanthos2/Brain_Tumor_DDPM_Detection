"""
baselines/raddit_anomaly_detector.py

Anomaly detection pipeline using Rad-DiT JiT (pixel-space flow matching).
Implements the same interface as src.inference.reconstruct.AnomalyDetector
for drop-in comparative evaluation.
"""

import sys
import types
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

# Add rad-dit/jit to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RADDIT_JIT_DIR = PROJECT_ROOT / "baselines" / "rad-dit" / "jit"
if str(RADDIT_JIT_DIR) not in sys.path:
    sys.path.insert(0, str(RADDIT_JIT_DIR))

try:
    from denoiser import Denoiser
    from model_jit import JiT_models
except ImportError as e:
    raise ImportError(
        f"Failed to import from rad-dit/jit. Ensure baselines/rad-dit is present. Error: {e}"
    )


class RadDiTAnomalyDetector:
    """
    Diffusion-based anomaly detector using Rad-DiT's JiT flow-matching architecture.

    Reconstructs input brain MRI images using partial forward noising followed
    by reverse ODE integration (Euler or Heun) toward clean data (t = 1.0).
    Anomalous regions are identified via pixel-level residual error |x - x_hat|.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_variant: str = "JiT-L/16",
        img_size: int = 256,
        in_channels: int = 3,
        class_num: int = 1005,
        cfg_drop_index: int = 1000,
        cfg_scale: float = 1.0,
        sampling_method: str = "heun",
        num_sampling_steps: int = 50,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.in_channels = in_channels
        self.class_num = class_num
        self.cfg_drop_index = cfg_drop_index
        self.cfg_scale = cfg_scale
        self.sampling_method = sampling_method
        self.num_sampling_steps = num_sampling_steps

        # Create dummy args object required by Denoiser.__init__
        args = types.SimpleNamespace(
            model=model_variant,
            img_size=img_size,
            in_channels=in_channels,
            class_num=class_num,
            attn_dropout=0.0,
            proj_dropout=0.0,
            label_drop_prob=0.1,
            P_mean=-0.8,
            P_std=0.8,
            t_eps=5e-2,
            noise_scale=1.0,
            cfg_drop_index=cfg_drop_index,
            ema_decay1=0.9999,
            ema_decay2=0.9996,
            sampling_method=sampling_method,
            num_sampling_steps=num_sampling_steps,
            cfg=cfg_scale,
            interval_min=0.0,
            interval_max=1.0,
        )

        self.denoiser = Denoiser(args).to(self.device)
        self.load_checkpoint(checkpoint_path)
        self.denoiser.eval()

        # Image transform (matching our pipeline: 256x256 grayscale normalized to [-1, 1])
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def load_checkpoint(self, checkpoint_path: str):
        """Loads weights from checkpoint, preferring EMA weights if present."""
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        ckpt = torch.load(ckpt_p, map_location=self.device)

        if isinstance(ckpt, dict):
            if "model_ema1" in ckpt:
                state_dict = ckpt["model_ema1"]
            elif "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Auto-detect channel mismatch if checkpoint was trained with different in_channels
        first_conv_key = "net.x_embedder.proj1.weight"
        if first_conv_key in state_dict:
            ckpt_in_channels = state_dict[first_conv_key].shape[1]
            if ckpt_in_channels != self.in_channels:
                print(
                    f"Notice: adjusting detector in_channels from {self.in_channels} to "
                    f"{ckpt_in_channels} to match checkpoint."
                )
                self.in_channels = ckpt_in_channels

        self.denoiser.load_state_dict(state_dict, strict=False)
        print(f"Loaded Rad-DiT checkpoint successfully from: {ckpt_p.name}")

    @torch.no_grad()
    def _flow_matching_forward(
        self, x_0: torch.Tensor, t_signal: float
    ) -> torch.Tensor:
        """
        Partial forward perturbation in flow matching.
        In Rad-DiT JiT:
            z = t * x_0 + (1 - t) * e
            where t in [0, 1]: t=0 is pure noise, t=1 is clean image.
        t_signal represents the retention fraction of clean image signal (e.g. 0.70).
        """
        noise = torch.randn_like(x_0) * self.denoiser.noise_scale
        z = t_signal * x_0 + (1.0 - t_signal) * noise
        return z

    @torch.no_grad()
    def _partial_ode_solve(
        self,
        z_start: torch.Tensor,
        t_start: float,
        labels: torch.Tensor,
        num_steps: int = 50,
        method: str = "heun",
    ) -> torch.Tensor:
        """
        Solves the flow matching ODE backward from t_start towards clean data t=1.0.
        """
        bsz = z_start.size(0)
        # Determine number of discrete steps proportional to remaining interval [t_start, 1.0]
        steps = max(int(num_steps * (1.0 - t_start)), 2)
        timesteps = torch.linspace(t_start, 1.0, steps + 1, device=self.device)
        timesteps = timesteps.view(-1, *([1] * z_start.ndim)).expand(-1, bsz, -1, -1, -1)

        stepper = self.denoiser._heun_step if method == "heun" else self.denoiser._euler_step

        z = z_start
        for i in range(steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)

        # Final step using Euler
        z = self.denoiser._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z.clamp(-1.0, 1.0)

    @torch.no_grad()
    def detect(
        self,
        image_path: str,
        t_start: float = 0.70,
        guidance_scale: Optional[float] = None,
        num_steps: Optional[int] = None,
        gaussian_sigma: float = 2.0,
        class_label: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Detect anomalies in a single brain MRI image.

        Args:
            image_path: Path to input brain MRI image.
            t_start: Image signal retention fraction in (0.0, 1.0].
                     Higher = less noise added (e.g. 0.70 - 0.85).
            guidance_scale: Optional CFG scale override.
            num_steps: Optional ODE steps override.
            gaussian_sigma: Sigma for post-processing Gaussian blur.
            class_label: Optional label index. Defaults to unconditional (cfg_drop_index).

        Returns:
            Dictionary matching our AnomalyDetector:
                - original: Tensor (1, H, W)
                - reconstruction: Tensor (1, H, W)
                - anomaly_map: 2D numpy array [0, 1] for visualization
                - anomaly_map_raw: 2D numpy array of absolute differences
                - anomaly_score: Float summary score
        """
        img = Image.open(image_path).convert("L")
        x_orig = self.transform(img).unsqueeze(0).to(self.device)  # (1, 1, 256, 256)

        # Handle channel expansion if model operates on 3 channels (RGB)
        if self.in_channels == 3:
            x_in = x_orig.repeat(1, 3, 1, 1)
        else:
            x_in = x_orig

        # Set up conditioning label (default: unconditional / CFG drop index)
        if class_label is None:
            label_val = self.cfg_drop_index
        else:
            label_val = class_label
        labels = torch.full((x_in.size(0),), label_val, device=self.device, dtype=torch.long)

        # Temporary CFG override if passed
        if guidance_scale is not None:
            orig_cfg = self.denoiser.cfg_scale
            self.denoiser.cfg_scale = guidance_scale

        steps = num_steps or self.num_sampling_steps

        # 1. Partial forward perturbation
        z_t = self._flow_matching_forward(x_in, t_signal=t_start)

        # 2. Denoise from t_start -> 1.0
        x_recon = self._partial_ode_solve(
            z_t, t_start=t_start, labels=labels, num_steps=steps, method=self.sampling_method
        )

        # Restore CFG scale if modified
        if guidance_scale is not None:
            self.denoiser.cfg_scale = orig_cfg

        # Collapse channels back to grayscale if needed
        if self.in_channels > 1:
            x_recon_gray = x_recon.mean(dim=1, keepdim=True)
        else:
            x_recon_gray = x_recon

        # 3. Compute anomaly residual map
        diff = torch.abs(x_orig - x_recon_gray)
        anomaly_map_raw = diff.squeeze().cpu().numpy()

        # Gaussian smoothing
        if gaussian_sigma > 0:
            anomaly_map_raw = gaussian_filter(anomaly_map_raw, sigma=gaussian_sigma)

        anomaly_score = float(anomaly_map_raw.mean())

        # Normalize to [0, 1] for visualization
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
        t_start: float = 0.70,
        guidance_scale: Optional[float] = None,
        num_steps: Optional[int] = None,
        gaussian_sigma: float = 2.0,
        class_label: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Run anomaly detection over all images in a directory."""
        from tqdm import tqdm
        dir_p = Path(image_dir)
        files = sorted([
            f for f in dir_p.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ])

        results = []
        for f in tqdm(files, desc=f"RadDiT detecting in {dir_p.name}"):
            res = self.detect(
                str(f),
                t_start=t_start,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                gaussian_sigma=gaussian_sigma,
                class_label=class_label,
            )
            res["filename"] = f.name
            results.append(res)
        return results
