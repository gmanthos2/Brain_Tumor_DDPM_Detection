"""
Attention map visualization from U-Net self-attention layers.

Extracts and visualizes which spatial regions the model attends to
at different denoising timesteps.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.utils.helpers import ensure_dir


class AttentionExtractor:
    """Hook-based extractor for U-Net self-attention maps."""

    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks on all SelfAttention modules."""
        from src.models.unet import SelfAttention
        for name, module in self.model.named_modules():
            if isinstance(module, SelfAttention):
                hook = module.register_forward_hook(
                    lambda m, inp, out, name=name: self._hook_fn(name, m, inp, out)
                )
                self.hooks.append(hook)

    def _hook_fn(self, name, module, input, output):
        """Store the attention weights from forward pass."""
        # Re-compute attention weights for visualization
        x = input[0]
        B, C, H, W = x.shape
        h = module.norm(x)
        qkv = module.qkv(h).reshape(B, 3, module.num_heads, module.head_dim, H * W)
        q, k = qkv[:, 0], qkv[:, 1]
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        k = k.permute(0, 1, 3, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * module.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # Average over heads and batch, reshape to spatial
        attn_avg = attn_weights.mean(dim=(0, 1))  # (HW, HW)
        self.attention_maps[name] = {
            "weights": attn_avg.cpu().detach(),
            "spatial_size": (H, W),
        }

    def clear(self):
        self.attention_maps = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def visualize_attention(
    attention_data: dict,
    original_image: np.ndarray,
    output_path: str,
    query_point: tuple = None,
):
    """
    Visualize attention maps overlaid on the original image.

    Args:
        attention_data: Dict of {layer_name: {weights, spatial_size}}
        original_image: Original image (H, W)
        output_path: Path to save visualization
        query_point: Optional (y, x) point to show attention from
    """
    num_layers = len(attention_data)
    fig, axes = plt.subplots(1, num_layers + 1, figsize=(5 * (num_layers + 1), 5))

    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, (name, data) in enumerate(attention_data.items()):
        H, W = data["spatial_size"]
        attn = data["weights"]

        if query_point:
            # Show attention from a specific query point
            qy, qx = query_point
            qy_scaled = int(qy * H / original_image.shape[0])
            qx_scaled = int(qx * W / original_image.shape[1])
            idx = qy_scaled * W + qx_scaled
            attn_map = attn[idx].reshape(H, W).numpy()
        else:
            # Average attention (mean of all queries)
            attn_map = attn.mean(dim=0).reshape(H, W).numpy()

        # Upsample to original resolution
        from scipy.ndimage import zoom
        scale = original_image.shape[0] / H
        attn_map = zoom(attn_map, scale, order=1)

        axes[i + 1].imshow(original_image, cmap="gray", alpha=0.5)
        axes[i + 1].imshow(attn_map, cmap="hot", alpha=0.5)
        short_name = name.split(".")[-2] if "." in name else name
        axes[i + 1].set_title(f"Attention: {short_name}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    import argparse
    from src.inference.reconstruct import AnomalyDetector
    
    parser = argparse.ArgumentParser(description="Extract U-Net Attention Maps")
    parser.add_argument("--image", type=str, default="data/processed/test/anomalous/y0.png", help="Image to analyze")
    parser.add_argument("--output", type=str, default="results/explainability/attention_maps.png")
    parser.add_argument("--vae-config", type=str, default="configs/vae_config.yaml")
    parser.add_argument("--ddpm-config", type=str, default="configs/ddpm_config.yaml")
    parser.add_argument("--vae-checkpoint", type=str, default="checkpoints/vae/best.pt")
    parser.add_argument("--ddpm-checkpoint", type=str, default="checkpoints/ddpm/step_100000.pt")
    parser.add_argument("--t-start", type=int, default=150)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--ddim-steps", type=int, default=50)
    args = parser.parse_args()
    
    output_path = project_root / args.output
    ensure_dir(output_path.parent)
    
    print("Loading models...")
    detector = AnomalyDetector(
        vae_config_path=str(project_root / args.vae_config),
        ddpm_config_path=str(project_root / args.ddpm_config),
        vae_checkpoint_path=str(project_root / args.vae_checkpoint),
        ddpm_checkpoint_path=str(project_root / args.ddpm_checkpoint)
    )
    
    print("Registering attention hooks on U-Net...")
    extractor = AttentionExtractor(detector.diffusion.model)
    extractor.register_hooks()
    
    print(f"Running detection on {args.image}...")
    result = detector.detect(
        str(project_root / args.image),
        t_start=args.t_start,
        guidance_scale=args.guidance_scale,
        ddim_steps=args.ddim_steps
    )
    
    print("Visualizing attention maps...")
    visualize_attention(
        attention_data=extractor.attention_maps,
        original_image=result["original"].numpy(),
        output_path=str(output_path)
    )
    
    extractor.remove_hooks()
    print(f"Saved attention maps to {output_path}")

if __name__ == "__main__":
    main()
