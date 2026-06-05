"""
U-Net noise prediction network for latent diffusion.

Architecture: Encoder-decoder with skip connections, sinusoidal time
embeddings, class conditioning, and self-attention at lower resolutions.
Designed to fit within 8 GB VRAM with gradient checkpointing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, List


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResNetBlock(nn.Module):
    """
    ResNet block with time and class conditioning.

    Applies: GroupNorm → SiLU → Conv → (+ time/class embed) → GroupNorm → SiLU → Dropout → Conv → (+ skip)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        num_groups: int = 32,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection
        self.skip_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        t_emb_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_emb_proj

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip_conv(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with GroupNorm."""

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Scaled dot-product attention
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return x + self.proj(attn)


class DownBlock(nn.Module):
    """Encoder block: ResNet blocks + optional attention + downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for i in range(num_res_blocks):
            ch_in = in_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResNetBlock(ch_in, out_channels, time_emb_dim, dropout)
            )
            self.attn_blocks.append(
                SelfAttention(out_channels, num_heads) if use_attention else nn.Identity()
            )

        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor
    ) -> tuple:
        skip_connections = []
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x, t_emb)
            x = attn_block(x)
            skip_connections.append(x)
        x = self.downsample(x)
        return x, skip_connections


class UpBlock(nn.Module):
    """Decoder block: upsample + ResNet blocks (with skip connections) + optional attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for i in range(num_res_blocks):
            ch_in = (in_channels + skip_channels) if i == 0 else out_channels
            self.res_blocks.append(
                ResNetBlock(ch_in, out_channels, time_emb_dim, dropout)
            )
            self.attn_blocks.append(
                SelfAttention(out_channels, num_heads) if use_attention else nn.Identity()
            )

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, skip_connections: list
    ) -> torch.Tensor:
        x = self.upsample(x)
        # Take the last skip connection (matches the resolution)
        skip = skip_connections.pop()
        x = torch.cat([x, skip], dim=1)

        for i, (res_block, attn_block) in enumerate(zip(self.res_blocks, self.attn_blocks)):
            x = res_block(x, t_emb)
            x = attn_block(x)
            # Concatenate remaining skip connections for subsequent res blocks
            if i == 0 and len(skip_connections) > 0:
                # remaining skips handled by having correct channel count
                pass
        return x


class MiddleBlock(nn.Module):
    """Middle block: ResNet → Attention → ResNet."""

    def __init__(self, channels: int, time_emb_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.res1 = ResNetBlock(channels, channels, time_emb_dim, dropout)
        self.attn = SelfAttention(channels, num_heads)
        self.res2 = ResNetBlock(channels, channels, time_emb_dim, dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t_emb)
        x = self.attn(x)
        x = self.res2(x, t_emb)
        return x


class UNet(nn.Module):
    """
    U-Net for latent diffusion noise prediction.

    Architecture for 32×32 latent space (4 channels):
    - Encoder: [32→16→8→4] with channel multipliers [1,2,4,4]
    - Middle: self-attention at 4×4
    - Decoder: [4→8→16→32] symmetric

    Supports class-conditional generation via Classifier-Free Guidance.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: list = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attention_resolutions: list = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.0,
        num_classes: int = 2,
        class_dropout_prob: float = 0.1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes

        time_emb_dim = base_channels * 4

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Class embedding (num_classes + 1 for unconditional/null token)
        self.class_embed = nn.Embedding(num_classes + 1, time_emb_dim)

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Build channel list
        channels = [base_channels * m for m in channel_multipliers]
        current_res = 32  # Starting latent resolution

        # Encoder
        self.down_blocks = nn.ModuleList()
        ch_in = base_channels
        for i, ch_out in enumerate(channels):
            use_attn = current_res in attention_resolutions
            self.down_blocks.append(
                DownBlock(
                    ch_in, ch_out, time_emb_dim,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attn,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )
            ch_in = ch_out
            current_res //= 2

        # Middle
        self.middle = MiddleBlock(channels[-1], time_emb_dim, num_heads, dropout)

        # Decoder (reverse order)
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for i, ch_out in enumerate(reversed_channels):
            ch_in_up = reversed_channels[i]
            skip_ch = ch_out  # Skip connection from corresponding encoder level
            out_ch = reversed_channels[i + 1] if i + 1 < len(reversed_channels) else base_channels
            current_res *= 2
            use_attn = current_res in attention_resolutions

            self.up_blocks.append(
                UpBlock(
                    ch_in_up, out_ch, skip_ch,
                    time_emb_dim,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attn,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )

        # Output projection
        self.output_norm = nn.GroupNorm(min(32, base_channels), base_channels)
        self.output_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

        # Initialize output conv to zero for stable training start
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict noise given noisy latent, timestep, and optional class label.

        Args:
            x: Noisy latent (B, 4, 32, 32)
            t: Timesteps (B,)
            class_labels: Class labels (B,), 0=healthy, 1=anomalous.
                          None or num_classes = unconditional.

        Returns:
            Predicted noise (B, 4, 32, 32)
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Class embedding with dropout for CFG
        if class_labels is not None:
            if self.training:
                # Randomly drop class labels for CFG training
                mask = torch.rand(class_labels.shape[0], device=class_labels.device) < self.class_dropout_prob
                class_labels = class_labels.clone()
                class_labels[mask] = self.num_classes  # Null class token
            c_emb = self.class_embed(class_labels)
        else:
            # Unconditional: use null token
            c_emb = self.class_embed(
                torch.full((x.shape[0],), self.num_classes, device=x.device, dtype=torch.long)
            )

        t_emb = t_emb + c_emb

        # Input
        h = self.input_conv(x)

        # Encoder path
        all_skips = []
        for down_block in self.down_blocks:
            if self.use_gradient_checkpointing and self.training:
                h, skips = checkpoint(down_block, h, t_emb, use_reentrant=False)
            else:
                h, skips = down_block(h, t_emb)
            all_skips.extend(skips)

        # Middle
        if self.use_gradient_checkpointing and self.training:
            h = checkpoint(self.middle, h, t_emb, use_reentrant=False)
        else:
            h = self.middle(h, t_emb)

        # Decoder path
        for up_block in self.up_blocks:
            # Each up block consumes skip connections from the matching encoder level
            num_skips = len(up_block.res_blocks)
            block_skips = all_skips[-num_skips:]
            all_skips = all_skips[:-num_skips]

            if self.use_gradient_checkpointing and self.training:
                h = checkpoint(up_block, h, t_emb, block_skips, use_reentrant=False)
            else:
                h = up_block(h, t_emb, block_skips)

        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)

        return h


def build_unet(config) -> UNet:
    """Build a U-Net from config object."""
    model_cfg = config.model
    return UNet(
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        base_channels=model_cfg.base_channels,
        channel_multipliers=model_cfg.channel_multipliers,
        num_res_blocks=model_cfg.num_res_blocks,
        attention_resolutions=model_cfg.attention_resolutions,
        num_heads=model_cfg.num_heads,
        dropout=model_cfg.dropout,
        num_classes=model_cfg.num_classes,
        class_dropout_prob=model_cfg.class_dropout_prob,
        use_gradient_checkpointing=model_cfg.use_gradient_checkpointing,
    )
