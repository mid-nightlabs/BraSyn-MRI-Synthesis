"""
Attention U-Net for brain MRI modality synthesis (BraSyn).
Replaces plain U-Net skip connections with attention gates.
Supports:
  - Attention gates on all skip connections (Attention U-Net, Oktay et al. 2018)
  - Deep supervision (auxiliary losses from intermediate decoder levels)
  - 2.5D context via configurable in_channels
  - Configurable depth, base_channels, dropout

References:
  - U-Net: Ronneberger et al., MICCAI 2015
  - Attention U-Net: Oktay et al., arXiv:1804.03999
  - BraSyn: arXiv:2305.09011
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout),
        )

    def forward(self, x):
        return self.pool_conv(x)


class AttentionGate(nn.Module):
    """
    Soft attention gate on skip connections (Oktay et al. 2018).

    g  = gating signal from decoder (lower resolution, richer semantics)
    x  = skip connection from encoder (higher resolution)
    Produces a spatial attention map α ∈ [0,1]^{H×W} that gates which
    encoder features are forwarded to the decoder.

    NLP analogy: like cross-attention where the decoder query (g) attends
    to encoder keys/values (x) — but in 2D spatial space.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
        # Store last computed attention map for visualization
        self._last_attention: torch.Tensor | None = None

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g is from decoder (may be smaller); upsample to match x
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        g1 = self.W_g(g_up)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)  # (B, 1, H, W)
        self._last_attention = alpha.detach()
        return x * alpha  # gated skip connection


class Up(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout: float = 0.0,
        use_attention: bool = True,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.attn = AttentionGate(F_g=in_ch // 2, F_l=in_ch // 2, F_int=in_ch // 4) if use_attention else None
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad to match encoder skip size
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        if self.attn is not None:
            x2 = self.attn(g=x1, x=x2)  # gate the skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """
    Attention U-Net for MRI slice synthesis.

    Args:
        in_channels:      Number of input channels (3 modalities × context width).
        out_channels:     Number of output channels (1 = synthesized modality).
        base_channels:    Feature channels at first level (doubles each level down).
        depth:            Number of encoder/decoder levels (3–5).
        dropout:          Spatial dropout rate in DoubleConv blocks.
        use_attention:    Enable attention gates on all skip connections.
        deep_supervision: Add auxiliary output heads at each decoder level.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
        use_attention: bool = True,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.depth = depth
        self.deep_supervision = deep_supervision
        ch = [base_channels * (2**i) for i in range(depth + 1)]

        self.inc = DoubleConv(in_channels, ch[0], dropout)
        self.down = nn.ModuleList([Down(ch[i], ch[i + 1], dropout) for i in range(depth)])
        self.up = nn.ModuleList(
            [Up(ch[depth - i], ch[depth - i - 1], dropout, use_attention) for i in range(depth)]
        )
        self.outc = nn.Conv2d(ch[0], out_channels, 1)

        # Deep supervision: auxiliary heads at each decoder level
        if deep_supervision:
            self.aux_heads = nn.ModuleList(
                [nn.Conv2d(ch[depth - i - 1], out_channels, 1) for i in range(depth - 1)]
            )
        else:
            self.aux_heads = None

    def forward(self, x: torch.Tensor):
        xs = [self.inc(x)]
        for i in range(self.depth):
            xs.append(self.down[i](xs[-1]))

        feat = xs[-1]
        aux_outputs = []
        for i in range(self.depth):
            feat = self.up[i](feat, xs[self.depth - 1 - i])
            if self.deep_supervision and self.aux_heads and i < len(self.aux_heads):
                # Upsample aux output to full resolution
                aux = self.aux_heads[i](feat)
                aux = F.interpolate(aux, size=xs[0].shape[2:], mode="bilinear", align_corners=False)
                aux_outputs.append(aux)

        out = self.outc(feat)
        if self.deep_supervision and aux_outputs:
            return out, aux_outputs  # tuple: (main, [aux1, aux2, ...])
        return out

    def get_attention_maps(self) -> list:
        """Return list of last attention maps from all Up blocks (for visualization)."""
        maps = []
        for up_block in self.up:
            if up_block.attn is not None and up_block.attn._last_attention is not None:
                maps.append(up_block.attn._last_attention)
        return maps


def build_model(
    in_channels: int = 3,
    out_channels: int = 1,
    base_channels: int = 32,
    depth: int = 4,
    dropout: float = 0.1,
    use_attention: bool = True,
    deep_supervision: bool = False,
    context_slices: int = 0,
    device: str = "cuda",
) -> UNet2D:
    """
    Build and return an Attention U-Net.

    context_slices: if > 0, in_channels is automatically set to
    3 * (2*context_slices + 1) to match the 2.5D dataset output.
    Explicit in_channels overrides this.
    """
    if context_slices > 0 and in_channels == 3:
        in_channels = 3 * (2 * context_slices + 1)
    model = UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        depth=depth,
        dropout=dropout,
        use_attention=use_attention,
        deep_supervision=deep_supervision,
    )
    return model.to(device)
