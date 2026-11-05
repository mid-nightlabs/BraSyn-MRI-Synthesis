"""
PatchGAN discriminator for BraSyn adversarial MRI synthesis.

Architecture: 70×70 PatchGAN (Isola et al., pix2pix, CVPR 2017).
Input: concatenation of (input_modalities [B,3,H,W] + target [B,1,H,W]) = [B,4,H,W].
Output: patch-level real/fake logits [B,1,H',W'] — no sigmoid (use BCEWithLogitsLoss).

Conditioning on the inputs makes the discriminator judge:
  "does this synthesis look realistic GIVEN these three input scans?"
— not just "does this look like an MRI in general?".

NLP analogy: like a conditional language model discriminator that scores
whether a generated answer is fluent *given* the question context.
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Conv2d → [BatchNorm] → LeakyReLU (standard PatchGAN block)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 2,
        norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not norm)
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.

    Args:
        in_channels:  input modalities + target (default: 3 + 1 = 4).
        base_channels: feature map size at first layer (default: 64).
        n_layers:     number of strided conv layers controlling receptive field.

    Receptive field grows with n_layers:
      n_layers=3 → 70×70 (standard pix2pix)
      n_layers=4 → 142×142

    Usage:
        disc = PatchGANDiscriminator()
        logits = disc(torch.cat([input_3ch, target_or_pred_1ch], dim=1))
        loss = F.binary_cross_entropy_with_logits(logits, labels)
    """

    def __init__(
        self,
        in_channels: int = 4,   # 3 input modalities + 1 target/prediction
        base_channels: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            # First block: no BatchNorm (as in original pix2pix)
            _ConvBlock(in_channels, base_channels, stride=2, norm=False)
        ]
        ch = base_channels
        for i in range(1, n_layers):
            next_ch = min(ch * 2, 512)
            layers.append(_ConvBlock(ch, next_ch, stride=2, norm=True))
            ch = next_ch

        # Final two layers: stride-1 conv + output (no activation — raw logits)
        next_ch = min(ch * 2, 512)
        layers.append(_ConvBlock(ch, next_ch, stride=1, norm=True))
        layers.append(nn.Conv2d(next_ch, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) — concatenated input modalities + target.
        Returns:
            logits: (B, 1, H', W') patch-level real/fake scores.
        """
        return self.model(x)


def discriminator_loss(
    disc: PatchGANDiscriminator,
    real_input: torch.Tensor,
    real_target: torch.Tensor,
    fake_target: torch.Tensor,
) -> torch.Tensor:
    """
    Standard GAN discriminator loss (real vs fake).
    real_input: (B, 3, H, W) — the 3 available modalities.
    real_target: (B, 1, H, W) — ground truth missing modality.
    fake_target: (B, 1, H, W) — synthesized modality (detached from generator graph).
    """
    real_pair = torch.cat([real_input, real_target], dim=1)
    fake_pair = torch.cat([real_input, fake_target.detach()], dim=1)

    real_logits = disc(real_pair)
    fake_logits = disc(fake_pair)

    real_labels = torch.ones_like(real_logits)
    fake_labels = torch.zeros_like(fake_logits)

    loss_real = nn.functional.binary_cross_entropy_with_logits(real_logits, real_labels)
    loss_fake = nn.functional.binary_cross_entropy_with_logits(fake_logits, fake_labels)
    return (loss_real + loss_fake) * 0.5


def generator_adversarial_loss(
    disc: PatchGANDiscriminator,
    real_input: torch.Tensor,
    fake_target: torch.Tensor,
) -> torch.Tensor:
    """Generator adversarial loss: fool the discriminator."""
    fake_pair = torch.cat([real_input, fake_target], dim=1)
    fake_logits = disc(fake_pair)
    # Generator wants discriminator to output "real" (1)
    real_labels = torch.ones_like(fake_logits)
    return nn.functional.binary_cross_entropy_with_logits(fake_logits, real_labels)
