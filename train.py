"""
Training script for BraSyn MRI modality synthesis.

BUG FIXES vs original:
- SSIM loss averaged correctly per-batch (not per-sample).
- Checkpoint now saves architecture config (base_channels, depth, dropout,
  context_slices, use_attention) so inference.py can rebuild the exact model.

NEW FEATURES:
- --gan flag: adversarial training with PatchGAN discriminator.
- --perceptual flag: VGG perceptual loss (torchvision required).
- --deep_supervision flag: auxiliary losses from intermediate decoder levels.
- --context_slices: 2.5D adjacent-slice input.
- --use_attention: Attention U-Net gates (default True).
- --ablation_tag: label checkpoint/log for ablation study variants.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from dataset import MODALITIES, get_dataloaders
from model import build_model
from metrics import compute_batch_metrics
from discriminator import PatchGANDiscriminator, discriminator_loss, generator_adversarial_loss


# ─── Losses ───────────────────────────────────────────────────────────────────

def _ssim_loss(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    from torchmetrics.functional import structural_similarity_index_measure
    return 1.0 - structural_similarity_index_measure(pred, target, data_range=data_range)


class PerceptualLoss(nn.Module):
    """VGG16 feature-level perceptual loss (relu2_2 layer).
    NLP analogy: like matching hidden-state representations, not raw tokens.
    """
    def __init__(self, device: str = "cpu"):
        super().__init__()
        try:
            from torchvision import models
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:9]
            self.vgg = vgg.eval().to(device)
            for p in self.vgg.parameters():
                p.requires_grad = False
            self.available = True
        except ImportError:
            print("[PerceptualLoss] torchvision not found — perceptual loss disabled.")
            self.available = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.available:
            return torch.tensor(0.0, device=pred.device)
        # VGG expects 3-channel input; repeat grayscale
        p3 = pred.repeat(1, 3, 1, 1) if pred.shape[1] == 1 else pred
        t3 = target.repeat(1, 3, 1, 1) if target.shape[1] == 1 else target
        return F.mse_loss(self.vgg(p3), self.vgg(t3))


# ─── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_weights,
    data_range=1.0,
    discriminator=None,
    disc_optimizer=None,
    perceptual_loss_fn=None,
    deep_supervision=False,
    max_batches=None,
):
    import itertools
    model.train()
    if discriminator is not None:
        discriminator.train()

    total_loss = total_l1 = total_ssim = 0.0
    n_batches = 0

    iterator = itertools.islice(loader, max_batches) if max_batches else loader
    for x, y in tqdm(iterator, desc="Train", leave=False, total=max_batches or len(loader)):
        x, y = x.to(device), y.to(device)

        # ── Generator forward ──
        optimizer.zero_grad()
        model_out = model(x)

        if deep_supervision and isinstance(model_out, tuple):
            out, aux_outputs = model_out
        else:
            out = model_out
            aux_outputs = []

        out = out.clamp(0.0, data_range)

        l1 = F.l1_loss(out, y)
        ssim_l = _ssim_loss(out, y, data_range=data_range)
        loss = loss_weights.get("l1", 1.0) * l1 + loss_weights.get("ssim", 0.5) * ssim_l

        # Deep supervision auxiliary losses (weighted by 0.4 each to not dominate)
        for aux in aux_outputs:
            aux = aux.clamp(0.0, data_range)
            loss = loss + 0.4 * F.l1_loss(aux, y)

        # Perceptual loss
        if perceptual_loss_fn is not None and loss_weights.get("perceptual", 0.0) > 0:
            perc_l = perceptual_loss_fn(out, y)
            loss = loss + loss_weights["perceptual"] * perc_l

        # Adversarial generator loss
        if discriminator is not None and loss_weights.get("gan", 0.0) > 0:
            gen_adv_loss = generator_adversarial_loss(discriminator, x, out)
            loss = loss + loss_weights["gan"] * gen_adv_loss

        loss.backward()
        optimizer.step()

        # ── Discriminator update ──
        if discriminator is not None and disc_optimizer is not None:
            disc_optimizer.zero_grad()
            d_loss = discriminator_loss(discriminator, x, y, out.detach())
            d_loss.backward()
            disc_optimizer.step()

        # FIX: accumulate per-batch (was dividing total_ssim by n_samples — wrong)
        total_loss += loss.item()
        total_l1 += l1.item()
        total_ssim += ssim_l.item()
        n_batches += 1

    denom = max(n_batches, 1)
    return {
        "loss": total_loss / denom,
        "l1": total_l1 / denom,
        "ssim_loss": total_ssim / denom,
    }


@torch.no_grad()
def validate(model, loader, device, data_range=1.0, deep_supervision=False):
    model.eval()
    metrics_sum = {"ssim": 0.0, "psnr": 0.0, "mse": 0.0}
    n_batches = 0
    for x, y in tqdm(loader, desc="Val", leave=False):
        x, y = x.to(device), y.to(device)
        model_out = model(x)
        if deep_supervision and isinstance(model_out, tuple):
            out = model_out[0]
        else:
            out = model_out
        out = out.clamp(0.0, data_range)
        m = compute_batch_metrics(out, y, data_range=data_range)
        for k in metrics_sum:
            metrics_sum[k] += m[k]
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in metrics_sum.items()}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BraSyn MRI synthesis training")
    parser.add_argument("--data_root", type=str, default="Dataset - Training")
    parser.add_argument("--target", type=str, default="t2f", choices=MODALITIES)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss_ssim_weight", type=float, default=0.5)
    parser.add_argument("--loss_gan_weight", type=float, default=0.1)
    parser.add_argument("--loss_perceptual_weight", type=float, default=0.0)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[160, 160])
    parser.add_argument("--slice_axis", type=int, default=2)
    parser.add_argument("--train_split", type=float, default=0.85)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "none"])
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    # Architecture flags
    parser.add_argument("--use_attention", action="store_true", default=True,
                        help="Use Attention U-Net gates (default: True)")
    parser.add_argument("--no_attention", dest="use_attention", action="store_false")
    parser.add_argument("--deep_supervision", action="store_true", default=False)
    parser.add_argument("--context_slices", type=int, default=0,
                        help="2.5D: number of adjacent slices to stack (0 = 2D)")
    # Training mode flags
    parser.add_argument("--gan", action="store_true", default=False,
                        help="Enable adversarial training with PatchGAN")
    parser.add_argument("--perceptual", action="store_true", default=False,
                        help="Enable VGG perceptual loss")
    parser.add_argument("--ablation_tag", type=str, default="",
                        help="Tag for ablation study — appended to checkpoint name")
    parser.add_argument("--max_batches_per_epoch", type=int, default=None,
                        help="Cap batches per epoch (e.g. 500 = ~30min/epoch on RTX 4060)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader worker processes (0 = main process only)")
    parser.add_argument("--val_slices_per_case", type=int, default=3,
                        help="Validation slices per patient (lower = faster val loop)")
    args = parser.parse_args()

    # ── Load YAML config (overrides defaults) ──
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        train_cfg = cfg.get("training", {})
        # args.data_root = data_cfg.get("data_root", args.data_root)
        args.patch_size = data_cfg.get("patch_size", args.patch_size)
        args.slice_axis = data_cfg.get("slice_axis", args.slice_axis)
        args.train_split = data_cfg.get("train_split", args.train_split)
        args.context_slices = data_cfg.get("context_slices", args.context_slices)
        args.base_channels = model_cfg.get("base_channels", args.base_channels)
        args.depth = model_cfg.get("depth", args.depth)
        args.dropout = model_cfg.get("dropout", args.dropout)
        # Parse max_batches_per_epoch from train config if not given via CLI
        if args.max_batches_per_epoch is None:
            args.max_batches_per_epoch = train_cfg.get("max_batches_per_epoch", args.max_batches_per_epoch)
        args.use_attention = model_cfg.get("use_attention", args.use_attention)
        args.deep_supervision = model_cfg.get("deep_supervision", args.deep_supervision)
        # args.epochs = train_cfg.get("epochs", args.epochs)
        args.batch_size = train_cfg.get("batch_size", args.batch_size)
        args.lr = train_cfg.get("learning_rate", args.lr)
        args.weight_decay = train_cfg.get("weight_decay", args.weight_decay)
        args.scheduler = train_cfg.get("lr_scheduler", args.scheduler)
        loss_w = train_cfg.get("loss_weights", {})
        args.loss_ssim_weight = loss_w.get("ssim", args.loss_ssim_weight)
        args.loss_gan_weight = loss_w.get("gan", args.loss_gan_weight)
        args.loss_perceptual_weight = loss_w.get("perceptual", args.loss_perceptual_weight)
        args.gan = train_cfg.get("gan", args.gan)
        args.perceptual = train_cfg.get("perceptual", args.perceptual)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tag = f"_{args.ablation_tag}" if args.ablation_tag else ""

    # ── Data ──
    train_loaders, val_loaders, _, _ = get_dataloaders(
        data_root=args.data_root,
        slice_axis=args.slice_axis,
        patch_size=tuple(args.patch_size),
        batch_size=args.batch_size,
        train_split=args.train_split,
        pin_memory=(device == "cuda"),
        context_slices=args.context_slices,
        num_workers=args.num_workers,
        val_slices_per_case=args.val_slices_per_case,
    )
    train_loader = train_loaders[args.target]
    val_loader = val_loaders[args.target]

    # ── Model ──
    model = build_model(
        base_channels=args.base_channels,
        depth=args.depth,
        dropout=args.dropout,
        use_attention=args.use_attention,
        deep_supervision=args.deep_supervision,
        context_slices=args.context_slices,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── Discriminator (optional) ──
    discriminator = None
    disc_optimizer = None
    if args.gan:
        in_ch_disc = (3 * (2 * args.context_slices + 1)) + 1 if args.context_slices > 0 else 4
        discriminator = PatchGANDiscriminator(in_channels=in_ch_disc).to(device)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        print(f"[GAN] PatchGAN discriminator enabled (in_ch={in_ch_disc}).")

    # ── Perceptual loss (optional) ──
    perceptual_loss_fn = None
    if args.perceptual:
        perceptual_loss_fn = PerceptualLoss(device=device)

    # ── Scheduler ──
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5)
    else:
        scheduler = None

    loss_weights = {
        "l1": 1.0,
        "ssim": args.loss_ssim_weight,
        "gan": args.loss_gan_weight if args.gan else 0.0,
        "perceptual": args.loss_perceptual_weight if args.perceptual else 0.0,
    }

    # ── Logging ──
    best_ssim = -1.0
    log_path = save_dir / f"train_log{tag}_{args.target}.csv"
    with open(log_path, "w") as f:
        f.write("epoch,loss,val_ssim,val_psnr,val_mse\n")

    print(f"Training target={args.target} | attention={args.use_attention} | "
          f"deep_sup={args.deep_supervision} | GAN={args.gan} | "
          f"perceptual={args.perceptual} | context={args.context_slices}D | "
          f"device={device}")

    # ── Training loop ──
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, loss_weights,
            data_range=1.0,
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
            perceptual_loss_fn=perceptual_loss_fn,
            deep_supervision=args.deep_supervision,
            max_batches=args.max_batches_per_epoch,
        )
        val_metrics = validate(
            model, val_loader, device, data_range=1.0,
            deep_supervision=args.deep_supervision,
        )

        if args.scheduler == "cosine":
            scheduler.step()
        elif args.scheduler == "plateau":
            scheduler.step(val_metrics["ssim"])

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_metrics['loss']:.6f},"
                    f"{val_metrics['ssim']:.6f},{val_metrics['psnr']:.6f},"
                    f"{val_metrics['mse']:.6f}\n")

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"val SSIM={val_metrics['ssim']:.4f} "
            f"PSNR={val_metrics['psnr']:.2f} "
            f"MSE={val_metrics['mse']:.6f}"
        )

        if val_metrics["ssim"] > best_ssim:
            best_ssim = val_metrics["ssim"]
            # FIX: save architecture config alongside weights so inference
            # can reconstruct the exact model without hardcoding defaults.
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_ssim": val_metrics["ssim"],
                    "val_psnr": val_metrics["psnr"],
                    "target": args.target,
                    "arch": {
                        "base_channels": args.base_channels,
                        "depth": args.depth,
                        "dropout": args.dropout,
                        "use_attention": args.use_attention,
                        "deep_supervision": args.deep_supervision,
                        "context_slices": args.context_slices,
                    },
                },
                save_dir / f"best_{args.target}{tag}.pt",
            )

    print(f"Best val SSIM: {best_ssim:.4f}. Saved to {save_dir / f'best_{args.target}{tag}.pt'}")


if __name__ == "__main__":
    main()
