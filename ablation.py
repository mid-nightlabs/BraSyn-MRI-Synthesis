"""
Ablation study runner for BraSyn MRI synthesis.

Trains all 4 ablation variants sequentially and produces a results table.
Variants:
  1. Baseline  — plain U-Net, L1 only
  2. + SSIM    — plain U-Net, L1 + SSIM
  3. + Attention— Attention U-Net, L1 + SSIM
  4. Full      — Attention U-Net, L1 + SSIM + GAN + deep supervision

Results are written to ablation_results.csv and a markdown-ready summary.

Usage:
    python ablation.py --data_root data/BraSyn --target t2f --epochs 50
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


VARIANTS = [
    {
        "tag": "ssim",
        "label": "Plain U-Net + L1 + SSIM",
        "config": "ablation_configs/ablation_ssim.yaml",
    },
    {
        "tag": "attention",
        "label": "Attention U-Net + L1 + SSIM",
        "config": "ablation_configs/ablation_attention.yaml",
    },
    {
        "tag": "full",
        "label": "Attention U-Net + L1 + SSIM + GAN (Ours)",
        "config": "ablation_configs/ablation_full.yaml",
    },
]


def run_variant(
    tag: str,
    config: str,
    target: str,
    epochs: int,
    data_root: str,
    save_dir: str,
    max_batches_per_epoch: int = None,
) -> Path:
    """Train one variant. Returns path to best checkpoint."""
    checkpoint = Path(save_dir) / f"best_{target}_{tag}.pt"
    if checkpoint.exists():
        print(f"[ablation] Checkpoint exists for '{tag}', skipping training.")
        return checkpoint

    cmd = [
        sys.executable, "train.py",
        "--config", config,
        "--target", target,
        "--epochs", str(epochs),
        "--data_root", data_root,
        "--save_dir", save_dir,
        "--ablation_tag", tag,
    ]
    
    if max_batches_per_epoch is not None:
        cmd += ["--max_batches_per_epoch", str(max_batches_per_epoch)]
    
    # GAN flag passed explicitly for 'full' variant
    if tag == "full":
        cmd += ["--gan", "--deep_supervision"]

    print(f"\n{'='*60}")
    print(f"[ablation] Training variant: {tag}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)
    # rename checkpoint from best_{target}_{tag}.pt (train.py naming)
    return checkpoint


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_root: str,
    target: str,
    device: str,
    n_val_slices: int = 5,
) -> dict:
    """Run evaluate.py and parse results."""
    from evaluate import evaluate
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        evaluate(
            data_root=data_root,
            checkpoint_path=str(checkpoint_path),
            target=target,
            output_dir=tmp,
            n_val_slices=n_val_slices,
            device=device,
        )
        agg_path = Path(tmp) / f"aggregate_{target}.txt"
        metrics = {}
        if agg_path.exists():
            with open(agg_path) as f:
                for line in f:
                    for k in ["SSIM", "PSNR", "MSE"]:
                        if line.startswith(k):
                            parts = line.split()
                            metrics[k.lower()] = float(parts[1])
    return metrics


def main():
    parser = argparse.ArgumentParser(description="BraSyn ablation study")
    parser.add_argument("--data_root", type=str, default="Dataset - Training")
    parser.add_argument("--target", type=str, default="t2f")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Epochs per variant (50 for ablation, 100 for full)")
    parser.add_argument("--save_dir", type=str, default="checkpoints/ablation")
    parser.add_argument("--output", type=str, default="ablation_results.csv")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n_val_slices", type=int, default=5)
    parser.add_argument("--max_batches_per_epoch", type=int, default=None)
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    rows = []

    for v in VARIANTS:
        print(f"\n[ablation] Variant: {v['label']}")
        ckpt = run_variant(
            tag=v["tag"],
            config=v["config"],
            target=args.target,
            epochs=args.epochs,
            data_root=args.data_root,
            save_dir=args.save_dir,
            max_batches_per_epoch=args.max_batches_per_epoch,
        )
        if ckpt.exists():
            metrics = evaluate_checkpoint(ckpt, args.data_root, args.target,
                                          device, args.n_val_slices)
        else:
            print(f"[ablation] WARNING: checkpoint not found for {v['tag']}")
            metrics = {"ssim": 0.0, "psnr": 0.0, "mse": 0.0}
        rows.append({**v, **metrics})

    # Write CSV
    out_path = Path(args.output)
    fieldnames = ["tag", "label", "ssim", "psnr", "mse"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_path}")

    # Print markdown table
    print("\n## Ablation Results\n")
    print(f"| Model Variant | SSIM ↑ | PSNR ↑ | MSE ↓ |")
    print(f"|{'─'*40}|{'─'*9}|{'─'*9}|{'─'*9}|")
    for r in rows:
        marker = " (**ours**)" if r["tag"] == "full" else ""
        print(f"| {r['label']}{marker} | {r.get('ssim', 0):.4f} | "
              f"{r.get('psnr', 0):.2f} | {r.get('mse', 0):.6f} |")


if __name__ == "__main__":
    main()
