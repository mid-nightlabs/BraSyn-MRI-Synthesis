"""
Plot training curves from train_log.csv (loss, val SSIM, PSNR, MSE).
Use for reports and CV/portfolio.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot BraSyn training curves")
    parser.add_argument("--log", type=str, default="checkpoints/train_log.csv", help="Path to train_log.csv")
    parser.add_argument("--output", type=str, default="outputs/training_curves.png")
    args = parser.parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log not found: {log_path}. Run train.py first to generate it.")
        return
    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    epochs = df["epoch"]
    axes[0, 0].plot(epochs, df["loss"], color="C0")
    axes[0, 0].set_title("Training loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(epochs, df["val_ssim"], color="C1")
    axes[0, 1].set_title("Validation SSIM (higher is better)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(epochs, df["val_psnr"], color="C2")
    axes[1, 0].set_title("Validation PSNR (higher is better)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(epochs, df["val_mse"], color="C3")
    axes[1, 1].set_title("Validation MSE (lower is better)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle("BraSyn training metrics", fontsize=12)
    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
