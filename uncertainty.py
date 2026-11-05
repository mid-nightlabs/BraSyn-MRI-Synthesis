"""
Monte Carlo Dropout uncertainty estimation for BraSyn MRI synthesis.

MC Dropout (Gal & Ghahramani, 2016): run N stochastic forward passes with
dropout ENABLED at test time. The spread (std) of predictions across runs
is a proxy for the model's epistemic uncertainty — where it "isn't sure".

Clinical relevance: high uncertainty around tumor edges tells a radiologist
*exactly* where to be skeptical about the synthesized scan.

NLP analogy: like running a language model N times with different dropout masks
and using the variance of token probabilities as a confidence score.

CLI usage:
    python uncertainty.py \\
        --case_dir path/to/case \\
        --checkpoint checkpoints/best_t2f.pt \\
        --n_samples 20 \\
        --output outputs/uncertainty_map.png
"""

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import nibabel as nib

from dataset import MODALITIES, _percentile_clip
from model import build_model


# ─── Core MC Dropout ─────────────────────────────────────────────────────────

def _enable_dropout(model: torch.nn.Module):
    """Activate dropout layers for stochastic inference."""
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()


def mc_dropout_predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run N stochastic forward passes and return (mean, std) tensors.

    Args:
        model:     Trained UNet2D (dropout layers will be temporarily activated).
        x:         Input tensor (1, C, H, W).
        n_samples: Number of MC Dropout forward passes.
    Returns:
        mean: (1, 1, H, W) — mean prediction across passes.
        std:  (1, 1, H, W) — pixel-wise standard deviation (uncertainty map).
    """
    model.eval()
    _enable_dropout(model)

    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]  # main head if deep supervision
            preds.append(out.clamp(0.0, 1.0))

    stacked = torch.stack(preds, dim=0)   # (N, 1, 1, H, W)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    return mean, std


def confidence_badge(mean_ssim: float, mean_uncertainty: float) -> str:
    """
    Traffic-light confidence summary.
    High uncertainty or low SSIM → warn the user.
    """
    if mean_uncertainty < 0.02 and mean_ssim > 0.80:
        return "🟢 HIGH confidence"
    elif mean_uncertainty < 0.05 and mean_ssim > 0.65:
        return "🟡 MEDIUM confidence"
    else:
        return "🔴 LOW confidence — inspect carefully"


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_uncertainty(
    input_slices: list,          # list of 3 np arrays [H,W]
    ground_truth: np.ndarray,    # [H,W]
    mean_pred: np.ndarray,       # [H,W]
    uncertainty: np.ndarray,     # [H,W]
    input_mods: list,            # e.g. ["t1n","t1c","t2w"]
    target_mod: str,             # e.g. "t2f"
    save_path: str | None = None,
    ssim_value: float | None = None,
) -> plt.Figure:
    """
    6-panel uncertainty figure (NLP-friendly captions included):
    [ T1 | T1ce | T2 | GT FLAIR | Synthesized | Uncertainty ]
    """
    mod_labels = {"t1n": "T1", "t1c": "T1ce", "t2w": "T2", "t2f": "FLAIR"}

    fig = plt.figure(figsize=(18, 4))
    gs = gridspec.GridSpec(1, 6, wspace=0.05)

    def _show(ax, img, title, cmap="gray", vmin=None, vmax=None, colorbar=False):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9, pad=4)
        ax.axis("off")
        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return im

    # Inputs
    for i, (sl, m) in enumerate(zip(input_slices, input_mods)):
        ax = fig.add_subplot(gs[i])
        _show(ax, sl, f"Input: {mod_labels.get(m, m)}\n(available scan)")

    # Ground truth
    ax_gt = fig.add_subplot(gs[3])
    _show(ax_gt, ground_truth, f"Ground truth\n{mod_labels.get(target_mod, target_mod)}")

    # Synthesized (mean over MC runs)
    ax_syn = fig.add_subplot(gs[4])
    ssim_str = f"\nSSIM={ssim_value:.3f}" if ssim_value is not None else ""
    _show(ax_syn, mean_pred, f"Synthesized{ssim_str}\n(model output)")

    # Uncertainty heatmap
    ax_unc = fig.add_subplot(gs[5])
    unc_norm = uncertainty / (uncertainty.max() + 1e-8)
    _show(ax_unc, unc_norm, f"Uncertainty\n(MC Dropout σ)\n← low | high →",
          cmap="hot", vmin=0, vmax=1, colorbar=True)

    badge = confidence_badge(ssim_value or 0.0, uncertainty.mean())

    fig.suptitle(
        f"BraSyn MRI Synthesis — MC Dropout Uncertainty   {badge}\n"
        f"NLP analogy: uncertainty = variance of model's 'token' predictions "
        f"across {20} stochastic forward passes",
        fontsize=9, y=1.02
    )
    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved uncertainty figure: {out}")

    return fig


# ─── CLI  ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MC Dropout uncertainty estimation")
    parser.add_argument("--case_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--target", type=str, default=None, choices=MODALITIES + [None])
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--slice_index", type=int, default=None)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[160, 160])
    parser.add_argument("--output", type=str, default="outputs/uncertainty_map.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    device = args.device if torch.cuda.is_available() else "cpu"

    # Load checkpoint and reconstruct architecture
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    target_mod = args.target or ckpt.get("target", "t2f")
    available = [m for m in MODALITIES if m != target_mod]

    model = build_model(
        base_channels=arch.get("base_channels", 32),
        depth=arch.get("depth", 4),
        dropout=arch.get("dropout", 0.1),
        use_attention=arch.get("use_attention", True),
        deep_supervision=arch.get("deep_supervision", False),
        context_slices=arch.get("context_slices", 0),
        device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    # Load volumes
    paths = {}
    for m in MODALITIES:
        for f in list(case_dir.glob("*.nii.gz")) + list(case_dir.glob("*.nii")):
            stem = f.name.replace(".nii.gz", "").replace(".nii", "")
            if stem.endswith(f"-{m}"):
                paths[m] = f; break

    ph, pw = args.patch_size
    from scipy.ndimage import zoom as ndz

    def _prep(vol, s):
        sl = vol[:, :, s]
        sl = _percentile_clip(sl, 0.5, 99.5)
        sh, sw = sl.shape
        if sh != ph or sw != pw:
            sl = ndz(sl, (ph / sh, pw / sw), order=1)
        return sl.astype(np.float32)

    ref_nii = nib.load(paths[available[0]])
    n_slices = ref_nii.shape[2]
    s = args.slice_index if args.slice_index is not None else n_slices // 2

    vols = {m: np.asarray(nib.load(paths[m]).dataobj) for m in available}
    input_slices_np = [_prep(vols[m], s) for m in available]

    if target_mod in paths:
        gt_vol = np.asarray(nib.load(paths[target_mod]).dataobj)
        gt_slice = _prep(gt_vol, s)
    else:
        gt_slice = np.zeros((ph, pw), dtype=np.float32)

    x = torch.from_numpy(np.stack(input_slices_np, axis=0)).unsqueeze(0).to(device)
    mean_t, std_t = mc_dropout_predict(model, x, n_samples=args.n_samples)
    mean_np = mean_t.squeeze().cpu().numpy()
    std_np = std_t.squeeze().cpu().numpy()

    # Quick SSIM for badge
    try:
        from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
        ssim_val = ssim_fn(mean_t.cpu(), torch.from_numpy(gt_slice)[None, None]).item()
    except Exception:
        ssim_val = None

    plot_uncertainty(
        input_slices=input_slices_np,
        ground_truth=gt_slice,
        mean_pred=mean_np,
        uncertainty=std_np,
        input_mods=available,
        target_mod=target_mod,
        save_path=args.output,
        ssim_value=ssim_val,
    )


if __name__ == "__main__":
    main()
