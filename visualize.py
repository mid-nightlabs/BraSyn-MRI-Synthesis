"""
Visualize BraSyn synthesis: input modalities, ground truth, and model prediction.
Saves a figure for CV/portfolio (e.g. outputs/synthesis_comparison.png).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import nibabel as nib

from dataset import MODALITIES, find_cases, _percentile_clip
from model import build_model


def load_slice_for_case(case_dir: Path, paths: list, target_modality: str, slice_idx: int, slice_axis: int = 2, patch_size=(160, 160)):
    """Load and preprocess one slice for a case (same logic as dataset)."""
    modalities = MODALITIES
    target_idx = modalities.index(target_modality)
    vols = [np.asarray(nib.load(p).dataobj) for p in paths]
    slices = []
    for v in vols:
        if slice_axis == 0:
            sl = v[slice_idx, :, :]
        elif slice_axis == 1:
            sl = v[:, slice_idx, :]
        else:
            sl = v[:, :, slice_idx]
        slices.append(sl)
    ref = slices[target_idx]
    for i in range(len(slices)):
        slices[i] = _percentile_clip(slices[i], 0.5, 99.5, ref=ref)
    ph, pw = patch_size
    from scipy.ndimage import zoom
    for i in range(len(slices)):
        sh, sw = slices[i].shape
        if sh != ph or sw != pw:
            if sh >= ph and sw >= pw:
                y0, x0 = (sh - ph) // 2, (sw - pw) // 2
                slices[i] = slices[i][y0:y0+ph, x0:x0+pw]
            else:
                slices[i] = zoom(slices[i], (ph/sh, pw/sw), order=1)
    input_ch = [s for i, s in enumerate(slices) if i != target_idx]
    target = slices[target_idx]
    x = np.stack(input_ch, axis=0).astype(np.float32)
    y = target[np.newaxis, ...].astype(np.float32)
    return x, y, [slices[i] for i in range(4)]  # all 4 for display


def main():
    parser = argparse.ArgumentParser(description="Visualize BraSyn synthesis (input | GT | predicted)")
    parser.add_argument("--data_root", type=str, required=True, help="Path to ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_<target>.pt")
    parser.add_argument("--target", type=str, default="t2f", choices=MODALITIES)
    parser.add_argument("--case_index", type=int, default=0, help="Which case (0-based)")
    parser.add_argument("--slice_index", type=int, default=None, help="Slice index (default: middle)")
    parser.add_argument("--output", type=str, default="outputs/synthesis_comparison.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cases = find_cases(args.data_root, MODALITIES)
    if not cases:
        raise FileNotFoundError(f"No cases in {args.data_root}")
    case_dir, paths = cases[args.case_index % len(cases)]
    case_dir = Path(case_dir)
    n_slices = np.asarray(nib.load(paths[0]).dataobj).shape[2]
    slice_idx = args.slice_index if args.slice_index is not None else n_slices // 2

    x, y, all_four = load_slice_for_case(case_dir, paths, args.target, slice_idx)
    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    model = build_model(
        base_channels=arch.get("base_channels", 32),
        depth=arch.get("depth", 4),
        dropout=0.0,
        use_attention=arch.get("use_attention", True),
        deep_supervision=arch.get("deep_supervision", False),
        context_slices=arch.get("context_slices", 0),
        device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)
        out = model(x_t)
        if isinstance(out, tuple):
            out = out[0]
        pred = out.squeeze(0).squeeze(0).cpu().numpy()

    # Labels for modalities
    mod_labels = {"t1n": "T1", "t1c": "T1ce", "t2w": "T2", "t2f": "FLAIR"}
    target_idx = MODALITIES.index(args.target)
    input_mods = [m for m in MODALITIES if m != args.target]

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    for i, m in enumerate(input_mods):
        axes[i].imshow(all_four[MODALITIES.index(m)], cmap="gray")
        axes[i].set_title(f"Input: {mod_labels.get(m, m)}")
        axes[i].axis("off")
        
    gt = all_four[target_idx]
    axes[3].imshow(gt, cmap="gray")
    axes[3].set_title(f"Ground truth: {mod_labels.get(args.target, args.target)}")
    axes[3].axis("off")
    
    axes[4].imshow(pred, cmap="gray")
    axes[4].set_title(f"Synthesized: {mod_labels.get(args.target, args.target)}")
    axes[4].axis("off")
    
    # Error map (Absolute difference)
    err = np.abs(gt - pred)
    # Exclude background from max calculation if necessary, or use fixed max for fair comparison
    err_plot = axes[5].imshow(err, cmap="hot", vmin=0, vmax=1.0)
    axes[5].set_title("Error Map (|GT - Pred|)")
    axes[5].axis("off")
    fig.colorbar(err_plot, ax=axes[5], fraction=0.046, pad=0.04)
    fig.suptitle(f"BraSyn MRI synthesis — Case {case_dir.name} (slice {slice_idx})", fontsize=12)
    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
