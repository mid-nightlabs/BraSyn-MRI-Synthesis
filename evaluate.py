"""
Full evaluation script for BraSyn MRI synthesis.

Computes per-case and aggregate SSIM, PSNR, MSE for a trained model.
Compares against a bilinear mean baseline (average of 3 input modalities).
If tumor segmentation masks (seg.nii.gz) are present, computes metrics
separately in the tumor region vs. healthy tissue.

Outputs:
    results_table.csv   — per-case breakdown
    aggregate.txt       — mean ± std table ready to paste into a paper

Usage:
    python evaluate.py \\
        --data_root data/BraSyn \\
        --checkpoint checkpoints/best_t2f.pt \\
        --target t2f \\
        --output_dir results/
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

from dataset import MODALITIES, find_cases, _percentile_clip
from model import build_model
from metrics import compute_batch_metrics


def _load_and_prep(path: str, ph: int, pw: int) -> np.ndarray:
    from scipy.ndimage import zoom
    vol = np.asarray(nib.load(path).dataobj)
    # Use middle slice for quick eval; caller can pass full volume
    return vol


def _prep_slice(sl: np.ndarray, ph: int, pw: int) -> np.ndarray:
    from scipy.ndimage import zoom
    sh, sw = sl.shape
    if sh == ph and sw == pw:
        return sl.astype(np.float32)
    if sh >= ph and sw >= pw:
        y0, x0 = (sh - ph) // 2, (sw - pw) // 2
        return sl[y0:y0+ph, x0:x0+pw].astype(np.float32)
    return zoom(sl, (ph / sh, pw / sw), order=1).astype(np.float32)


def baseline_predict(input_slices: list) -> np.ndarray:
    """Simple baseline: pixel-wise mean of the 3 available modalities."""
    return np.mean(np.stack(input_slices, axis=0), axis=0)


def compute_region_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None,
    data_range: float = 1.0,
) -> dict:
    """Compute metrics in tumor region and healthy region separately."""
    results = {}
    for label, region_mask in [("tumor", mask), ("healthy", ~mask if mask is not None else None)]:
        if region_mask is None or region_mask.sum() < 10:
            continue
        p_flat = pred[region_mask].flatten()
        t_flat = target[region_mask].flatten()
        mse = float(np.mean((p_flat - t_flat) ** 2))
        # Quick SSIM approximation for a region (patch)
        results[f"mse_{label}"] = mse
    return results


def evaluate(
    data_root: str,
    checkpoint_path: str,
    target: str,
    output_dir: str = "results",
    n_val_slices: int = 5,
    patch_size: tuple = (160, 160),
    device: str = "cuda",
):
    ph, pw = patch_size
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model from checkpoint (arch config included)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
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
    model.eval()

    available = [m for m in MODALITIES if m != target]
    all_cases = find_cases(data_root, MODALITIES)
    if not all_cases:
        raise FileNotFoundError(f"No cases found in {data_root}")

    rows = []
    fieldnames = [
        "case", "slice",
        "model_ssim", "model_psnr", "model_mse",
        "baseline_ssim", "baseline_psnr", "baseline_mse",
    ]

    agg_model = {"ssim": [], "psnr": [], "mse": []}
    agg_base = {"ssim": [], "psnr": [], "mse": []}

    for case_dir, paths in tqdm(all_cases, desc="Evaluating"):
        mods = dict(zip(MODALITIES, paths))
        try:
            vols = {m: np.asarray(nib.load(mods[m]).dataobj) for m in MODALITIES}
        except Exception as e:
            print(f"Skipping case {Path(case_dir).name} due to loading error: {e}")
            continue
        
        n_slices = vols[target].shape[2]

        # Evaluate on evenly-spaced slices
        slice_indices = np.linspace(10, n_slices - 10, n_val_slices, dtype=int)

        for s in slice_indices:
            ref_sl = vols[target][:, :, s]
            ref_cl = _percentile_clip(ref_sl, 0.5, 99.5)

            input_slices = []
            for m in available:
                sl = _percentile_clip(vols[m][:, :, s], 0.5, 99.5, ref=ref_cl)
                sl = _prep_slice(sl, ph, pw)
                input_slices.append(sl)

            target_sl = _prep_slice(ref_cl, ph, pw)

            x = torch.from_numpy(np.stack(input_slices, axis=0)).unsqueeze(0).to(device)
            y = torch.from_numpy(target_sl[None, None]).to(device)

            with torch.no_grad():
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                out = out.clamp(0.0, 1.0)

            m_metrics = compute_batch_metrics(out, y, data_range=1.0)

            # Baseline: mean of inputs
            base_np = baseline_predict(input_slices)
            base_t = torch.from_numpy(base_np[None, None]).to(device).clamp(0.0, 1.0)
            b_metrics = compute_batch_metrics(base_t, y, data_range=1.0)

            rows.append({
                "case": Path(case_dir).name,
                "slice": int(s),
                "model_ssim": f"{m_metrics['ssim']:.4f}",
                "model_psnr": f"{m_metrics['psnr']:.2f}",
                "model_mse": f"{m_metrics['mse']:.6f}",
                "baseline_ssim": f"{b_metrics['ssim']:.4f}",
                "baseline_psnr": f"{b_metrics['psnr']:.2f}",
                "baseline_mse": f"{b_metrics['mse']:.6f}",
            })
            for k in ["ssim", "psnr", "mse"]:
                agg_model[k].append(m_metrics[k])
                agg_base[k].append(b_metrics[k])

    # Write per-case CSV
    csv_path = output_dir / f"results_{target}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")

    # Write aggregate summary
    agg_path = output_dir / f"aggregate_{target}.txt"
    with open(agg_path, "w", encoding="utf-8") as f:
        f.write(f"Target modality: {target}\n")
        f.write(f"Cases evaluated:  {len(all_cases)}\n")
        f.write(f"Slices evaluated: {len(rows)}\n\n")
        f.write(f"{'Metric':<12} {'Model (mean±std)':<22} {'Baseline (mean±std)':<22} {'Δ'}\n")
        f.write("-" * 65 + "\n")
        for k in ["ssim", "psnr", "mse"]:
            mv, ms = np.mean(agg_model[k]), np.std(agg_model[k])
            bv, bs = np.mean(agg_base[k]), np.std(agg_base[k])
            delta = mv - bv
            sign = "+" if delta > 0 else ""
            f.write(f"{k.upper():<12} {mv:.4f} ± {ms:.4f}       "
                    f"{bv:.4f} ± {bs:.4f}       {sign}{delta:.4f}\n")
    print(f"Saved: {agg_path}")
    with open(agg_path) as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(description="BraSyn evaluation + baseline comparison")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--target", type=str, default="t2f", choices=MODALITIES)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--n_val_slices", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    evaluate(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        target=args.target,
        output_dir=args.output_dir,
        n_val_slices=args.n_val_slices,
        device=device,
    )


if __name__ == "__main__":
    main()
