"""
Inference: load 3 modalities, detect missing one, synthesize with trained model.

BUG FIXES vs original:
- Architecture params (base_channels, depth, dropout, use_attention, context_slices)
  are loaded FROM the checkpoint dict instead of being hardcoded.
  Previously, if HPO found depth=5 but inference used depth=4, the model would
  crash or silently produce garbage.
- args.target handling uses explicit None check instead of truthiness hack.

NEW: --mc_samples N runs Monte Carlo Dropout uncertainty estimation.
     Saves both the mean prediction AND a pixel-wise uncertainty (std) volume.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import nibabel as nib

from dataset import MODALITIES, _percentile_clip
from model import build_model


def detect_missing_modality(folder: Path) -> str:
    """Return the single missing modality in folder (BraSyn test format)."""
    files = list(folder.glob("*.nii.gz")) + list(folder.glob("*.nii"))
    found = []
    for m in MODALITIES:
        for f in files:
            stem = f.name.replace(".nii.gz", "").replace(".nii", "")
            if stem.endswith(f"-{m}"):
                found.append(m)
                break
    present = set(found)
    missing = set(MODALITIES) - present
    if len(missing) != 1:
        raise ValueError(
            f"Expected exactly one missing modality in {folder}. "
            f"Found: {present}, missing: {missing}"
        )
    return missing.pop()


def load_volume(path: Path) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj)


def save_volume(arr: np.ndarray, path: Path, ref_nii: nib.Nifti1Image):
    nii = nib.Nifti1Image(arr.astype(np.float32), ref_nii.affine, ref_nii.header)
    nib.save(nii, path)


def _crop_center(sl: np.ndarray, ph: int, pw: int) -> np.ndarray:
    from scipy.ndimage import zoom
    sh, sw = sl.shape
    if sh == ph and sw == pw:
        return sl
    if sh >= ph and sw >= pw:
        y0, x0 = (sh - ph) // 2, (sw - pw) // 2
        return sl[y0 : y0 + ph, x0 : x0 + pw].copy()
    return zoom(sl, (ph / sh, pw / sw), order=1)


def _resize_to(sl: np.ndarray, h: int, w: int) -> np.ndarray:
    from scipy.ndimage import zoom
    if sl.shape[0] == h and sl.shape[1] == w:
        return sl
    return zoom(sl, (h / sl.shape[0], w / sl.shape[1]), order=1)


def _enable_mc_dropout(model: torch.nn.Module):
    """Set all Dropout layers to train mode so they fire during inference."""
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()


def run_inference(
    case_dir: str,
    checkpoint_path: str,
    target_modality: str | None,
    device: str = "cuda",
    slice_axis: int = 2,
    patch_size: tuple = (160, 160),
    mc_samples: int = 1,
) -> None:
    case_dir = Path(case_dir)
    if target_modality is None or target_modality == "":
        target_modality = detect_missing_modality(case_dir)

    available = [m for m in MODALITIES if m != target_modality]
    paths = {}
    for m in MODALITIES:
        for f in list(case_dir.glob("*.nii.gz")) + list(case_dir.glob("*.nii")):
            stem = f.name.replace(".nii.gz", "").replace(".nii", "")
            if stem.endswith(f"-{m}"):
                paths[m] = f
                break

    if len(paths) != 3 or set(paths.keys()) != set(available):
        raise FileNotFoundError(f"Need exactly 3 files for {available} in {case_dir}")

    ref_nii = nib.load(paths[available[0]])
    vols = {m: load_volume(paths[m]) for m in available}
    shape = ref_nii.shape

    # FIX: load architecture config from checkpoint, not hardcoded defaults
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    base_channels = arch.get("base_channels", 32)
    depth = arch.get("depth", 4)
    dropout = arch.get("dropout", 0.1)
    use_attention = arch.get("use_attention", True)
    deep_supervision = arch.get("deep_supervision", False)
    context_slices = arch.get("context_slices", 0)

    model = build_model(
        base_channels=base_channels,
        depth=depth,
        dropout=dropout,
        use_attention=use_attention,
        deep_supervision=deep_supervision,
        context_slices=context_slices,
        device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    ph, pw = patch_size
    n_slices = shape[slice_axis]
    use_mc = mc_samples > 1

    if use_mc:
        # MC Dropout: keep dropout ON during inference
        _enable_mc_dropout(model)
        out_vol_sum = np.zeros(shape, dtype=np.float64)
        out_vol_sq = np.zeros(shape, dtype=np.float64)
        print(f"[MC Dropout] Running {mc_samples} stochastic forward passes...")
    else:
        model.eval()
        out_vol = np.zeros(shape, dtype=np.float32)

    h_out = shape[1] if slice_axis == 0 else shape[0]
    w_out = shape[2] if slice_axis in (0, 1) else shape[1]

    for run in range(mc_samples if use_mc else 1):
        for s in range(n_slices):
            slices = []
            for m in available:
                sl = vols[m][s, :, :] if slice_axis == 0 else \
                     vols[m][:, s, :] if slice_axis == 1 else \
                     vols[m][:, :, s]
                sl = _percentile_clip(sl, 0.5, 99.5)
                sl = _crop_center(sl, ph, pw)
                slices.append(sl)

            x = np.stack(slices, axis=0).astype(np.float32)
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)

            with torch.no_grad():
                model_out = model(x_t)
                if isinstance(model_out, tuple):
                    raw = model_out[0]
                else:
                    raw = model_out
                pred_slice = raw.squeeze(0).squeeze(0).cpu().numpy()

            pred_slice = _resize_to(pred_slice, h_out, w_out)

            if use_mc:
                if slice_axis == 0:
                    out_vol_sum[s] += pred_slice
                    out_vol_sq[s] += pred_slice ** 2
                elif slice_axis == 1:
                    out_vol_sum[:, s, :] += pred_slice
                    out_vol_sq[:, s, :] += pred_slice ** 2
                else:
                    out_vol_sum[:, :, s] += pred_slice
                    out_vol_sq[:, :, s] += pred_slice ** 2
            else:
                if slice_axis == 0:
                    out_vol[s] = pred_slice
                elif slice_axis == 1:
                    out_vol[:, s, :] = pred_slice
                else:
                    out_vol[:, :, s] = pred_slice

    out_name = case_dir.name + f"-{target_modality}.nii.gz"
    out_path = case_dir / out_name

    if use_mc:
        mean_vol = (out_vol_sum / mc_samples).astype(np.float32)
        std_vol = np.sqrt(
            np.maximum(out_vol_sq / mc_samples - (out_vol_sum / mc_samples) ** 2, 0)
        ).astype(np.float32)
        save_volume(mean_vol, out_path, ref_nii)
        unc_path = case_dir / (case_dir.name + f"-{target_modality}_uncertainty.nii.gz")
        save_volume(std_vol, unc_path, ref_nii)
        print(f"Saved synthesis:    {out_path}")
        print(f"Saved uncertainty:  {unc_path}")
        print(f"Mean uncertainty (std): {std_vol.mean():.4f}")
    else:
        save_volume(out_vol, out_path, ref_nii)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=str, help="Case folder with 3 NIfTI modalities")
    parser.add_argument("checkpoint", type=str, help="Path to best_<target>.pt")
    parser.add_argument("--target", type=str, default=None,
                        choices=list(MODALITIES) + [None],
                        help="Missing modality (auto-detect if not set)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mc_samples", type=int, default=1,
                        help="MC Dropout samples (>1 enables uncertainty estimation)")
    args = parser.parse_args()

    # FIX: explicit None check instead of truthiness (empty string is falsy)
    target = args.target if args.target is not None else None
    run_inference(
        args.case_dir,
        args.checkpoint,
        target,
        args.device,
        mc_samples=args.mc_samples,
    )


if __name__ == "__main__":
    main()
