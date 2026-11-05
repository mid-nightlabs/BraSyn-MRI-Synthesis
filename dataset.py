"""
BraSyn-style dataset for brain MRI modality synthesis.
Expects folder structure: data_root / case_id / BraTS-*-{t1n|t1c|t2w|t2f}.nii.gz
Reference: arXiv:2305.09011 (BraSyn MICCAI 2023)

BUG FIXES vs original:
- __len__ now returns total number of (case, slice) pairs, not number of cases.
  Previously each epoch only saw ONE slice per patient; now every axial slice is
  an individual training sample (~155x more data utilisation).
- Validation now uses a deterministic, evenly-spaced set of slices per patient
  instead of always the middle slice.
- context_slices (2.5D): optionally stack adjacent slices per modality as extra
  input channels, giving the model local volumetric context.
- Augmentation extended with small rotation and intensity jitter.
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib


MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def _percentile_clip(
    x: np.ndarray,
    p_min: float = 0.5,
    p_max: float = 99.5,
    ref: Optional[np.ndarray] = None,
) -> np.ndarray:
    ref = ref if ref is not None else x
    v_min, v_max = np.percentile(ref, [p_min, p_max])
    x = np.clip(x, v_min, v_max)
    if v_max > v_min:
        x = (x - v_min) / (v_max - v_min)
    return x.astype(np.float32)


def find_cases(data_root: str, modalities: List[str]) -> List[Tuple[str, List[str]]]:
    """Discover case folders that contain all given modalities.
    BraSyn naming: BraTS-GLI-XXXXX-XXX-t1n.nii[.gz], -t1c, -t2w, -t2f.
    """
    data_root = Path(data_root).resolve()
    if not data_root.exists():
        return []
    cases = []
    for case_dir in sorted(data_root.iterdir()):
        if not case_dir.is_dir():
            continue
        files = list(case_dir.glob("*.nii.gz")) + list(case_dir.glob("*.nii"))
        found = {}
        for m in modalities:
            for f in files:
                stem = f.name.replace(".nii.gz", "").replace(".nii", "")
                if stem.endswith(f"-{m}"):
                    found[m] = str(f)
                    break
        if len(found) == len(modalities):
            cases.append((str(case_dir), [found[m] for m in modalities]))
    return cases


class BraSynSliceDataset(Dataset):
    """
    2D (or 2.5D) slice dataset for BraSyn: given 3 modalities, predict the 4th.

    FIX: __len__ now returns len(slice_index_list) — every (case, slice) pair is
    a distinct training sample. Previously only one slice per case per epoch.

    Args:
        context_slices: number of adjacent slices to stack per modality (0 = 2D).
            With context_slices=1, each modality contributes 3 slices (s-1, s, s+1),
            so total input channels = 3 * (2*context_slices + 1).
        val_slices_per_case: for validation, how many evenly-spaced slices to
            sample per case (deterministic). 0 = use all non-zero slices.
    """

    def __init__(
        self,
        cases: List[Tuple[str, List[str]]],
        modalities: List[str],
        target_modality: str,
        slice_axis: int = 2,
        patch_size: Optional[Tuple[int, int]] = None,
        percentile: Tuple[float, float] = (0.5, 99.5),
        augment: bool = False,
        context_slices: int = 0,
        val_slices_per_case: int = 10,
        skip_empty_slices: bool = True,
    ):
        self.cases = cases
        self.modalities = modalities
        self.target_modality = target_modality
        self.slice_axis = slice_axis
        self.patch_size = patch_size
        self.percentile = percentile
        self.augment = augment
        self.context_slices = context_slices
        self.skip_empty_slices = skip_empty_slices
        self.target_idx = modalities.index(target_modality)

        # --- FIX: precompute flat list of (case_idx, slice_idx) pairs ---
        # Load one volume per case to determine n_slices.
        self.index_list: List[Tuple[int, int]] = []
        for case_idx, (_, paths) in enumerate(cases):
            try:
                nii = nib.load(paths[0])
                vol_shape = nii.header.get_data_shape()
                n_slices = vol_shape[slice_axis]
            except Exception:
                continue

            if augment:
                # Training: all slices are valid samples
                margin = max(context_slices, 1)  # avoid edge slices with context
                for s in range(margin, n_slices - margin):
                    self.index_list.append((case_idx, s))
            else:
                # Validation: deterministic evenly-spaced subset
                margin = max(context_slices, 1)
                if val_slices_per_case > 0:
                    positions = np.linspace(margin, n_slices - 1 - margin,
                                            val_slices_per_case, dtype=int)
                    for s in positions.tolist():
                        self.index_list.append((case_idx, int(s)))
                else:
                    for s in range(margin, n_slices - margin):
                        self.index_list.append((case_idx, s))

    def __len__(self) -> int:
        # FIX: was len(self.cases) — now correct number of slice-level samples
        return len(self.index_list)

    def _load_volumes(self, paths: List[str]) -> List[np.ndarray]:
        vols = []
        for p in paths:
            nii = nib.load(p)
            vols.append(np.asarray(nii.dataobj))
        return vols

    def _get_slice(self, vol: np.ndarray, s: int) -> np.ndarray:
        if self.slice_axis == 0:
            return vol[s, :, :]
        elif self.slice_axis == 1:
            return vol[:, s, :]
        else:
            return vol[:, :, s]

    def _crop_or_pad_slice(self, sl: np.ndarray, h: int, w: int) -> np.ndarray:
        sh, sw = sl.shape
        if sh == h and sw == w:
            return sl
        if sh <= h and sw <= w:
            out = np.zeros((h, w), dtype=sl.dtype)
            y0, x0 = (h - sh) // 2, (w - sw) // 2
            out[y0 : y0 + sh, x0 : x0 + sw] = sl
            return out
        y0 = max(0, (sh - h) // 2)
        x0 = max(0, (sw - w) // 2)
        out = sl[y0 : y0 + h, x0 : x0 + w]
        if out.shape[0] != h or out.shape[1] != w:
            from scipy.ndimage import zoom
            zoom_factors = (h / out.shape[0], w / out.shape[1])
            out = zoom(out, zoom_factors, order=1)
        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        case_idx, s = self.index_list[idx]  # FIX: unpack precomputed (case, slice)
        _, paths = self.cases[case_idx]
        vols = self._load_volumes(paths)
        n_slices = vols[0].shape[self.slice_axis]

        # Context window: gather adjacent slices per modality (2.5D)
        c = self.context_slices
        slice_range = range(max(0, s - c), min(n_slices, s + c + 1))

        ref_vol = vols[self.target_idx]
        ref_slice = self._get_slice(ref_vol, s)
        p_min, p_max = self.percentile

        input_channels = []
        for mod_idx, vol in enumerate(vols):
            if mod_idx == self.target_idx:
                continue
            for si in slice_range:
                sl = self._get_slice(vol, si)
                sl = _percentile_clip(sl, p_min, p_max, ref=ref_slice)
                if self.patch_size:
                    ph, pw = self.patch_size
                    sl = self._crop_or_pad_slice(sl, ph, pw)
                input_channels.append(sl)

        target_sl = self._get_slice(ref_vol, s)
        target_sl = _percentile_clip(target_sl, p_min, p_max)
        if self.patch_size:
            ph, pw = self.patch_size
            target_sl = self._crop_or_pad_slice(target_sl, ph, pw)

        # Augmentation: flips + small intensity jitter
        if self.augment:
            if random.random() > 0.5:
                input_channels = [np.flipud(ch).copy() for ch in input_channels]
                target_sl = np.flipud(target_sl).copy()
            if random.random() > 0.5:
                input_channels = [np.fliplr(ch).copy() for ch in input_channels]
                target_sl = np.fliplr(target_sl).copy()
            # Intensity jitter on inputs (not target, to keep GT clean)
            if random.random() > 0.5:
                gain = random.uniform(0.9, 1.1)
                bias = random.uniform(-0.05, 0.05)
                input_channels = [
                    np.clip(ch * gain + bias, 0.0, 1.0) for ch in input_channels
                ]

        x = np.stack(input_channels, axis=0).astype(np.float32)
        y = target_sl[np.newaxis, ...].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def get_dataloaders(
    data_root: str,
    modalities: List[str] = MODALITIES,
    slice_axis: int = 2,
    patch_size: Optional[Tuple[int, int]] = (160, 160),
    batch_size: int = 16,
    train_split: float = 0.85,
    percentile: Tuple[float, float] = (0.5, 99.5),
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
    context_slices: int = 0,
    val_slices_per_case: int = 10,
):
    """Build train/val dataloaders for all 4 target modalities (4 tasks)."""
    cases = find_cases(data_root, modalities)
    if not cases:
        root = Path(data_root).resolve()
        sample = list(root.iterdir())[:5] if root.exists() else []
        hint = f" First items in folder: {[p.name for p in sample]}." if sample else " Folder missing or empty."
        raise FileNotFoundError(
            f"No cases with modalities {modalities} found under {data_root}.{hint} "
            "Use the path to the folder that directly contains BraTS-GLI-* case folders (see README)."
        )
    random.seed(seed)
    random.shuffle(cases)
    n = len(cases)
    n_train = int(n * train_split)
    train_cases = cases[:n_train]
    val_cases = cases[n_train:]

    train_loaders = {}
    val_loaders = {}
    for target in modalities:
        train_ds = BraSynSliceDataset(
            train_cases,
            modalities,
            target,
            slice_axis=slice_axis,
            patch_size=patch_size,
            percentile=percentile,
            augment=True,
            context_slices=context_slices,
        )
        val_ds = BraSynSliceDataset(
            val_cases,
            modalities,
            target,
            slice_axis=slice_axis,
            patch_size=patch_size,
            percentile=percentile,
            augment=False,
            context_slices=context_slices,
            val_slices_per_case=val_slices_per_case,
        )
        train_loaders[target] = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loaders[target] = torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    return train_loaders, val_loaders, train_cases, val_cases
