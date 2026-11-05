"""
Explainability & Accessibility Visualizations for BraSyn MRI Synthesis.

Designed so someone from NLP, CV, or any non-medical-imaging background
can understand exactly what the model is doing and why.

Each function produces a labeled, captioned figure with:
  - Plain-English title and subtitle
  - An NLP/general-ML analogy in the caption
  - No assumed medical imaging knowledge

Functions:
  plot_attention_maps()       — Where does the model focus? (like transformer attention)
  plot_error_map()            — Where did the model go wrong?
  plot_modality_importance()  — Which input modality matters most? (occlusion study)
  plot_synthesis_journey()    — Synthesis across the full brain volume (multi-slice strip)
  plot_intensity_distribution()— Does the model match the true pixel distribution?
  plot_mri_explainer()        — What IS MRI? (static reference card)

All functions return a matplotlib Figure so they can be embedded
in the Streamlit app or saved to disk.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import nibabel as nib


# ─── 1. Attention Map Visualizer ─────────────────────────────────────────────

def plot_attention_maps(
    model,
    x: torch.Tensor,
    background_slice: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize where the Attention U-Net focuses on the input image.

    Uses the stored _last_attention tensors from AttentionGate modules.
    Alpha-blends each attention map over the background MRI slice.

    NLP analogy: "Like visualizing which tokens a transformer attends to
    at each layer — but the 'tokens' are spatial image patches."

    Args:
        model:            Trained UNet2D with use_attention=True.
        x:                Input tensor (1, C, H, W).
        background_slice: 2D numpy array [H, W] for overlay (typically T1 slice).
        save_path:        If set, save figure to this path.
    """
    model.eval()
    with torch.no_grad():
        _ = model(x)  # populate _last_attention in each AttentionGate

    attn_maps = model.get_attention_maps()
    if not attn_maps:
        print("[explainability] No attention maps found. Is use_attention=True?")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No attention gates found\n(model trained without use_attention)",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    n = len(attn_maps)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
    if n == 1:
        axes = [axes]

    # Background
    bg = background_slice
    ax0 = axes[0] if isinstance(axes, (list, np.ndarray)) else axes
    ax0.imshow(bg, cmap="gray")
    ax0.set_title("Input: T1\n(background)", fontsize=9)
    ax0.axis("off")

    for i, attn in enumerate(attn_maps):
        a = attn.squeeze().cpu().numpy()
        # Resize to match background if needed
        if a.shape != bg.shape:
            from scipy.ndimage import zoom
            a = zoom(a, (bg.shape[0] / a.shape[0], bg.shape[1] / a.shape[1]), order=1)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)

        ax = axes[i + 1]
        ax.imshow(bg, cmap="gray")
        ax.imshow(a, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(f"Attention layer {i+1}\n(decoder depth {n-i})", fontsize=9)
        ax.axis("off")

    fig.suptitle(
        "Attention Maps — Where does the model look?\n"
        "NLP analogy: like transformer cross-attention heatmaps over input tokens, "
        "but applied to brain image patches",
        fontsize=9,
    )
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── 2. Error Map ─────────────────────────────────────────────────────────────

def plot_error_map(
    pred: np.ndarray,
    ground_truth: np.ndarray,
    target_mod: str = "FLAIR",
    ssim_val: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Pixel-wise absolute error |pred - GT| as a hot colormap.

    "Red = model was very wrong here; dark = model was accurate."
    NLP analogy: "Like a token-level error heatmap on a machine translation output."
    """
    error = np.abs(pred - ground_truth)
    mae = error.mean()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(ground_truth, cmap="gray"); axes[0].set_title("Ground Truth"); axes[0].axis("off")
    axes[1].imshow(pred, cmap="gray"); axes[1].set_title("Synthesized"); axes[1].axis("off")
    im = axes[2].imshow(error, cmap="hot", vmin=0, vmax=error.max())
    axes[2].set_title(f"Error map (MAE={mae:.4f})\nRed=wrong, Dark=accurate")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    ssim_str = f"  |  SSIM={ssim_val:.4f}" if ssim_val is not None else ""
    fig.suptitle(
        f"Synthesis Error Map — {target_mod}{ssim_str}\n"
        "NLP analogy: token-level error heatmap (red patches = model needs more training data here)",
        fontsize=9,
    )
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── 3. Modality Importance (Input Occlusion) ─────────────────────────────────

def plot_modality_importance(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    input_mods: list,
    target_mod: str,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Zero out one modality at a time; measure SSIM drop.
    Bar chart shows relative importance of each input scan.

    NLP analogy: "Like ablating one input token or feature — which input
    matters most for predicting the output?"
    """
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

    model.eval()
    with torch.no_grad():
        base_out = model(x)
        if isinstance(base_out, tuple):
            base_out = base_out[0]
        base_ssim = ssim_fn(base_out.cpu().clamp(0, 1), y.cpu()).item()

    drops = []
    for i, mod in enumerate(input_mods):
        x_occ = x.clone()
        x_occ[:, i, :, :] = 0.0  # zero out channel i
        with torch.no_grad():
            occ_out = model(x_occ)
            if isinstance(occ_out, tuple):
                occ_out = occ_out[0]
            occ_ssim = ssim_fn(occ_out.cpu().clamp(0, 1), y.cpu()).item()
        drops.append(base_ssim - occ_ssim)

    mod_labels = {"t1n": "T1", "t1c": "T1ce", "t2w": "T2", "t2f": "FLAIR"}
    labels = [mod_labels.get(m, m) for m in input_mods]
    colors = ["#e74c3c" if d == max(drops) else "#3498db" for d in drops]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, drops, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("SSIM drop when modality removed")
    ax.set_xlabel("Removed MRI modality")
    ax.set_title(
        f"Modality Importance for Synthesizing {mod_labels.get(target_mod, target_mod)}\n"
        "NLP analogy: feature ablation — which input 'token' is most informative?\n"
        "(Tallest bar = most important scan to have available)",
        fontsize=9,
    )
    for bar, d in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"Δ={d:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(drops) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── 4. Synthesis Journey (Multi-Slice Strip) ─────────────────────────────────

def plot_synthesis_journey(
    model,
    vol_paths: dict,        # {modality: path_str}
    target_mod: str,
    patch_size: tuple = (160, 160),
    n_slices: int = 5,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Show synthesis at multiple depths (20%, 40%, 60%, 80% of brain).
    For each depth: GT | Synthesized | Error.

    "The model synthesizes every 2D cross-section of the 3D brain — front to back."
    NLP analogy: "Like generating output for every token position in a sequence."
    """
    from dataset import MODALITIES, _percentile_clip
    from scipy.ndimage import zoom

    available = [m for m in MODALITIES if m != target_mod]
    vols = {m: np.asarray(nib.load(vol_paths[m]).dataobj) for m in MODALITIES}
    total_slices = vols[target_mod].shape[2]
    positions = np.linspace(int(0.15 * total_slices), int(0.85 * total_slices),
                            n_slices, dtype=int)
    ph, pw = patch_size

    def prep(vol, s):
        sl = vol[:, :, s]
        sl = _percentile_clip(sl, 0.5, 99.5)
        sh, sw = sl.shape
        if sh != ph or sw != pw:
            sl = zoom(sl, (ph / sh, pw / sw), order=1)
        return sl.astype(np.float32)

    model.eval()
    fig, axes = plt.subplots(3, n_slices, figsize=(3.5 * n_slices, 10))

    for col, s in enumerate(positions):
        inp = np.stack([prep(vols[m], s) for m in available], axis=0)
        gt = prep(vols[target_mod], s)
        x = torch.from_numpy(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            pred = out.squeeze().cpu().numpy().clip(0, 1)

        depth_pct = int(100 * s / total_slices)
        axes[0, col].imshow(gt, cmap="gray")
        axes[0, col].set_title(f"Depth {depth_pct}%\nGround Truth", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(pred, cmap="gray")
        axes[1, col].set_title("Synthesized", fontsize=8)
        axes[1, col].axis("off")
        err = np.abs(pred - gt)
        axes[2, col].imshow(err, cmap="hot")
        axes[2, col].set_title(f"Error (MAE={err.mean():.3f})", fontsize=8)
        axes[2, col].axis("off")

    fig.suptitle(
        "Synthesis Journey — Across the Full Brain Volume\n"
        "NLP analogy: generating output at every position in a sequence, "
        "top-to-bottom = front-to-back through the brain",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── 5. Pixel Intensity Distribution ─────────────────────────────────────────

def plot_intensity_distribution(
    pred: np.ndarray,
    ground_truth: np.ndarray,
    target_mod: str = "FLAIR",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay histogram of GT vs synthesized pixel intensities.

    "A well-calibrated model should match the true distribution."
    NLP analogy: "Like comparing the token probability distributions of
    a language model's output vs. the reference corpus."
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ground_truth.flatten(), bins=80, alpha=0.6, color="#2ecc71",
            label="Ground Truth", density=True)
    ax.hist(pred.flatten(), bins=80, alpha=0.6, color="#e74c3c",
            label="Synthesized", density=True)
    ax.set_xlabel("Normalised pixel intensity")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(
        f"Pixel Intensity Distribution: Ground Truth vs. Synthesized {target_mod}\n"
        "NLP analogy: comparing output token probability distribution to reference "
        "(closer = better-calibrated model)",
        fontsize=9,
    )
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── 6. "What is MRI?" Explainer Card ────────────────────────────────────────

def plot_mri_explainer(
    sample_slices: Optional[dict] = None,   # {modality: np.ndarray [H,W]}
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Static reference card explaining the 4 MRI modalities in plain English.
    Shows sample slices (or placeholder) with labeled descriptions.
    No medical imaging background required.
    """
    descriptions = {
        "t1n":  ("T1 — Anatomy\n(water = dark, fat = bright)\n"
                  "Analogy: the 'default view'\nof a document, raw text"),
        "t1c":  ("T1ce — Contrast-Enhanced\n(tumors light up with dye)\n"
                  "Analogy: highlight mode —\nonly important parts glow"),
        "t2w":  ("T2 — Water-Sensitive\n(fluid = bright, white matter = dark)\n"
                  "Analogy: like TF-IDF weighting,\nfluid-heavy regions score high"),
        "t2f":  ("FLAIR — Tumor-Sensitive\n(suppresses normal fluid,\nkeeps tumor signal)\n"
                  "Analogy: a classifier trained\nto flag anomalies only"),
    }
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    cmap_colors = ["Blues", "Oranges", "Greens", "Purples"]

    for i, (mod, desc) in enumerate(descriptions.items()):
        ax = axes[i]
        if sample_slices and mod in sample_slices:
            ax.imshow(sample_slices[mod], cmap="gray")
        else:
            # Placeholder noise pattern so the card isn't blank
            rng = np.random.default_rng(i)
            placeholder = rng.uniform(0, 1, (160, 160))
            ax.imshow(placeholder, cmap=cmap_colors[i], alpha=0.5)
            ax.text(80, 80, f"[{mod}]\nsample\nnot loaded",
                    ha="center", va="center", fontsize=8, color="white",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.4))
        ax.set_title(desc, fontsize=8, pad=6)
        ax.axis("off")
        # Color-coded border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)

    fig.suptitle(
        "What is MRI? — A Quick Guide (No Medical Background Needed)\n"
        "Each modality = a different 'feature extractor' applied to the same brain\n"
        "The model synthesizes the MISSING one from the other three",
        fontsize=10, y=1.03,
    )
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
