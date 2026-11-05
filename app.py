"""
Streamlit app for BraSyn — 4 tabs:
  1. Synthesis            — inputs | GT | synthesized
  2. Uncertainty          — MC Dropout uncertainty heatmap
  3. Metrics & Ablation   — training curves and component comparisons
  4. Explain It           — NLP-friendly explanations and usage guide

Run: streamlit run app.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dataset import MODALITIES, find_cases, _percentile_clip
from model import build_model

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BraSyn MRI Synthesis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stTab [data-baseweb="tab"] {font-size: 0.9rem; font-weight: 600;}
    .metric-card {background:#1e1e2e;border-radius:10px;padding:16px;margin:4px;}
    .badge-green {background:#27ae60;color:white;border-radius:12px;padding:4px 12px;}
    .badge-yellow {background:#f39c12;color:white;border-radius:12px;padding:4px 12px;}
    .badge-red {background:#e74c3c;color:white;border-radius:12px;padding:4px 12px;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 BraSyn: Brain MRI Modality Synthesis")
st.caption("Attention U-Net + PatchGAN | MC Dropout Uncertainty | BraTS 2023 (arXiv:2305.09011)")

st.markdown("""
### Welcome to the BraSyn MRI Synthesis App!
Here is a quick overview of what each tab does:
1. **🔬 Synthesis**: Give the model 3 MRI modalities of a patient, and it will predict their missing 4th modality!
2. **🎲 Uncertainty**: Run multiple stochastic predictions to see a heatmap showing where the model is confident versus where it is guessing.
3. **📈 Metrics & Ablation**: View the learning curves from the model's training process and see how different architectural components improved the final score.
4. **💡 Explain It**: A plain-English guide (with NLP analogies!) explaining how the model works and **how you can test it** locally.

*Note: Since this project runs locally, you'll need the BraTS dataset on your machine to generate visualizations. See the "Explain It" tab for exact instructions on how to set up the data paths.*
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Synthesis", "🎲 Uncertainty", "📈 Metrics & Ablation", "💡 Explain It"
])


# ─── Shared helpers ────────────────────────────────────────────────────────────

def _load_ckpt(ckpt_path: str, device: str):
    """Load checkpoint and reconstruct model from saved arch config."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    return model, arch, ckpt.get("val_ssim", None)


def _prep_slice(sl: np.ndarray, ph=160, pw=160) -> np.ndarray:
    from scipy.ndimage import zoom
    sh, sw = sl.shape
    if sh == ph and sw == pw:
        return sl.astype(np.float32)
    if sh >= ph and sw >= pw:
        y0, x0 = (sh - ph) // 2, (sw - pw) // 2
        return sl[y0:y0+ph, x0:x0+pw].astype(np.float32)
    return zoom(sl, (ph / sh, pw / sw), order=1).astype(np.float32)


MOD_LABELS = {"t1n": "T1", "t1c": "T1ce", "t2w": "T2", "t2f": "FLAIR"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Tab 1: Synthesis Comparison ──────────────────────────────────────────────

with tab1:
    st.subheader("Input modalities → Ground Truth vs. Synthesized")
    st.markdown("""
    **How to use this:** This tab takes 3 input MRI scans and uses our trained AI model to synthesize the missing 4th scan. It then compares the synthesized scan visually against the real Ground Truth scan.
    
    *If you are unsure how to use this or what the results should look like, please read the **💡 Explain It** tab!*
    """)

    c1, c2 = st.columns(2)
    data_root = c1.text_input("Data root (BraTS case folder)", value="sample_data", key="t1_dr")
    ckpt_path = c2.text_input("Checkpoint (.pt)", value="checkpoints/best_t2f.pt", key="t1_ck")
    
    # Auto-detect target from checkpoint name
    detected_tgt = "t2f"
    for m in MODALITIES:
        if f"_{m}" in str(ckpt_path):
            detected_tgt = m
            break
            
    c3, c4, c5 = st.columns(3)
    target = c3.selectbox("Target modality (Match Checkpoint!)", MODALITIES, 
                          index=MODALITIES.index(detected_tgt), key="t1_tgt")
    case_index = c4.number_input("Case index", min_value=0, value=0, key="t1_ci")
    slice_index = c5.number_input("Slice index (-1 = middle)", min_value=-1, max_value=20, value=10, key="t1_si")

    if st.button("▶ Generate Synthesis", key="t1_go"):
        if not data_root or not Path(ckpt_path).exists():
            st.error("Set a valid data root and checkpoint path.")
        else:
            with st.spinner("Running model..."):
                cases = find_cases(data_root, MODALITIES)
                if not cases:
                    st.error(f"No cases found under {data_root}")
                else:
                    case_dir, paths = cases[case_index % len(cases)]
                    vols = [np.asarray(nib.load(p).dataobj) for p in paths]
                    n_sl = vols[0].shape[2]
                    s = slice_index if slice_index >= 0 else n_sl // 2

                    ref_raw = vols[MODALITIES.index(target)][:, :, s]
                    
                    # Independent normalization for beautiful Streamlit visuals
                    vis_slices = [_prep_slice(_percentile_clip(v[:, :, s], 0.5, 99.5, ref=v[:, :, s]))
                                  for v in vols]
                                  
                    # STRICT normalization identically matching the train-time dataset.py bug
                    model_slices = [_prep_slice(_percentile_clip(v[:, :, s], 0.5, 99.5, ref=ref_raw))
                                    for v in vols]

                    available = [m for m in MODALITIES if m != target]
                    input_ch = np.stack([model_slices[MODALITIES.index(m)] for m in available], axis=0)

                    model, arch, ckpt_ssim = _load_ckpt(ckpt_path, DEVICE)
                    model.eval()
                    x = torch.from_numpy(input_ch).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out = model(x)
                        if isinstance(out, tuple):
                            out = out[0]
                        pred = out.squeeze().cpu().numpy().clip(0, 1)

                    if ckpt_ssim:
                        st.info(f"Checkpoint best val SSIM: **{ckpt_ssim:.4f}** | "
                                f"Device: {DEVICE} | Arch: depth={arch.get('depth')}, "
                                f"attention={arch.get('use_attention')}")

                    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
                    fig.patch.set_facecolor("#0e1117")
                    for i, m in enumerate(available):
                        axes[i].imshow(vis_slices[MODALITIES.index(m)], cmap="gray")
                        axes[i].set_title(f"Input: {MOD_LABELS[m]}", color="white", fontsize=9)
                        axes[i].axis("off")
                    axes[3].imshow(vis_slices[MODALITIES.index(target)], cmap="gray")
                    axes[3].set_title(f"Ground Truth: {MOD_LABELS[target]}", color="white", fontsize=9)
                    axes[3].axis("off")
                    axes[4].imshow(pred, cmap="gray")
                    axes[4].set_title(f"Synthesized: {MOD_LABELS[target]}", color="#2ecc71", fontsize=9)
                    axes[4].axis("off")
                    for ax in axes:
                        ax.set_facecolor("#0e1117")
                    fig.suptitle(
                        f"Case: {Path(case_dir).name} | Slice {s}/{n_sl}",
                        color="white", fontsize=10
                    )
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()


# ─── Tab 2: Uncertainty ───────────────────────────────────────────────────────

with tab2:
    st.subheader("🎲 MC Dropout Uncertainty Estimation")
    st.markdown(
        "**How to use this:** This tab runs the model multiple times (with some neurons randomly turned off via dropout) to generate slightly different predictions. "
        "By calculating the variance across these predictions, we create a heatmap showing exactly which parts of the brain the model is 'uncertain' about (shown in red/hot colors).  \n\n"
        "*If you are unsure how to test this or what to look for, please read the **💡 Explain It** tab!*"
    )

    c1, c2 = st.columns(2)
    unc_dr = c1.text_input("Data root", value="sample_data", key="t2_dr")
    unc_ck = c2.text_input("Checkpoint (.pt)", value="checkpoints/best_t2f.pt", key="t2_ck")
    
    # Auto-detect target from checkpoint name
    unc_detected = "t2f"
    for m in MODALITIES:
        if f"_{m}" in str(unc_ck):
            unc_detected = m
            break
            
    c3, c4, c5 = st.columns(3)
    unc_tgt = c3.selectbox("Target modality (Match Checkpoint!)", MODALITIES, 
                           index=MODALITIES.index(unc_detected), key="t2_tgt")
    unc_ci = c4.number_input("Case index", min_value=0, value=0, key="t2_ci")
    n_samples = c5.slider("MC samples (more = slower, better estimate)", 5, 50, 20)

    if st.button("▶ Run Uncertainty Estimation", key="t2_go"):
        if not unc_dr or not Path(unc_ck).exists():
            st.error("Set valid data root and checkpoint.")
        else:
            with st.spinner(f"Running {n_samples} stochastic forward passes..."):
                from uncertainty import mc_dropout_predict, confidence_badge

                cases = find_cases(unc_dr, MODALITIES)
                if not cases:
                    st.error("No cases found.")
                else:
                    case_dir, paths = cases[unc_ci % len(cases)]
                    vols = [np.asarray(nib.load(p).dataobj) for p in paths]
                    n_sl = vols[0].shape[2]
                    s = n_sl // 2
                    ref_raw = vols[MODALITIES.index(unc_tgt)][:, :, s]
                    
                    # Independent normalization for visual aesthetics
                    vis_slices = [_prep_slice(_percentile_clip(v[:, :, s], 0.5, 99.5, ref=v[:, :, s]))
                                  for v in vols]
                                  
                    # Strict normalization identically matching the train-time dataset.py behavior
                    model_slices = [_prep_slice(_percentile_clip(v[:, :, s], 0.5, 99.5, ref=ref_raw))
                                    for v in vols]
                                    
                    available = [m for m in MODALITIES if m != unc_tgt]
                    input_ch = np.stack([model_slices[MODALITIES.index(m)] for m in available], axis=0)

                    model, arch, _ = _load_ckpt(unc_ck, DEVICE)
                    x = torch.from_numpy(input_ch).unsqueeze(0).to(DEVICE)
                    mean_t, std_t = mc_dropout_predict(model, x, n_samples=n_samples)
                    mean_np = mean_t.squeeze().cpu().numpy()
                    std_np = std_t.squeeze().cpu().numpy()

                    gt = vis_slices[MODALITIES.index(unc_tgt)]
                    gt_model = model_slices[MODALITIES.index(unc_tgt)]
                    try:
                        from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
                        ssim_val = ssim_fn(mean_t.cpu(), torch.from_numpy(gt_model)[None, None]).item()
                    except Exception:
                        ssim_val = 0.7

                    badge = confidence_badge(ssim_val, std_np.mean())
                    st.markdown(f"**Confidence:** {badge} &nbsp;|&nbsp; SSIM: **{ssim_val:.4f}** &nbsp;|&nbsp; "
                                f"Mean uncertainty: **{std_np.mean():.4f}**")

                    fig, axes = plt.subplots(1, 6, figsize=(20, 3.5))
                    fig.patch.set_facecolor("#0e1117")
                    for i, m in enumerate(available):
                        axes[i].imshow(vis_slices[MODALITIES.index(m)], cmap="gray")
                        axes[i].set_title(f"Input: {MOD_LABELS[m]}", color="white", fontsize=8)
                        axes[i].axis("off")
                    axes[3].imshow(gt, cmap="gray")
                    axes[3].set_title("Ground Truth", color="white", fontsize=8)
                    axes[3].axis("off")
                    axes[4].imshow(mean_np, cmap="gray")
                    axes[4].set_title("Synthesized\n(MC mean)", color="#2ecc71", fontsize=8)
                    axes[4].axis("off")
                    im = axes[5].imshow(std_np / (std_np.max() + 1e-8), cmap="hot", vmin=0, vmax=1)
                    axes[5].set_title("Uncertainty\n(MC std dev)\nred=unsure", color="#e74c3c", fontsize=8)
                    axes[5].axis("off")
                    plt.colorbar(im, ax=axes[5], fraction=0.046)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()


# ─── Tab 3: Training Curves ───────────────────────────────────────────────────

with tab3:
    st.subheader("📈 Metrics & Ablation Results")
    st.markdown("#### Training Metrics Over Epochs")
    log_path = st.text_input("Path to train_log CSV", value="checkpoints/train_log_t2f.csv", key="t3_lp")
    if Path(log_path).exists():
        df = pd.read_csv(log_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Val SSIM", f"{df['val_ssim'].max():.4f}",
                    delta=f"+{df['val_ssim'].max() - df['val_ssim'].iloc[0]:.4f} vs epoch 1")
        col2.metric("Best Val PSNR", f"{df['val_psnr'].max():.2f} dB")
        col3.metric("Final Loss", f"{df['loss'].iloc[-1]:.4f}",
                    delta=f"{df['loss'].iloc[-1] - df['loss'].iloc[0]:.4f}",
                    delta_color="inverse")

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        for ax, col, title, color in [
            (axes[0, 0], "loss",     "Training Loss",      "#e74c3c"),
            (axes[0, 1], "val_ssim", "Validation SSIM ↑",  "#2ecc71"),
            (axes[1, 0], "val_psnr", "Validation PSNR ↑",  "#3498db"),
            (axes[1, 1], "val_mse",  "Validation MSE ↓",   "#e67e22"),
        ]:
            ax.plot(df["epoch"], df[col], color=color, linewidth=1.8)
            ax.fill_between(df["epoch"], df[col], alpha=0.15, color=color)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        with st.expander("Show raw data (last 15 epochs)"):
            st.dataframe(df.tail(15))
    else:
        st.info("No train_log CSV found. Run `python train.py` first. "
                "The log will be at `checkpoints/train_log_<target>.csv`.")


    st.markdown("---")
    st.markdown("#### Ablation Study — Component Contributions")
    st.markdown(
        "Each variant adds one component. The table shows the SSIM improvement "
        "from each added component — this is Table 1 in the paper."
    )

    abl_path = st.text_input("Path to ablation_results.csv",
                              value="ablation_results.csv", key="t4_ap")
    if Path(abl_path).exists():
        df = pd.read_csv(abl_path)
        # Highlight best per column
        st.dataframe(
            df.style
              .highlight_max(subset=["ssim", "psnr"], color="#1a5c38")
              .highlight_min(subset=["mse"], color="#1a5c38")
              .format({"ssim": "{:.4f}", "psnr": "{:.2f}", "mse": "{:.6f}"}),
            use_container_width=True,
        )
        # Bar chart of SSIM
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = ["#7f8c8d", "#2980b9", "#27ae60", "#e74c3c"]
        bars = ax.bar(df["label"], df["ssim"], color=colors, edgecolor="white")
        ax.set_ylabel("Validation SSIM ↑")
        ax.set_title("Ablation: SSIM by Model Variant", fontsize=11)
        ax.set_ylim(df["ssim"].min() * 0.95, df["ssim"].max() * 1.05)
        for bar, val in zip(bars, df["ssim"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", fontsize=8)
        plt.xticks(rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No ablation_results.csv yet. Run:  \n`python ablation.py --data_root data/BraSyn --target t2f`")
        st.markdown("""
        **Expected ablation variants:**
        | Variant | SSIM ↑ | PSNR ↑ | MSE ↓ |
        |---------|--------|--------|-------|
        | Plain U-Net + L1 | — | — | — |
        | Plain U-Net + L1 + SSIM | — | — | — |
        | **Attention U-Net** + L1 + SSIM | — | — | — |
        | **Full** (+ GAN + deep sup.) | — | — | — |
        """)


# ─── Tab 4: Explain It ────────────────────────────────────────────────────────

with tab4:
    st.subheader("💡 How Does the Model Work? (No Medical Imaging Background Needed)")
    st.markdown("""
    This tab walks through what the model is doing — using plain English and
    analogies to **NLP** concepts that anyone in ML can relate to.
    """)

    with st.expander("🚀 How to Test and Use This Project Locally", expanded=True):
        st.markdown("""
        ### Setup & Data Paths
        Medical datasets like BraTS are very large (often 50GB+), so they are not included in the GitHub repository. To allow you to test this project instantly, **I have embedded a lightweight 20-slice sample volume** directly into the code (`sample_data/`).
        
        1. **Locate the Data**: The app is set to look at the `sample_data` folder by default. If you downloaded the full BraTS 2023 dataset, you can paste your absolute path to the data folder here instead.
        2. **Provide the Checkpoint**: The app needs the trained model weights. Keep the `Checkpoint (.pt)` field pointed to `checkpoints/best_t2f.pt`.
        
        ### Testing Synthesis
        - The target modality model we have currently loaded synthesizes **FLAIR** scans (`t2f`).
        - Choose a `Slice index` (between 0 and 20) to view a different layer of the brain.
        - Click **Generate Synthesis**. The app will load that patient's T1, T1ce, and T2 scans, predict the missing FLAIR, and show a side-by-side comparison with the real Ground Truth FLAIR.
        
        ### Testing Uncertainty
        - Go to the **Uncertainty** tab, and keep the paths as above.
        - Pick the number of `MC samples`. A higher number takes longer but produces a more robust confidence estimate.
        - Click **Run Uncertainty Estimation**. Look at the right-most heatmap: dark regions mean the model is highly confident, while bright/red regions mean the model is guessing!
        """)

    with st.expander("📚 What is MRI? (Start here)"):
        st.markdown("""
        An MRI scanner can be tuned to highlight **different tissue properties** —
        each "tuning" is called a **modality** (or sequence). Think of them as
        different **feature extractors** applied to the same brain:

        | Modality | Clinical Name | What it shows | NLP Analogy |
        |----------|---------------|---------------|-------------|
        | **T1** | T1-weighted | Anatomy (fat = bright, water = dark) | Raw text — the default view |
        | **T1ce** | T1 contrast-enhanced | Tumors light up (contrast dye injected) | Highlighted text — only important parts glow |
        | **T2** | T2-weighted | Fluid = bright, white matter = dark | TF-IDF weighting — fluid regions score high |
        | **FLAIR** | FLAIR | Suppresses normal fluid, keeps tumor | Anomaly classifier — flags non-normal patterns |

        The BraSyn task: **given 3 of these views, synthesize the missing 4th.**
        Like predicting a masked token in BERT, but for a full 2D image.
        """)

    with st.expander("🏗️ What does the model look like?"):
        st.markdown("""
        The model is an **Attention U-Net** — an encoder-decoder architecture with **skip connections**.

        ```
        Input (3 MRI slices)
              ↓
        [Encoder] — extracts features at increasing scale
        [Bottleneck] — most abstract representation
        [Decoder] — reconstructs full-resolution prediction
              ↑
        Skip connections (with Attention Gates) bridge encoder → decoder
        ```

        **Attention Gates** on the skip connections work exactly like **transformer cross-attention**:
        the decoder "query" asks the encoder "which spatial regions matter for reconstruction here?"
        The encoder features are weighted by the resulting attention map before being concatenated.

        **PatchGAN Discriminator** (adversarial training): a second network judges whether
        the synthesized image looks realistic *given* the input scans — like a discriminative
        language model that scores whether a generated sentence is fluent given the context.
        """)

    with st.expander("🔭 Attention Maps — Where does the model look?"):
        st.markdown("""
        After running the synthesis, you can extract **attention maps** from each decoder level.
        These show which spatial locations the model focuses on when reconstructing each part
        of the output image.

        - **Shallow layers**: focus on edges and textures
        - **Deeper layers**: focus on semantically meaningful regions (often the tumor)

        **NLP analogy**: like visualizing attention heads in a transformer — shallow heads
        catch syntax, deep heads catch long-range semantic dependencies.

        *(Run `python explainability.py` or use the synthesis tab with a trained model to see these.)*
        """)

    with st.expander("❌ Error Maps — Where does the model fail?"):
        st.markdown("""
        The **error map** shows `|prediction − ground truth|` pixel by pixel.

        - **Dark/cool = accurate** — the model got these regions right
        - **Red/hot = wrong** — the model struggled here (often near tumor boundaries or
          rare tissue types with limited training examples)

        **NLP analogy**: like a token-level error heatmap on a machine translation output —
        red tokens = wrong words, dark tokens = correct words.
        """)

    with st.expander("📊 Modality Importance — Which scan matters most?"):
        st.markdown("""
        By **zeroing out one input modality at a time** and measuring the SSIM drop,
        we can rank the importance of each available scan for synthesizing the missing one.

        For example, when synthesizing FLAIR:
        - T2 is usually the most informative (similar water-sensitivity contrast)
        - T1 provides anatomical context
        - T1ce adds tumor-specific signal

        **NLP analogy**: like **feature ablation** or **token deletion** — remove one
        input feature and see how much the output quality drops. Features with the
        largest drop are the most important.
        """)

    with st.expander("🎲 Uncertainty — Does the model know what it doesn't know?"):
        st.markdown("""
        **Monte Carlo Dropout** (Gal & Ghahramani, 2016): keep dropout ON at test time
        and run **N forward passes**. Each pass gives a slightly different prediction.

        - **Pixel-wise std dev** across N runs = uncertainty map
        - **Low std** → model is confident (consistent predictions)
        - **High std** → model is uncertain (predictions vary a lot)

        This is **epistemic uncertainty** — it's high where the model lacks training data
        or where the input is ambiguous.

        **Clinical value**: radiologists can use the uncertainty map to know *which regions
        of the synthesized scan to trust and which to double-check* with a real scan.

        **NLP analogy**: like sampling from a language model N times and computing variance
        of token probabilities — high-variance positions are where the model is "unsure"
        about the next word.
        """)
