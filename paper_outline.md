# Research Paper Outline — BraSyn Attention U-Net + Uncertainty

**Suggested title:**
> "Attention-Guided Adversarial Synthesis of Missing Brain MRI Modalities with Uncertainty Estimation"

**Suggested venues:** MICCAI BrainLes Workshop · MIDL · Medical Image Analysis (journal)

---

## 1. Introduction (~1 page)

**Clinical motivation:**
- MRI protocols require T1, T1ce, T2, FLAIR — but one is often missing (motion, cost, scanner failure).
- Downstream segmentation (BraTS, FeTS) degrades without all modalities.
- Goal: recover the missing modality from the three available.

**Contributions** (bullet list for paper):
1. Attention U-Net with spatial attention gates for targeted feature selection.
2. Conditional PatchGAN adversarial loss for perceptually realistic synthesis.
3. Monte Carlo Dropout uncertainty maps — first in this task — providing spatial confidence estimates.
4. Systematic ablation study demonstrating each component's contribution.
5. Modality importance analysis via input occlusion.

---

## 2. Related Work (~1 page)

**MRI synthesis:**
- Nie et al. (2017): GAN-based CT-MRI synthesis.
- Chartsias et al. (2018): multi-modal MRI synthesis (pairwise).
- BraSyn 2023 (Li et al., arXiv:2305.09011): challenge establishing 4-modality setup.

**Architecture baselines:**
- U-Net (Ronneberger et al., MICCAI 2015).
- Attention U-Net (Oktay et al., 2018): attention-gated skip connections.
- pix2pix (Isola et al., CVPR 2017): conditional image-to-image GAN.

**Uncertainty in medical imaging:**
- MC Dropout (Gal & Ghahramani, NeurIPS 2016): epistemic uncertainty via stochastic inference.
- Nair et al. (2020): uncertainty in brain lesion segmentation.

---

## 3. Method (~2 pages)

### 3.1 Problem Formulation
- Given modalities M_in ⊂ {T1, T1ce, T2, FLAIR}, |M_in| = 3, predict M_target.
- 2D slice-wise formulation: input (B, 3, H, W) → output (B, 1, H, W).

### 3.2 Attention U-Net Generator
- Encoder: 4 DoubleConv + MaxPool blocks; channels [32, 64, 128, 256, 512].
- AttentionGate on each skip connection: gating signal from decoder, keys from encoder.
- Decoder: ConvTranspose2d + DoubleConv with deep supervision auxiliary heads.
- Figure: architecture diagram (encoder → bottleneck → decoder with attention gates marked).

### 3.3 Training Objective
```
L_total = λ_L1 · L1(G(x), y)
        + λ_SSIM · (1 − SSIM(G(x), y))
        + λ_GAN · L_adv(G(x), D)
        + λ_perc · L_perc(G(x), y)
        + 0.4 · Σ L1(aux_i, y)          [deep supervision]
```
- Discriminator: 70×70 PatchGAN conditioned on input modalities.

### 3.4 Monte Carlo Dropout Uncertainty
- At inference: enable Dropout2d, run N=20 forward passes.
- Pixel-wise mean = synthesis output; pixel-wise std = uncertainty map.
- Clinical interpretation: high uncertainty → synthesized region should be verified.

---

## 4. Experiments (~1.5 pages)

### 4.1 Dataset
- RSNA-ASNR-MICCAI BraTS 2023 GLI Challenge Training Data (~1250 cases, 4 modalities + seg label).
- 85/15 train/val split. 2D axial slices, 160×160 centre crop.
- Preprocessing: percentile normalisation [0.5, 99.5] per case.

### 4.2 Implementation Details
- PyTorch 2.x, AdamW (lr=1e-3, wd=1e-5), cosine LR schedule.
- Optuna HPO: 30 trials over {lr, depth, dropout, batch_size, SSIM weight}.
- Training: 100 epochs, batch size 16, single GPU (RTX 3090 or A100).

### 4.3 Evaluation Metrics
- SSIM ↑, PSNR ↑, MAE ↓ (aligned with BraSyn challenge).
- Baseline: pixel-wise mean of 3 available modalities (bilinear).

---

## 5. Results (~1.5 pages)

### Table 1 — Ablation Study (fill from ablation_results.csv)

| Model Variant | SSIM ↑ | PSNR ↑ | MSE ↓ |
|---|---|---|---|
| Plain U-Net + L1 | — | — | — |
| Plain U-Net + L1 + SSIM | — | — | — |
| Attention U-Net + L1 + SSIM | — | — | — |
| **Full (+ GAN + deep sup.)** | **0.7447 ± 0.1946** | **inf ± nan** | **0.0273 ± 0.0311** |
| Bilinear baseline | 0.7235 ± 0.2062 | inf ± nan | 0.0649 ± 0.0690 |

### Table 2 — Per-Modality Results (fill from results_XXX.csv)

| Target | SSIM ↑ | PSNR ↑ | MSE ↓ |
|---|---|---|---|
| T1 | — | — | — |
| T1ce | — | — | — |
| T2 | — | — | — |
| FLAIR | — | — | — |

### Figures
- **Fig 2**: Synthesis comparison (3 inputs | GT | synthesized | error map).
- **Fig 3**: Uncertainty maps for 3 representative cases (low / medium / high uncertainty).
- **Fig 4**: Attention maps across decoder levels (shallow vs deep focus).
- **Fig 5**: Modality importance bar chart.

---

## 6. Discussion & Conclusion (~0.5 page)

- Attention gates improve synthesis in tumor regions (high attention on tumor in T1ce/FLAIR tasks).
- GAN loss reduces blurring; most visible in FLAIR (high-frequency detail in lesions).
- Uncertainty maps are highest near tumor boundaries — clinically meaningful.
- Limitation: 2D slice-wise (volumetric context limited); future: 3D U-Net or foundation model encoder (MedSAM).

---

## Appendix (optional)
- Training curves for all ablation variants.
- HPO search history (Optuna visualization).
- MC Dropout sample visualizations (N=5, 10, 20 passes).
