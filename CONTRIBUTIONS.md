# Contributions & What I Did Differently — For CV / Portfolio

This document summarises **what I implemented and extended** beyond the baseline BraSyn challenge setup, for use in applications (e.g. MBZUAI internship) and CVs.

---

## 1. Problem & Paper

- **Source:** [BraTS 2023 BraSyn](https://arxiv.org/abs/2305.09011) — Brain MR Image Synthesis for Tumor Segmentation (MICCAI 2023).
- **Task:** Given 3 of 4 MRI modalities (T1, T1ce, T2, FLAIR), **synthesise the missing one** so downstream segmentation can still run when clinical data is incomplete.
- **Dataset:** BraTS-GLI (glioma brain MRI, 4 modalities per case, NIfTI).

---

## 2. What I Built From Scratch

| Component | Description |
|-----------|-------------|
| **Data pipeline** | NIfTI loading, case discovery, 2D slice sampling, train/val split, percentile normalisation, BraSyn folder structure. |
| **Model** | 2D U-Net (3 input channels → 1 output channel), configurable depth and dropout. |
| **Training** | PyTorch training loop with L1 + SSIM loss, AdamW, optional cosine/plateau scheduler, checkpointing best model by validation SSIM. |
| **Evaluation** | SSIM, PSNR, MSE (aligned with challenge metrics). |
| **Hyperparameter optimisation** | Optuna over lr, batch size, depth, dropout, loss weight; MedianPruner; best config export. |
| **Inference** | Script to synthesise the missing modality for a case folder (BraSyn output format). |
| **Visualisation** | Slice comparison (input | GT | predicted), training curves (loss, SSIM, PSNR, MSE), Streamlit app for interactive demo. |
| **Documentation** | README (setup, dataset, training, HPO, inference), CONTRIBUTIONS.md (this file), config-driven defaults. |

---

## 3. What I Did Differently / Extra

- **Combined L1 + SSIM loss** so training directly optimises a metric used in the challenge (SSIM).
- **Optuna HPO** for learning rate, network depth, dropout, batch size, and SSIM loss weight (not in the base paper).
- **Config-driven setup** (`config.yaml`) and **training log** (CSV) for reproducibility and curve plotting.
- **Visualisation suite:**  
  - `visualize.py` — single-case slice comparison figure.  
  - `plot_curves.py` — training/validation curves from `train_log.csv`.  
  - `app.py` — Streamlit app: interactive synthesis view + curves + short “About / CV” section.
- **Explicit handling** of Windows/Anaconda (OpenMP env var, `pin_memory` only on CUDA, path handling).

---

## 4. Skills Demonstrated (for CV / interviews)

- **Literature → implementation:** Read the BraSyn paper and challenge, reproduced the task and metrics.
- **Medical imaging:** NIfTI, MRI modalities, BraTS-GLI dataset, train/val protocol.
- **Deep learning:** PyTorch, U-Net, training loop, validation, checkpointing.
- **Optimisation:** Hyperparameter search with Optuna, pruning.
- **Software practice:** Project layout, config files, CLI arguments, logging, README.
- **Communication:** Figures (synthesis comparison, training curves), Streamlit demo, written summary of contributions.

---

## 5. How to Showcase (e.g. MBZUAI)

- **CV / cover letter:** One line such as: *“Implemented BraSyn (BraTS 2023) MRI modality synthesis: U-Net, L1+SSIM loss, Optuna HPO, and visualisation (Streamlit + plots); repo: [link].”*
- **Portfolio / GitHub:** README + CONTRIBUTIONS.md + screenshots or GIFs of the Streamlit app and sample figures (`outputs/synthesis_comparison.png`, `outputs/training_curves.png`).
- **Interview:** Walk through: problem (missing modality) → data (BraTS-GLI) → model (U-Net) → training (loss, metrics) → what you added (HPO, visualisation, logging).

---

## 6. Quick Commands for Figures & Demo

```bash
# After training (checkpoints/best_t2f.pt and checkpoints/train_log.csv exist):
python visualize.py --data_root "path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" --checkpoint checkpoints/best_t2f.pt --target t2f --output outputs/synthesis_comparison.png
python plot_curves.py --log checkpoints/train_log.csv --output outputs/training_curves.png

# Interactive demo (from project root):
streamlit run app.py
```

Then use `outputs/synthesis_comparison.png` and `outputs/training_curves.png` in reports or portfolio; use the Streamlit app live or record a short screen capture for the CV.
