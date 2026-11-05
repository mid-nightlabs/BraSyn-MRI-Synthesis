"""
Evaluation metrics for BraSyn synthesis (aligned with challenge evaluation).
SSIM, PSNR, MSE as in arXiv:2305.09011 and official evaluation_utils.
"""

import numpy as np
import torch


def ssim_2d(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Structural Similarity (2D). pred/target: (B,1,H,W)."""
    from torchmetrics.functional import structural_similarity_index_measure
    return structural_similarity_index_measure(pred, target, data_range=data_range)


def psnr_2d(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio (2D)."""
    mse = torch.nn.functional.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(100.0, device=pred.device)  # High arbitrary upper limit instead of inf
    return 10.0 * torch.log10((data_range ** 2) / mse)


def mse_2d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error (2D)."""
    return torch.nn.functional.mse_loss(pred, target)


def compute_batch_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> dict:
    """Compute SSIM, PSNR, MSE over a batch. Tensors (B,1,H,W)."""
    pred = pred.clamp(0.0, data_range)
    target = target.clamp(0.0, data_range)
    ssim = ssim_2d(pred, target, data_range=data_range).item()
    psnr = psnr_2d(pred, target, data_range=data_range).item()
    mse = mse_2d(pred, target).item()
    return {"ssim": ssim, "psnr": psnr, "mse": mse}
