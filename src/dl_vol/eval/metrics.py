"""Evaluation metrics for multi-horizon log-RV forecasts.

All functions operate on log-variance tensors of shape (B, H) and return
per-horizon vectors of shape (H,).
"""

import torch
from torch.utils.data import DataLoader


def gaussian_nll_per_horizon(pred_log: torch.Tensor, target_log: torch.Tensor) -> torch.Tensor:
    """Per-horizon Gaussian NLL (MLE) in log-variance space.

    Assumes log(σ²) | X ~ N(ŷ, τ²). Ignoring the constant log(τ) term,
    minimising this is equivalent to minimising MSE in log-variance space:
        L = (y - ŷ)²
    """
    return ((pred_log - target_log) ** 2).mean(dim=0)


def qlike_per_horizon(pred_log: torch.Tensor, target_log: torch.Tensor) -> torch.Tensor:
    """Per-horizon QLIKE computed from log-space predictions.

    QLIKE(y, y_hat) = y / y_hat - log(y / y_hat) - 1

    in variance space. Using r = exp(target_log - pred_log) avoids exp() on the prediction
    alone and is numerically friendlier early in training.
    """
    diff = (target_log - pred_log).clamp(-15.0, 15.0)
    return (diff.exp() - diff - 1.0).mean(dim=0)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    num_horizons: int,
) -> list[float]:
    """Run `model` on `loader` and return per-horizon QLIKE."""
    model.eval()
    qlike_sum = torch.zeros(num_horizons, device=device)
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        yb_pred = model(xb)
        bsz = xb.size(0)
        qlike_sum += qlike_per_horizon(yb_pred, yb) * bsz
        n += bsz
    if n == 0:
        return [0.0] * num_horizons
    return (qlike_sum / n).cpu().tolist()
