"""Dummy (HAR persistence) model evaluation for multi-horizon log-RV forecasts.

The dummy model predicts realized variance by naive persistence:
    h1  <- log rv5_ss at t
    h5  <- log mean rv5_ss over [t-4, ..., t]
    h22 <- log mean rv5_ss over [t-21, ..., t]

These predictions are pre-computed during preprocessing (PanelData.test_dummy_preds)
and are in the same un-normalized log-RV space as the targets.
"""

import torch
from torch.utils.data import DataLoader

from dl_vol.eval.metrics import qlike_per_horizon


def evaluate_dummy(
    dummy_preds: torch.Tensor,
    test_loader: DataLoader,
) -> list[float]:
    """Per-horizon QLIKE for the persistence dummy model.

    Args:
        dummy_preds: (N, H) tensor of raw log-RV predictions, in the same
                     order as test_loader samples (shuffle=False required).
        test_loader: DataLoader yielding (X, y) batches; only y is used.

    Returns:
        Per-horizon QLIKE as a list of length H.
    """
    targets = []
    for _, yb in test_loader:
        targets.append(yb)
    y = torch.cat(targets, dim=0)  # (N, H)

    if dummy_preds.shape != y.shape:
        raise ValueError(
            f'dummy_preds shape {dummy_preds.shape} != target shape {y.shape}. '
            'Ensure test_loader uses shuffle=False.'
        )

    return qlike_per_horizon(dummy_preds, y).tolist()
