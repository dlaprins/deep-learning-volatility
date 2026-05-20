"""Self-contained TCN training for multi-horizon log realized variance.

Tensor layouts (channels-first to match nn.Conv1d):
    X : (B, F, L)   = (Batch, Features, Length of window): features for the past L days 
    y : (B, H)      = (Batch, Horizons) log-transformed multi-horizon targets (i.e., H targets)
"""
import torch
from torch.utils.data import DataLoader

from dl_vol.tcn.architecture import TCN, TCNForecast
from dl_vol.eval.metrics import gaussian_nll_per_horizon, qlike_per_horizon, evaluate

def str_formatv(v):
    return '[' + ', '.join(f'{x:.4f}' for x in v) + ']'

def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_features: int,
    num_horizons: int = 3,
    head_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
    hidden_channels=(32, 32, 32, 32),
    kernel_size: int = 3,
    dropout: float = 0.2,
    num_epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
    device: str | None = None,
):
    """Train a TCNForecast model with Gaussian NLL (MSE in log space) training loss.

    Loaders yield (xb, yb) batches:
        xb: (B, F, L) float32  — already normalized features
        yb: (B, H)    float32  — log-transformed targets

    Training loss : Gaussian NLL = MSE in log-variance space (stable, unbiased MLE).
    Early stopping: val QLIKE (Patton-robust, matches backtest metric).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    w = torch.tensor(head_weights, dtype=torch.float32, device=device)  # (H,)

    tcn = TCN(
        num_inputs=num_features,
        num_channels=list(hidden_channels),
        kernel_size=kernel_size,
        dropout=dropout,
    )
    model = TCNForecast(tcn, output_size=num_horizons).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_qlike = float('inf')
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = torch.zeros(num_horizons, device=device)
        train_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            yb_pred = model(xb)                                    # (B, H)
            nll_per_h = gaussian_nll_per_horizon(yb_pred, yb)      # (H,)
            loss = (w * nll_per_h).sum()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                bsz = xb.size(0)
                train_loss_sum += nll_per_h.detach() * bsz
                train_n += bsz

        train_loss = (train_loss_sum / max(train_n, 1)).cpu().tolist()

        val_qlike = evaluate(model, val_loader, device, num_horizons)

        val_qlike_scalar = sum(val_qlike)
        improved = val_qlike_scalar < best_val_qlike
        if improved:
            best_val_qlike = val_qlike_scalar
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f'epoch {epoch + 1:02d}  '
            f'train NLL={str_formatv(train_loss)}  '
            f'val QLIKE={str_formatv(val_qlike)}'
            + ('' if improved else f'  (no improvement {epochs_without_improvement}/{patience})')
        )

        if epochs_without_improvement >= patience:
            print(f'early stopping at epoch {epoch + 1}')
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model