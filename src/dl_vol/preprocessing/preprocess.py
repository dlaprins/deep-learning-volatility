"""Data preprocessing for the realized-volatility panel.

Owns:
  - feature engineering (HAR-style log features + jump + return signals),
  - forward-looking multi-horizon log-RV targets,
  - per-symbol rolling z-score normalization (252-day cap),
  - date-based train/val/test split with a 22-day gap to prevent target leakage,
  - sliding-window construction into (X, y) tensors suitable for a causal TCN,
  - DataLoader assembly.

Tensor layouts (channels-first to match nn.Conv1d):
    X : (N, F, L)   feature windows
    y : (N, H)      multi-horizon log-RV targets aligned with the LAST day of X
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


FEATURE_NAMES: tuple[str, ...] = (
    'log_rv5', 'log_rv5_roll5', 'log_rv5_roll22',
    'log_rsv', 'log_jump', 'ret', 'ret2',
)
TARGET_NAMES: tuple[str, ...] = ('h1', 'h5', 'h22')

SEQ_LEN = 64
NORM_WINDOW = 252            # rolling normalization cap (trading days)
LEAKAGE_GAP = 22             # days dropped between splits to avoid target leakage
LOG_FEATURE_MASK = (True, True, True, True, True, False, False)  # ret, ret2 stay raw


@dataclass
class PanelData:
    """Container for prepared tensors and metadata."""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_features: int
    num_horizons: int
    seq_len: int
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    head_weights: tuple[float, ...]
    train_target_std: tuple[float, ...] = field(default_factory=tuple)


# --- feature / target engineering -----------------------------------------

def build_features(grp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (T, F) features and (T, H) targets for one symbol.

    Features are computed with information up to and including day t.
    `log_*` features are computed AFTER taking the relevant rolling mean
    (i.e. log of mean, per design.md). `log_jump` uses log1p on a floored
    jump component. Returns and squared returns are kept in raw space.

    Targets are forward-looking log-RV.
    """
    log_rv5        = np.log(grp['rv5_ss'])
    log_rv5_roll5  = np.log(grp['rv5_ss'].rolling(5).mean())
    log_rv5_roll22 = np.log(grp['rv5_ss'].rolling(22).mean())
    log_rsv        = np.log(grp['rsv_ss'])
    log_jump       = np.log1p(np.maximum(grp['rv5_ss'] - grp['bv_ss'], 0.0))
    ret            = grp['open_to_close']
    ret2           = grp['open_to_close'] ** 2

    target_h1  = np.log(grp['rv5_ss'].shift(-1))
    target_h5  = np.log(grp['rv5_ss'].rolling(5).mean().shift(-5))
    target_h22 = np.log(grp['rv5_ss'].rolling(22).mean().shift(-22))

    feat = pd.concat(
        [log_rv5, log_rv5_roll5, log_rv5_roll22, log_rsv, log_jump, ret, ret2],
        axis=1,
    )
    feat.columns = list(FEATURE_NAMES)

    tgt = pd.concat([target_h1, target_h5, target_h22], axis=1)
    tgt.columns = list(TARGET_NAMES)
    return feat, tgt


# --- normalization --------------------------------------------------------

def rolling_zscore(df: pd.DataFrame, window: int = NORM_WINDOW) -> pd.DataFrame:
    """Per-column rolling z-score with a 252-day cap.

    For t < window the statistics use the full available expanding history
    [0, t]; for t >= window they use only the past `window` rows. This
    matches design.md: "normalize with all data before t=252, afterwards
    use only the past 252 days".
    """
    # `min_periods=1` makes the initial part expanding; once `window` rows
    # are available `pandas.rolling` becomes a true 252-day window.
    roll = df.rolling(window=window, min_periods=1)
    mu = roll.mean()
    sd = roll.std(ddof=0).replace(0.0, np.nan)
    out = (df - mu) / sd
    return out


# --- windowing ------------------------------------------------------------

def make_windows(
    feat_arr: np.ndarray,
    tgt_arr: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Slide a length-`seq_len` window over one symbol's series."""
    T = feat_arr.shape[0]
    if T < seq_len:
        return None, None, None

    idx = np.arange(seq_len)[None, :] + np.arange(T - seq_len + 1)[:, None]  # (N, L)
    X = feat_arr[idx]                  # (N, L, F)
    X = X.transpose(0, 2, 1)           # (N, F, L)
    y = tgt_arr[seq_len - 1:]          # (N, H)
    # window_end_idx[i] = i + L - 1 is the absolute time index of the last
    # day in window i (used later for date-based splitting).
    window_end_idx = np.arange(seq_len - 1, T)

    feat_ok = np.isfinite(X).all(axis=(1, 2))
    tgt_ok  = np.isfinite(y).all(axis=1)
    keep = feat_ok & tgt_ok
    return X[keep], y[keep], window_end_idx[keep]


# --- top-level pipeline ---------------------------------------------------

def _split_mask(
    end_dates: pd.Series,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp, val_end: pd.Timestamp,
    test_start: pd.Timestamp,
    gap_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Date-based split with a `gap_days` buffer between splits.

    `end_dates` are the dates of the LAST day of each window (the day on
    which the forecast is emitted). The buffer is applied on both sides of
    the gap, so e.g. windows whose last day is in `[train_end - gap, train_end]`
    are dropped from training to ensure their forward targets do not bleed
    into the validation period.
    """
    gap = pd.Timedelta(days=gap_days)
    train = (end_dates <= (train_end - gap)).to_numpy()
    val   = ((end_dates >= val_start) & (end_dates <= (val_end - gap))).to_numpy()
    test  = (end_dates >= test_start).to_numpy()
    return train, val, test


def build_panel(
    csv_path: str | Path,
    seq_len: int = SEQ_LEN,
    batch_size: int = 128,
    train_end: str = '2015-12-31',
    val_start: str = '2016-01-01',
    val_end: str = '2017-12-31',
    test_start: str = '2018-01-01',
    gap_days: int = LEAKAGE_GAP,
    num_workers: int = 0,
) -> PanelData:
    """End-to-end preprocessing returning ready-to-train DataLoaders."""
    csv_path = Path(csv_path)
    data = pd.read_csv(csv_path, index_col=0)
    data.index.name = 'Date'
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'], utc=True).dt.tz_localize(None).dt.normalize()
    data = data.sort_values(['Symbol', 'Date']).reset_index(drop=True)

    train_end_ts = pd.Timestamp(train_end)
    val_start_ts = pd.Timestamp(val_start)
    val_end_ts   = pd.Timestamp(val_end)
    test_start_ts = pd.Timestamp(test_start)

    log_mask = np.array(LOG_FEATURE_MASK, dtype=bool)

    X_tr_l, y_tr_l = [], []
    X_va_l, y_va_l = [], []
    X_te_l, y_te_l = [], []

    for sym, grp in data.groupby('Symbol', sort=False):
        grp = grp.sort_values('Date').reset_index(drop=True)
        feat, tgt = build_features(grp)

        # Per-symbol rolling z-score. Only the columns flagged as log-space
        # (HAR/log-RV-family) are z-scored; returns and squared returns are
        # already on a stable scale and would be distorted by z-scoring.
        feat_norm = feat.copy()
        feat_norm.loc[:, log_mask] = rolling_zscore(
            feat.loc[:, log_mask], window=NORM_WINDOW
        ).values

        X_sym, y_sym, end_idx = make_windows(feat_norm.values, tgt.values, seq_len)
        if X_sym is None or len(X_sym) == 0:
            continue
        end_dates = grp['Date'].iloc[end_idx].reset_index(drop=True)

        tr, va, te = _split_mask(
            end_dates, train_end_ts, val_start_ts, val_end_ts, test_start_ts, gap_days
        )

        if tr.any():
            X_tr_l.append(X_sym[tr]); y_tr_l.append(y_sym[tr])
        if va.any():
            X_va_l.append(X_sym[va]); y_va_l.append(y_sym[va])
        if te.any():
            X_te_l.append(X_sym[te]); y_te_l.append(y_sym[te])

    def _stack(xs, ys):
        if not xs:
            F = len(FEATURE_NAMES); H = len(TARGET_NAMES)
            return (
                torch.empty(0, F, seq_len, dtype=torch.float32),
                torch.empty(0, H, dtype=torch.float32),
            )
        X = torch.from_numpy(np.concatenate(xs, axis=0)).float()
        y = torch.from_numpy(np.concatenate(ys, axis=0)).float()
        return X, y

    X_tr, y_tr = _stack(X_tr_l, y_tr_l)
    X_va, y_va = _stack(X_va_l, y_va_l)
    X_te, y_te = _stack(X_te_l, y_te_l)

    # Head weights = 1 / Var(target) on the training set, so each horizon
    # contributes comparably to the multi-task loss.
    if len(y_tr) > 0:
        var = y_tr.var(dim=0, unbiased=False).clamp_min(1e-8)
        head_weights = tuple((1.0 / var).tolist())
        train_target_std = tuple(y_tr.std(dim=0, unbiased=False).tolist())
    else:
        head_weights = (1.0,) * len(TARGET_NAMES)
        train_target_std = (1.0,) * len(TARGET_NAMES)

    def _loader(X, y, shuffle):
        return DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )

    return PanelData(
        train_loader=_loader(X_tr, y_tr, shuffle=True),
        val_loader=_loader(X_va, y_va, shuffle=False),
        test_loader=_loader(X_te, y_te, shuffle=False),
        num_features=len(FEATURE_NAMES),
        num_horizons=len(TARGET_NAMES),
        seq_len=seq_len,
        feature_names=FEATURE_NAMES,
        target_names=TARGET_NAMES,
        head_weights=head_weights,
        train_target_std=train_target_std,
    )
