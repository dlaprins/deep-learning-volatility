"""Data preprocessing for the realized-volatility panel.

Contains:
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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


FEATURE_NAMES: tuple[str, ...] = (
    # z-normalized (rolling 252-day): capture deviations from recent mean
    'log_rv5', 'log_rv5_roll5', 'log_rv5_roll22',
    'log_rsv', 'log_jump', 'ret',
    # raw (un-normalized): anchor absolute level so the output head can
    # predict symbol-specific log-RV means without a symbol embedding
    'raw_log_rv5', 'raw_log_rv5_roll5', 'raw_log_rv5_roll22',
)
TARGET_NAMES: tuple[str, ...] = ('h1', 'h5', 'h22')

SEQ_LEN = 64
NORM_WINDOW = 252            # rolling normalization cap (trading days)
LEAKAGE_GAP = 22             # days dropped between splits to avoid target leakage
LOG_FEATURE_MASK = (True, True, True, True, True, False, False, False, False)  # ret and raw_* stay un-normalized


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
    test_dummy_preds: torch.Tensor   # (N_test, 3) raw log-RV persistence predictions


# --- feature / target engineering -----------------------------------------

def build_features(grp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (T, F) features and (T, H) targets for one symbol.

    Features are computed with information up to and including day t.
    `log_*` features are computed AFTER taking the relevant rolling mean
    (i.e. log of mean). `log_jump` uses log1p on a floored
    jump component. Returns are kept in raw space.

    Targets are forward-looking log-RV and log-mean-RV.
    """
    log_rv5        = np.log(grp['rv5_ss'])
    log_rv5_roll5  = np.log(grp['rv5_ss'].rolling(5).mean())
    log_rv5_roll22 = np.log(grp['rv5_ss'].rolling(22).mean())
    log_rsv        = np.log(grp['rsv_ss'])
    log_jump       = np.log1p(np.maximum(grp['rv5_ss'] - grp['bv_ss'], 0.0))
    ret            = grp['open_to_close']

    target_h1  = np.log(grp['rv5_ss'].shift(-1))
    target_h5  = np.log(grp['rv5_ss'].rolling(5).mean().shift(-5))
    target_h22 = np.log(grp['rv5_ss'].rolling(22).mean().shift(-22))

    feat = pd.concat(
        [log_rv5, log_rv5_roll5, log_rv5_roll22, log_rsv, log_jump, ret,
         log_rv5, log_rv5_roll5, log_rv5_roll22],  # raw repeats (not z-scored)
        axis=1,
    )
    feat.columns = list(FEATURE_NAMES)

    target = pd.concat([target_h1, target_h5, target_h22], axis=1)
    target.columns = list(TARGET_NAMES)
    return feat, target


# --- normalization --------------------------------------------------------

def rolling_zscore(df: pd.DataFrame, window: int = NORM_WINDOW) -> pd.DataFrame:
    """Per-column rolling z-score with a 252-day cap.

    For t < window the statistics use the full available expanding history
    [0, t]; for t >= window they use only the past `window` rows.
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
    target_arr: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Slide a length-`seq_len` window over one symbol's series."""
    T = feat_arr.shape[0]
    if T < seq_len:
        return None, None, None

    idx = np.arange(seq_len)[None, :] + np.arange(T - seq_len + 1)[:, None]  # (N, L)
    X = feat_arr[idx]                  # (N, L, F)
    X = X.transpose(0, 2, 1)           # (N, F, L)
    y = target_arr[seq_len - 1:]          # (N, H)
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
    val_start: pd.Timestamp,
    test_start: pd.Timestamp,
    gap_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Date-based split with a business-day gap buffer on both sides of each boundary.

    `end_dates` are the dates of the LAST day of each window (the day on
    which the forecast is emitted). `gap_days` is in business days, matching
    the maximum forecast horizon, so no forward target can bleed across a
    split boundary.

    Boundaries:
      train/val : last train window <= train_end - gap
                  first val window  >= val_start  (= train_end + 1bd)
      val/test  : last val window   <= test_start - gap
                  first test window >= test_start
    """
    gap = pd.offsets.BDay(gap_days)
    train = (end_dates <= (train_end - gap)).to_numpy()
    val   = ((end_dates >= val_start) & (end_dates <= (test_start - gap))).to_numpy()
    test  = (end_dates >= test_start).to_numpy()
    return train, val, test


def build_panel(
    csv_path: str | Path,
    seq_len: int = SEQ_LEN,
    batch_size: int = 128,
    train_end: str = '2013-12-31',
    val_start: str = '2014-01-01',
    test_start: str = '2016-01-01',
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

    train_end_ts  = pd.Timestamp(train_end)
    val_start_ts  = pd.Timestamp(val_start)
    test_start_ts = pd.Timestamp(test_start)

    log_mask = np.array(LOG_FEATURE_MASK, dtype=bool)

    X_train_l, y_train_l = [], []
    X_val_l, y_val_l = [], []
    X_test_l, y_test_l = [], []
    dummy_test_l: list[np.ndarray] = []

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
        # Raw log-RV at last time step: used as dummy persistence predictions.
        # feat.values[end_idx, :3] = [log_rv5, log_rv5_roll5, log_rv5_roll22] at time t,
        # in the same (un-normalized) space as the targets.
        dummy_sym = feat.values[end_idx, :3].astype(np.float32)
        end_dates = grp['Date'].iloc[end_idx].reset_index(drop=True)

        tr, va, te = _split_mask(
            end_dates, train_end_ts, val_start_ts, test_start_ts, gap_days
        )

        if tr.any():
            X_train_l.append(X_sym[tr]); y_train_l.append(y_sym[tr])
        if va.any():
            X_val_l.append(X_sym[va]); y_val_l.append(y_sym[va])
        if te.any():
            X_test_l.append(X_sym[te]); y_test_l.append(y_sym[te])
            dummy_test_l.append(dummy_sym[te])

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

    X_train, y_train = _stack(X_train_l, y_train_l)
    X_val, y_val = _stack(X_val_l, y_val_l)
    X_test, y_test = _stack(X_test_l, y_test_l)

    test_dummy_preds = (
        torch.from_numpy(np.concatenate(dummy_test_l, axis=0))
        if dummy_test_l
        else torch.empty(0, 3, dtype=torch.float32)
    )

    head_weights = (1.0,) * len(TARGET_NAMES)

    def _loader(X, y, shuffle):
        return DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )

    return PanelData(
        train_loader=_loader(X_train, y_train, shuffle=True),
        val_loader=_loader(X_val, y_val, shuffle=False),
        test_loader=_loader(X_test, y_test, shuffle=False),
        num_features=len(FEATURE_NAMES),
        num_horizons=len(TARGET_NAMES),
        seq_len=seq_len,
        feature_names=FEATURE_NAMES,
        target_names=TARGET_NAMES,
        head_weights=head_weights,
        test_dummy_preds=test_dummy_preds,
    )
