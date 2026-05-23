"""Data preprocessing for the realized-volatility panel.

Two output formats share the same upstream pipeline:

  build_panel      -> PanelData : (X, y) tensors + DataLoaders for the TCN
  build_har_panel  -> HARData   : tidy DataFrames + symbol dummies for OLS

Shared steps:
  1. load Oxford-Man CSV, parse dates, sort by (Symbol, Date)
  2. per-symbol feature engineering (HAR-style log features + jump + return)
  3. per-symbol forward-looking multi-horizon log-RV targets
  4. date-based train/val/test split with a 22-business-day leakage gap

TCN-only steps:
  5. per-symbol rolling z-score (252-day cap) on the log-RV-family columns
  6. concatenate raw log-RV anchor channels (un-normalized) for level anchoring
  7. sliding-window construction into (N, F, L) feature tensors

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

# Base per-day features (one column each, computed in build_features).
BASE_FEATURE_NAMES: tuple[str, ...] = (
    "log_rv5",
    "log_rv5_roll5",
    "log_rv5_roll22",
    "log_rsv",
    "log_jump",
    "ret",
)
# Columns the TCN z-scores per-symbol; `ret` is already on a stable scale.
NORMALIZE_NAMES: tuple[str, ...] = (
    "log_rv5",
    "log_rv5_roll5",
    "log_rv5_roll22",
    "log_rsv",
    "log_jump",
)
# Un-normalized log-RV anchors appended to the TCN feature set so the linear
# head can recover symbol-specific log-RV means without a symbol embedding.
ANCHOR_NAMES: tuple[str, ...] = (
    "raw_log_rv5",
    "raw_log_rv5_roll5",
    "raw_log_rv5_roll22",
)
FEATURE_NAMES: tuple[str, ...] = BASE_FEATURE_NAMES + ANCHOR_NAMES
TARGET_NAMES: tuple[str, ...] = ("h1", "h5", "h22")
HAR_FEATURE_NAMES: tuple[str, ...] = (
    "log_rv5",
    "log_rv5_roll5",
    "log_rv5_roll22",
)

SEQ_LEN = 64
NORM_WINDOW = 252  # rolling normalization cap (trading days)
LEAKAGE_GAP = 22  # days dropped between splits to avoid target leakage


@dataclass
class PanelData:
    """TCN-ready tensors + DataLoaders and panel metadata."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_features: int
    num_horizons: int
    seq_len: int
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    head_weights: tuple[float, ...]
    test_dummy_preds: torch.Tensor  # (N_test, 3) raw log-RV persistence predictions


@dataclass
class HARData:
    """HAR DataFrames ready for scikit-learn / statsmodels OLS.

    X_* columns : HAR_FEATURE_NAMES + one-hot symbol dummies (first symbol
                  dropped as reference category; sklearn's default
                  fit_intercept=True supplies the constant).
    y_* columns : TARGET_NAMES ('h1', 'h5', 'h22') in log-variance space.
    """

    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_val: pd.DataFrame
    y_val: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame


# --- shared helpers -------------------------------------------------------


def _load_raw_panel(csv_path: str | Path) -> pd.DataFrame:
    """Read Oxford-Man CSV; return a tz-naive, (Symbol, Date)-sorted frame."""
    data = pd.read_csv(Path(csv_path), index_col=0)
    data.index.name = "Date"
    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    return data.sort_values(["Symbol", "Date"]).reset_index(drop=True)


def _split_bounds(
    train_end: str,
    val_start: str,
    test_start: str,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(train_end), pd.Timestamp(val_start), pd.Timestamp(test_start)


def build_features(grp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-symbol base features (T, H) and multi-horizon targets (T, H), with F=6, H=3.

    Features use information up to and including day t. `log_*` columns are
    `log(mean(...))`. `log_jump` is `log1p` on a floored jump component.
    Returns are kept on their raw scale. Targets are forward-looking log-RV
    (h=1) and log-mean-RV (h=5, 22).
    """
    rv5 = grp["rv5_ss"]
    feat = pd.DataFrame(
        {
            "log_rv5": np.log(rv5),
            "log_rv5_roll5": np.log(rv5.rolling(5).mean()),
            "log_rv5_roll22": np.log(rv5.rolling(22).mean()),
            "log_rsv": np.log(grp["rsv_ss"]),
            "log_jump": np.log1p(np.maximum(rv5 - grp["bv_ss"], 0.0)),
            "ret": grp["open_to_close"],
        }
    )
    target = pd.DataFrame(
        {
            "h1": np.log(rv5.shift(-1)),
            "h5": np.log(rv5.rolling(5).mean().shift(-5)),
            "h22": np.log(rv5.rolling(22).mean().shift(-22)),
        }
    )
    return feat, target


def rolling_zscore(df: pd.DataFrame, window: int = NORM_WINDOW) -> pd.DataFrame:
    """Per-column rolling z-score capped at `window` rows.

    For t < window statistics use the expanding history [0, t]; from t >=
    window onward it is a true rolling window (`min_periods=1`).
    """
    roll = df.rolling(window=window, min_periods=1)
    sd = roll.std(ddof=0).replace(0.0, np.nan)
    return (df - roll.mean()) / sd


def _split_mask(
    dates: pd.Series,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    test_start: pd.Timestamp,
    gap_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Date-based split with a business-day gap on both sides of each boundary.

    `dates` are forecast-emission dates (last day of a TCN window, or the
    HAR row date). `gap_days` is in business days and matches the maximum
    forecast horizon, so no forward target can bleed across a boundary.
    """
    gap = pd.offsets.BDay(gap_days)
    train = (dates <= (train_end - gap)).to_numpy()
    val = ((dates >= val_start) & (dates <= (test_start - gap))).to_numpy()
    test = (dates >= test_start).to_numpy()
    return train, val, test


# --- TCN pipeline ---------------------------------------------------------


def make_windows(
    feat_arr: np.ndarray,
    target_arr: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Slide a length-`seq_len` window over one symbol's series."""
    T = feat_arr.shape[0]
    if T < seq_len:
        return None, None, None

    idx = np.arange(seq_len)[None, :] + np.arange(T - seq_len + 1)[:, None]
    X = feat_arr[idx].transpose(0, 2, 1)  # (N, F, L)
    y = target_arr[seq_len - 1 :]  # (N, H)
    end_idx = np.arange(seq_len - 1, T)

    keep = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(y).all(axis=1)
    return X[keep], y[keep], end_idx[keep]


def _stack(xs: list[np.ndarray], ys: list[np.ndarray], seq_len: int):
    if not xs:
        return (
            torch.empty(0, len(FEATURE_NAMES), seq_len, dtype=torch.float32),
            torch.empty(0, len(TARGET_NAMES), dtype=torch.float32),
        )
    return (
        torch.from_numpy(np.concatenate(xs, axis=0)).float(),
        torch.from_numpy(np.concatenate(ys, axis=0)).float(),
    )


def _normalize_with_anchors(feat: pd.DataFrame) -> np.ndarray:
    """Return a (T, F) array: z-scored base cols + raw log-RV anchor cols.

    Column order matches FEATURE_NAMES.
    """
    z = feat.copy()
    z[list(NORMALIZE_NAMES)] = rolling_zscore(feat[list(NORMALIZE_NAMES)]).values
    anchors = feat[list(HAR_FEATURE_NAMES)].to_numpy()  # raw log-RV trio
    return np.concatenate([z[list(BASE_FEATURE_NAMES)].to_numpy(), anchors], axis=1)


def build_panel(
    csv_path: str | Path,
    seq_len: int = SEQ_LEN,
    batch_size: int = 128,
    train_end: str = "2013-12-31",
    val_start: str = "2014-01-01",
    test_start: str = "2016-01-01",
    gap_days: int = LEAKAGE_GAP,
    num_workers: int = 0,
) -> PanelData:
    """End-to-end TCN preprocessing returning ready-to-train DataLoaders."""
    data = _load_raw_panel(csv_path)
    train_end_ts, val_start_ts, test_start_ts = _split_bounds(train_end, val_start, test_start)

    X_train_l, y_train_l = [], []
    X_val_l, y_val_l = [], []
    X_test_l, y_test_l = [], []
    dummy_test_l: list[np.ndarray] = []

    for _, grp in data.groupby("Symbol", sort=False):
        grp = grp.sort_values("Date").reset_index(drop=True)
        feat, target = build_features(grp)
        feat_arr = _normalize_with_anchors(feat)

        X_sym, y_sym, end_idx = make_windows(feat_arr, target.values, seq_len)
        if X_sym is None or len(X_sym) == 0:
            continue

        # Raw log-RV anchors at each window's last day, in target-scale.
        # Used as a HAR-style persistence baseline on the test set.
        dummy_sym = feat[list(HAR_FEATURE_NAMES)].to_numpy()[end_idx].astype(np.float32)
        end_dates = grp["Date"].iloc[end_idx].reset_index(drop=True)

        train, val, test = _split_mask(
            end_dates, train_end_ts, val_start_ts, test_start_ts, gap_days
        )

        if train.any():
            X_train_l.append(X_sym[train])
            y_train_l.append(y_sym[train])
        if val.any():
            X_val_l.append(X_sym[val])
            y_val_l.append(y_sym[val])
        if test.any():
            X_test_l.append(X_sym[test])
            y_test_l.append(y_sym[test])
            dummy_test_l.append(dummy_sym[test])

    X_train, y_train = _stack(X_train_l, y_train_l, seq_len)
    X_val, y_val = _stack(X_val_l, y_val_l, seq_len)
    X_test, y_test = _stack(X_test_l, y_test_l, seq_len)
    test_dummy_preds = (
        torch.from_numpy(np.concatenate(dummy_test_l, axis=0))
        if dummy_test_l
        else torch.empty(0, len(HAR_FEATURE_NAMES), dtype=torch.float32)
    )

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
        head_weights=(1.0,) * len(TARGET_NAMES),
        test_dummy_preds=test_dummy_preds,
    )


# --- HAR pipeline ---------------------------------------------------------


def build_har_panel(
    csv_path: str | Path,
    train_end: str = "2013-12-31",
    val_start: str = "2014-01-01",
    test_start: str = "2016-01-01",
    gap_days: int = LEAKAGE_GAP,
) -> HARData:
    """Build train/val/test DataFrames for the HAR linear model.

    Features are raw (un-normalized) log-RV values — standard Corsi (2009)
    specification. Symbol dummies are fitted on the full panel so column
    ordering is identical across all three splits. Same 22-business-day
    leakage gap as `build_panel`.
    """
    data = _load_raw_panel(csv_path)
    train_end_ts, val_start_ts, test_start_ts = _split_bounds(train_end, val_start, test_start)

    rows: list[pd.DataFrame] = []
    for sym, grp in data.groupby("Symbol", sort=False):
        grp = grp.sort_values("Date").reset_index(drop=True)
        feat, target = build_features(grp)
        row = feat[list(HAR_FEATURE_NAMES)].copy()
        row["Symbol"] = sym
        row["Date"] = grp["Date"].values
        rows.append(pd.concat([row, target], axis=1))

    panel = pd.concat(rows, ignore_index=True)
    finite_cols = list(HAR_FEATURE_NAMES) + list(TARGET_NAMES)
    panel = panel[np.isfinite(panel[finite_cols].to_numpy()).all(axis=1)]

    dummies = pd.get_dummies(panel["Symbol"], drop_first=True, dtype=np.float32)
    X_full = pd.concat([panel[list(HAR_FEATURE_NAMES)].astype(np.float32), dummies], axis=1)
    y_full = panel[list(TARGET_NAMES)].astype(np.float32)

    train_mask, val_mask, test_mask = _split_mask(
        panel["Date"], train_end_ts, val_start_ts, test_start_ts, gap_days
    )

    return HARData(
        X_train=X_full[train_mask].reset_index(drop=True),
        y_train=y_full[train_mask].reset_index(drop=True),
        X_val=X_full[val_mask].reset_index(drop=True),
        y_val=y_full[val_mask].reset_index(drop=True),
        X_test=X_full[test_mask].reset_index(drop=True),
        y_test=y_full[test_mask].reset_index(drop=True),
    )
