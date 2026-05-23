import numpy as np
from sklearn.linear_model import LinearRegression

from dl_vol.preprocessing.preprocess import TARGET_NAMES, HARData


def train_har(
    har_data: HARData,
    target_names: tuple[str, ...] = TARGET_NAMES,
) -> dict[str, LinearRegression]:
    """Train one OLS model per horizon on the HAR training split."""
    return {h: LinearRegression().fit(har_data.X_train, har_data.y_train[h]) for h in target_names}


def backtest_har(
    models: dict[str, LinearRegression],
    har_data: HARData,
    target_names: tuple[str, ...] = TARGET_NAMES,
) -> list[float]:
    """Per-horizon QLIKE for the HAR models on the test split."""
    qlike = []
    for h in target_names:
        pred = models[h].predict(har_data.X_test)  # (N,)
        target = har_data.y_test[h].to_numpy()  # (N,)
        diff = np.clip(target - pred, -15.0, 15.0)
        qlike.append(float((np.exp(diff) - diff - 1.0).mean()))
    return qlike
