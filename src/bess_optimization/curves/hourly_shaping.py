"""Historical hourly-shape estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def is_peak(index: pd.DatetimeIndex) -> np.ndarray:
    """Continental European Peakload: weekdays, hours [08:00, 20:00)."""
    return np.asarray((index.dayofweek < 5) & (index.hour >= 8) & (index.hour < 20))


def normalized_hourly_shape(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Estimate multiplicative month/day-type/hour shapes with robust fallbacks."""
    frame = history.rename("price").to_frame()
    history_index = pd.DatetimeIndex(frame.index)
    frame["month"] = history_index.month
    frame["day_type"] = np.where(history_index.dayofweek < 5, "weekday", "weekend")
    frame["hour"] = history_index.hour
    overall = float(frame["price"].mean())
    profile = frame.groupby(["month", "day_type", "hour"])["price"].mean() / overall
    keys = pd.MultiIndex.from_arrays(
        [
            target_index.month,
            np.where(target_index.dayofweek < 5, "weekday", "weekend"),
            target_index.hour,
        ],
        names=["month", "day_type", "hour"],
    )
    values = profile.reindex(keys).fillna(1.0).to_numpy()
    return pd.Series(values, index=target_index, name="shape")


def historical_residuals(history: pd.Series) -> pd.Series:
    """Additive residuals around a month/day-type/hour conditional mean."""
    index = pd.DatetimeIndex(history.index)
    keys = [index.month, index.dayofweek < 5, index.hour]
    expected = history.groupby(keys).transform("mean")
    return (history - expected).rename("residual_eur_mwh")
