"""Convert block forward products into an assumed hourly central curve."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bess_optimization.curves.hourly_shaping import is_peak, normalized_hourly_shape


def latest_snapshot(quotes: pd.DataFrame) -> pd.DataFrame:
    """Select the latest observation available independently for every product."""
    ordered = quotes.sort_values("observation_date")
    keys = ["delivery_start", "delivery_end", "load_type", "market"]
    return ordered.groupby(keys, as_index=False).tail(1).reset_index(drop=True)


def build_hourly_forward_curve(
    quotes: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    history: pd.Series,
) -> pd.Series:
    """Shape Calendar blocks to hours while matching quoted base and peak averages.

    The historical multiplicative shape is first applied, then affine corrections are
    made separately to peak and off-peak hours. If both products exist, the implied
    off-peak level is computed so that the all-hours mean equals Baseload.
    """
    result = pd.Series(np.nan, index=target_index, name="forward_eur_mwh")
    snapshot = latest_snapshot(quotes)
    shape = normalized_hourly_shape(history, target_index)
    for year in sorted(set(target_index.year)):
        year_mask = target_index.year == year
        idx = target_index[year_mask]
        rows = snapshot[snapshot["delivery_start"].dt.year == year]
        if rows.empty:
            continue
        prices = rows.groupby("load_type")["price_eur_mwh"].mean()
        base = float(prices.get("baseload", np.nan))
        peak_quote = float(prices.get("peakload", np.nan))
        peak_mask = is_peak(idx)
        n_peak, n_all = int(peak_mask.sum()), len(idx)
        n_off = n_all - n_peak
        if np.isnan(base):
            base = peak_quote
        if np.isnan(peak_quote):
            peak_quote = base
        off_quote = (base * n_all - peak_quote * n_peak) / n_off if n_off else base
        values = shape.loc[idx].to_numpy().copy() * base
        for mask, target in ((peak_mask, peak_quote), (~peak_mask, off_quote)):
            if mask.any():
                values[mask] += target - float(values[mask].mean())
        result.loc[idx] = values
    if result.isna().any():
        missing_index = pd.DatetimeIndex(result[result.isna()].index)
        missing_years = sorted(set(missing_index.year))
        raise ValueError(f"No forward quotes cover target years: {missing_years}")
    return result
