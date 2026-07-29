"""Statistical controls for generated hourly scenarios."""

from __future__ import annotations

import pandas as pd

from bess_optimization.curves.hourly_shaping import is_peak
from bess_optimization.scenarios.base import PriceScenarioSet


def scenario_diagnostics(
    scenarios: PriceScenarioSet, hourly_forward: pd.Series
) -> dict[str, pd.DataFrame | pd.Series | float]:
    prices = scenarios.prices
    flat = prices.stack(future_stack=True)
    summary = flat.describe(percentiles=[0.05, 0.1, 0.5, 0.9, 0.95])
    index = pd.DatetimeIndex(prices.index)
    by_hour = prices.groupby(index.hour).agg(["mean", "std"])
    scenario_means = prices.mean()
    scenario_volatility = prices.std()
    autocorrelation = prices.apply(lambda series: series.autocorr(lag=1))
    mean_error = prices.mean(axis=1) - hourly_forward
    peak = is_peak(index)
    spreads = prices.loc[peak].mean() - prices.loc[~peak].mean()
    forward_spread = float(hourly_forward.loc[peak].mean() - hourly_forward.loc[~peak].mean())
    return {
        "distribution": summary,
        "scenario_means": scenario_means,
        "scenario_volatility": scenario_volatility,
        "lag1_autocorrelation": autocorrelation,
        "hourly_profile": by_hour,
        "mean_forward_mae_eur_mwh": float(mean_error.abs().mean()),
        "peak_offpeak_spread": spreads,
        "forward_peak_offpeak_spread": forward_spread,
    }
