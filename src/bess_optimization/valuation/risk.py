"""Downside and distribution risk indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def risk_metrics(pnl: pd.Series | np.ndarray, minimum_target_eur: float = 0.0) -> dict[str, float]:
    """Compute P&L statistics.

    VaR and CVaR are reported as positive losses: ``VaR95 = -P5(P&L)`` and
    ``CVaR95 = -E[P&L | P&L <= P5]``. They can be negative if even the lower tail is
    profitable.
    """
    values = np.asarray(pnl, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("P&L values must be finite and non-empty")
    p5 = float(np.quantile(values, 0.05))
    tail = values[values <= p5]
    return {
        "expected_pnl_eur": float(values.mean()),
        "std_pnl_eur": float(values.std(ddof=0)),
        "variance_pnl_eur2": float(values.var(ddof=0)),
        "p5_eur": p5,
        "p10_eur": float(np.quantile(values, 0.10)),
        "p50_eur": float(np.quantile(values, 0.50)),
        "p90_eur": float(np.quantile(values, 0.90)),
        "p95_eur": float(np.quantile(values, 0.95)),
        "minimum_eur": float(values.min()),
        "maximum_eur": float(values.max()),
        "var_95_eur": -p5,
        "cvar_95_eur": -float(tail.mean()),
        "probability_below_zero": float(np.mean(values < 0)),
        "probability_below_target": float(np.mean(values < minimum_target_eur)),
        "minimum_target_eur": float(minimum_target_eur),
    }
