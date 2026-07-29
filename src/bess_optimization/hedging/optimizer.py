"""Bounded variance-minimizing and mean-variance futures hedge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from bess_optimization.config import HedgeConfig
from bess_optimization.valuation.risk import risk_metrics


@dataclass(frozen=True)
class HedgeResult:
    volumes_mw: pd.Series
    payoffs_eur: pd.Series
    unhedged_pnl_eur: pd.Series
    hedged_pnl_eur: pd.Series
    statistics: dict[str, float]
    product_correlations: pd.Series


def optimize_hedge(
    spot_pnl_eur: pd.Series,
    unit_payoffs_eur_per_mw: pd.DataFrame,
    config: HedgeConfig,
    minimum_target_eur: float = 0.0,
    reference_power_mw: float = 1.0,
) -> HedgeResult:
    """Optimize signed MW volumes under bounds; positive means long, negative short."""
    aligned = unit_payoffs_eur_per_mw.reindex(spot_pnl_eur.index)
    if aligned.isna().any().any():
        raise ValueError("P&L scenarios and payoff scenarios do not align")
    y = spot_pnl_eur.to_numpy(dtype=float)
    x = aligned.to_numpy(dtype=float)
    covariance_scale = max(float(np.var(x, axis=0).mean()), 1.0)

    def objective(volumes: np.ndarray) -> float:
        total = y + x @ volumes
        variance = float(np.var(total))
        ridge = config.regularization * covariance_scale * float(volumes @ volumes)
        if config.objective == "mean_variance":
            return -float(np.mean(total)) + config.risk_aversion_lambda * variance + ridge
        return variance + ridge

    result = minimize(
        objective,
        x0=np.zeros(x.shape[1]),
        method="L-BFGS-B",
        bounds=[(config.min_volume_mw, config.max_volume_mw)] * x.shape[1],
    )
    if not result.success:
        raise RuntimeError(f"Hedge optimization failed: {result.message}")
    volumes = result.x
    if config.volume_step_mw:
        volumes = np.round(volumes / config.volume_step_mw) * config.volume_step_mw
        volumes = np.clip(volumes, config.min_volume_mw, config.max_volume_mw)
    hedge_payoff = x @ volumes
    hedged = y + hedge_payoff
    before = risk_metrics(y, minimum_target_eur)
    after = risk_metrics(hedged, minimum_target_eur)
    statistics = {
        **{f"before_{key}": value for key, value in before.items()},
        **{f"after_{key}": value for key, value in after.items()},
        "variance_reduction_pct": _reduction(
            before["variance_pnl_eur2"], after["variance_pnl_eur2"]
        ),
        "std_reduction_pct": _reduction(before["std_pnl_eur"], after["std_pnl_eur"]),
        "gross_hedge_volume_mw": float(np.abs(volumes).sum()),
        "hedge_ratio_abs_mw_per_bess_mw": float(np.abs(volumes).sum()) / reference_power_mw,
    }
    return HedgeResult(
        volumes_mw=pd.Series(volumes, index=aligned.columns, name="volume_mw"),
        payoffs_eur=pd.Series(hedge_payoff, index=spot_pnl_eur.index, name="hedge_pnl_eur"),
        unhedged_pnl_eur=spot_pnl_eur.rename("unhedged_pnl_eur"),
        hedged_pnl_eur=pd.Series(hedged, index=spot_pnl_eur.index, name="hedged_pnl_eur"),
        statistics=statistics,
        product_correlations=aligned.apply(lambda column: spot_pnl_eur.corr(column)).rename(
            "correlation_with_bess_pnl"
        ),
    )


def _reduction(before: float, after: float) -> float:
    return 100 * (1 - after / before) if before > 0 else 0.0
