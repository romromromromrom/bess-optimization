"""Value dispatch decisions consistently across price scenarios."""

from __future__ import annotations

import pandas as pd

from bess_optimization.battery.dispatch import DispatchResult
from bess_optimization.battery.metrics import dispatch_metrics
from bess_optimization.config import BESSConfig
from bess_optimization.scenarios.base import PriceScenarioSet


def value_common_dispatch(
    dispatch: DispatchResult,
    scenarios: PriceScenarioSet,
    config: BESSConfig,
) -> pd.DataFrame:
    """Value one ex-ante schedule on every scenario (no scenario-specific decisions)."""
    rows: list[dict[str, float | str]] = []
    annualization = 8760 / (len(dispatch.schedule) * config.timestep_hours)
    for scenario, prices in scenarios.prices.items():
        metrics = dispatch_metrics(dispatch.schedule, prices, config)
        rows.append(
            {
                **metrics,
                "scenario": str(scenario),
                "annualized_pnl_eur": metrics["net_pnl_eur"] * annualization,
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def value_perfect_foresight(
    dispatches: dict[str, DispatchResult],
    scenarios: PriceScenarioSet,
    config: BESSConfig,
) -> pd.DataFrame:
    """Value each scenario-specific optimized schedule on its own scenario."""
    rows: list[dict[str, float | str]] = []
    for scenario, dispatch in dispatches.items():
        metrics = dispatch_metrics(dispatch.schedule, scenarios.prices[scenario], config)
        rows.append(
            {
                **metrics,
                "scenario": scenario,
                "annualized_pnl_eur": metrics["net_pnl_eur"]
                * 8760
                / (len(dispatch.schedule) * config.timestep_hours),
            }
        )
    return pd.DataFrame(rows).set_index("scenario")
