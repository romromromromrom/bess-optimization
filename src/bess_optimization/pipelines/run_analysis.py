"""Synthetic or user-data end-to-end analysis pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bess_optimization.battery.dispatch import (
    DispatchResult,
    optimize_common_dispatch,
    optimize_perfect_foresight,
)
from bess_optimization.config import AnalysisConfig
from bess_optimization.curves.forward_curve import build_hourly_forward_curve
from bess_optimization.data.synthetic_data import (
    synthetic_forward_quotes,
    synthetic_spot_prices,
)
from bess_optimization.hedging.optimizer import HedgeResult, optimize_hedge
from bess_optimization.hedging.payoffs import build_payoff_matrix
from bess_optimization.hedging.products import products_from_quotes
from bess_optimization.scenarios.base import PriceScenarioSet, ScenarioGenerator
from bess_optimization.scenarios.bootstrap import ConditionalBootstrapGenerator
from bess_optimization.scenarios.diagnostics import scenario_diagnostics
from bess_optimization.scenarios.factor_model import FactorScenarioGenerator
from bess_optimization.valuation.pnl import value_common_dispatch, value_perfect_foresight
from bess_optimization.valuation.risk import risk_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisResult:
    config: AnalysisConfig
    history: pd.Series
    forward_quotes: pd.DataFrame
    hourly_forward: pd.Series
    scenarios: PriceScenarioSet
    scenario_diagnostics: dict[str, pd.DataFrame | pd.Series | float]
    representative_dispatch: DispatchResult
    pnl_by_scenario: pd.DataFrame
    risk: dict[str, float]
    hedge: HedgeResult | None


def run_analysis(
    config: AnalysisConfig,
    history: pd.Series | None = None,
    forward_quotes: pd.DataFrame | None = None,
    export: bool = True,
) -> AnalysisResult:
    """Run all stages. Defaults are explicitly synthetic and immediately runnable."""
    hist = (
        history
        if history is not None
        else synthetic_spot_prices(
            config.historical_start,
            config.historical_end,
            config.timezone,
            config.scenarios.seed,
        )
    )
    target_index = pd.date_range(config.start, config.end, freq="h", tz=config.timezone)
    years = sorted(set(target_index.year))
    quotes = (
        forward_quotes
        if forward_quotes is not None
        else synthetic_forward_quotes(
            years, timezone=config.timezone, market=config.market, seed=config.scenarios.seed
        )
    )
    hourly_forward = build_hourly_forward_curve(quotes, target_index, hist)
    logger.info("Built hourly forward curve with %d periods", len(hourly_forward))
    if config.scenarios.method == "bootstrap":
        generator: ScenarioGenerator = ConditionalBootstrapGenerator(config.scenarios.block_days)
    else:
        generator = FactorScenarioGenerator().fit(quotes)
    scenarios = generator.generate(
        hist, hourly_forward, config.scenarios.n_scenarios, config.scenarios.seed
    )
    diagnostics = scenario_diagnostics(scenarios, hourly_forward)
    logger.info("Generated %d scenarios using %s", scenarios.n_scenarios, scenarios.method)
    if config.dispatch_mode == "expected":
        representative = optimize_common_dispatch(
            scenarios, config.battery, config.dispatch_quantile
        )
        pnl = value_common_dispatch(representative, scenarios, config.battery)
    else:
        dispatches = optimize_perfect_foresight(scenarios, config.battery)
        representative = next(iter(dispatches.values()))
        pnl = value_perfect_foresight(dispatches, scenarios, config.battery)
    risk = risk_metrics(pnl["net_pnl_eur"], config.minimum_revenue_target_eur)
    hedge = None
    if config.hedge.enabled:
        products = products_from_quotes(quotes, config.hedge.products)
        payoffs = build_payoff_matrix(scenarios, products, config.battery.timestep_hours)
        hedge = optimize_hedge(
            pnl["net_pnl_eur"],
            payoffs,
            config.hedge,
            config.minimum_revenue_target_eur,
            config.battery.max_discharge_mw,
        )
    result = AnalysisResult(
        config,
        hist,
        quotes,
        hourly_forward,
        scenarios,
        diagnostics,
        representative,
        pnl,
        risk,
        hedge,
    )
    if export:
        export_analysis(result, config.output_dir)
    return result


def export_analysis(result: AnalysisResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.history.to_frame().to_parquet(output / "synthetic_spot_history.parquet")
    result.forward_quotes.to_csv(output / "forward_quotes.csv", index=False)
    result.hourly_forward.to_frame().to_parquet(output / "hourly_forward.parquet")
    result.scenarios.to_parquet(output / "price_scenarios.parquet")
    scalar_diagnostics: dict[str, float] = {}
    for name, diagnostic in result.scenario_diagnostics.items():
        if isinstance(diagnostic, (pd.DataFrame, pd.Series)):
            diagnostic.to_csv(output / f"diagnostic_{name}.csv")
        else:
            scalar_diagnostics[name] = diagnostic
    (output / "scenario_diagnostics.json").write_text(
        json.dumps(scalar_diagnostics, indent=2), encoding="utf-8"
    )
    result.representative_dispatch.schedule.to_csv(output / "dispatch.csv")
    result.pnl_by_scenario.to_csv(output / "pnl_by_scenario.csv")
    (output / "risk_metrics.json").write_text(json.dumps(result.risk, indent=2), encoding="utf-8")
    if result.hedge is not None:
        result.hedge.volumes_mw.to_csv(output / "hedge_volumes.csv")
        pd.concat(
            [
                result.hedge.unhedged_pnl_eur,
                result.hedge.payoffs_eur,
                result.hedge.hedged_pnl_eur,
            ],
            axis=1,
        ).to_csv(output / "hedged_pnl.csv")
        (output / "hedge_statistics.json").write_text(
            json.dumps(result.hedge.statistics, indent=2), encoding="utf-8"
        )
