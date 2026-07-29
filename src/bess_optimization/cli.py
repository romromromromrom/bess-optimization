"""Command-line interface for the research workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from bess_optimization.battery.dispatch import optimize_common_dispatch
from bess_optimization.config import AnalysisConfig
from bess_optimization.curves.forward_curve import build_hourly_forward_curve
from bess_optimization.data.entsoe_client import EntsoeSpotClient
from bess_optimization.data.synthetic_data import (
    synthetic_forward_quotes,
    synthetic_spot_prices,
)
from bess_optimization.hedging.optimizer import optimize_hedge
from bess_optimization.hedging.payoffs import build_payoff_matrix
from bess_optimization.hedging.products import products_from_quotes
from bess_optimization.pipelines.run_analysis import run_analysis
from bess_optimization.scenarios.base import PriceScenarioSet
from bess_optimization.scenarios.bootstrap import ConditionalBootstrapGenerator
from bess_optimization.valuation.pnl import value_common_dispatch

app = typer.Typer(help="BESS spot revenue and Calendar futures hedge research toolkit.")


@app.command("generate-data")
def generate_data(config: Path = Path("configs/example.yaml")) -> None:
    cfg = AnalysisConfig.from_yaml(config)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = synthetic_spot_prices(
        cfg.historical_start, cfg.historical_end, cfg.timezone, cfg.scenarios.seed
    )
    years = sorted(set(pd.date_range(cfg.start, cfg.end, freq="h", tz=cfg.timezone).year))
    history.to_frame().to_parquet(output / "synthetic_spot_history.parquet")
    synthetic_forward_quotes(years, timezone=cfg.timezone).to_csv(
        output / "synthetic_forward_quotes.csv", index=False
    )
    typer.echo(f"Synthetic data written to {output}")


@app.command("fetch-entsoe")
def fetch_entsoe(
    start: str,
    end: str,
    zone: str = "FR",
    output: Path = Path("data/raw/entsoe_spot.parquet"),
) -> None:
    prices = EntsoeSpotClient().fetch_day_ahead_prices(zone, start, end)
    output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_frame().to_parquet(output)
    typer.echo(f"ENTSO-E prices written to {output}")


@app.command("generate-scenarios")
def generate_scenarios(config: Path = Path("configs/example.yaml")) -> None:
    cfg = AnalysisConfig.from_yaml(config)
    history = synthetic_spot_prices(
        cfg.historical_start, cfg.historical_end, cfg.timezone, cfg.scenarios.seed
    )
    index = pd.date_range(cfg.start, cfg.end, freq="h", tz=cfg.timezone)
    quotes = synthetic_forward_quotes(sorted(set(index.year)), timezone=cfg.timezone)
    forward = build_hourly_forward_curve(quotes, index, history)
    scenarios = ConditionalBootstrapGenerator(cfg.scenarios.block_days).generate(
        history, forward, cfg.scenarios.n_scenarios, cfg.scenarios.seed
    )
    output = Path(cfg.output_dir) / "price_scenarios.parquet"
    scenarios.to_parquet(output)
    typer.echo(f"{scenarios.n_scenarios} scenarios written to {output}")


@app.command("optimize-dispatch")
def optimize_dispatch(config: Path = Path("configs/example.yaml")) -> None:
    cfg = AnalysisConfig.from_yaml(config)
    scenarios = PriceScenarioSet.from_parquet(Path(cfg.output_dir) / "price_scenarios.parquet")
    result = optimize_common_dispatch(scenarios, cfg.battery, cfg.dispatch_quantile)
    output = Path(cfg.output_dir) / "dispatch.csv"
    result.schedule.to_csv(output)
    net_pnl = result.metrics["net_pnl_eur"]
    typer.echo(f"Dispatch written to {output}; net optimization P&L={net_pnl:.2f} EUR")


@app.command("optimize-hedge")
def optimize_hedge_command(config: Path = Path("configs/example.yaml")) -> None:
    cfg = AnalysisConfig.from_yaml(config)
    scenarios = PriceScenarioSet.from_parquet(Path(cfg.output_dir) / "price_scenarios.parquet")
    dispatch = optimize_common_dispatch(scenarios, cfg.battery, cfg.dispatch_quantile)
    pnl = value_common_dispatch(dispatch, scenarios, cfg.battery)["net_pnl_eur"]
    years = sorted(set(pd.DatetimeIndex(scenarios.prices.index).year))
    quotes = synthetic_forward_quotes(years, timezone=cfg.timezone)
    payoffs = build_payoff_matrix(
        scenarios, products_from_quotes(quotes, cfg.hedge.products), cfg.battery.timestep_hours
    )
    result = optimize_hedge(
        pnl,
        payoffs,
        cfg.hedge,
        cfg.minimum_revenue_target_eur,
        cfg.battery.max_discharge_mw,
    )
    typer.echo(result.volumes_mw.to_string())
    typer.echo(f"Standard-deviation reduction: {result.statistics['std_reduction_pct']:.2f}%")


@app.command("run-all")
def run_all(config: Path = Path("configs/example.yaml")) -> None:
    """Run the complete fast synthetic demonstration."""
    result = run_analysis(AnalysisConfig.from_yaml(config))
    typer.echo(f"Analysis complete: {result.config.output_dir}")
    typer.echo(f"Expected unhedged P&L: {result.risk['expected_pnl_eur']:.2f} EUR")
    if result.hedge:
        typer.echo(f"Hedge volumes (MW):\n{result.hedge.volumes_mw.to_string()}")
        typer.echo(f"Variance reduction: {result.hedge.statistics['variance_reduction_pct']:.2f}%")


if __name__ == "__main__":
    app()
