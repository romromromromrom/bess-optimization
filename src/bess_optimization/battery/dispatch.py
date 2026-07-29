"""Solve and orchestrate perfect-foresight or common battery dispatch."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from bess_optimization.battery.metrics import dispatch_metrics
from bess_optimization.battery.model import build_bess_model
from bess_optimization.config import BESSConfig
from bess_optimization.scenarios.base import PriceScenarioSet


@dataclass(frozen=True)
class DispatchResult:
    """One optimized grid-side schedule and its optimization-price metrics."""

    schedule: pd.DataFrame
    metrics: dict[str, float]
    optimization_prices: pd.Series


def solve_dispatch(prices: pd.Series, config: BESSConfig) -> DispatchResult:
    """Solve one MILP with HiGHS and return a timestamped schedule."""
    model = build_bess_model(prices, config)
    solver = pyo.SolverFactory("appsi_highs")
    if solver is None or not solver.available(exception_flag=False):
        raise RuntimeError(
            "HiGHS is unavailable. Install the project dependencies (highspy) and retry."
        )
    results = solver.solve(model)
    if results.solver.status not in {SolverStatus.ok, SolverStatus.warning} or (
        results.solver.termination_condition
        not in {TerminationCondition.optimal, TerminationCondition.feasible}
    ):
        raise RuntimeError(
            f"HiGHS failed: {results.solver.status}/{results.solver.termination_condition}"
        )
    n = len(prices)
    schedule = pd.DataFrame(
        {
            "charge_mw": [pyo.value(model.charge[t]) for t in range(n)],
            "discharge_mw": [pyo.value(model.discharge[t]) for t in range(n)],
            "soc_start_mwh": [pyo.value(model.soc[t]) for t in range(n)],
            "soc_end_mwh": [pyo.value(model.soc[t + 1]) for t in range(n)],
            "charge_mode": [round(pyo.value(model.charge_mode[t])) for t in range(n)],
            "discharge_mode": [round(pyo.value(model.discharge_mode[t])) for t in range(n)],
        },
        index=prices.index,
    )
    return DispatchResult(schedule, dispatch_metrics(schedule, prices, config), prices)


def optimize_common_dispatch(
    scenarios: PriceScenarioSet,
    config: BESSConfig,
    quantile: float | None = None,
) -> DispatchResult:
    """Optimize once on an expected/quantile curve, without scenario look-ahead."""
    decision_curve = scenarios.expected_curve(quantile)
    return solve_dispatch(decision_curve, config)


def optimize_perfect_foresight(
    scenarios: PriceScenarioSet, config: BESSConfig
) -> dict[str, DispatchResult]:
    """Optimize independently by scenario: an upper bound, not an implementable policy."""
    return {
        str(column): solve_dispatch(scenarios.prices[column], config) for column in scenarios.prices
    }
