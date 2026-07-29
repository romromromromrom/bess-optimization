"""Pyomo MILP formulation for battery arbitrage."""

from __future__ import annotations

import math

import pandas as pd
import pyomo.environ as pyo

from bess_optimization.config import BESSConfig


def build_bess_model(prices: pd.Series, config: BESSConfig) -> pyo.ConcreteModel:
    """Build a unit-consistent MILP.

    ``charge`` and ``discharge`` are grid-side MW. ``soc`` is stored MWh. The market
    cashflow for one period is ``price * (discharge-charge) * timestep_hours``.
    """
    if prices.empty:
        raise ValueError("At least one price is required")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices require a DatetimeIndex")
    n = len(prices)
    dt = config.timestep_hours
    model = pyo.ConcreteModel("bess_dispatch")
    model.T = pyo.RangeSet(0, n - 1)
    model.S = pyo.RangeSet(0, n)
    price_values = dict(enumerate(prices.astype(float).to_numpy()))
    availability = _availability(prices.index, config)
    model.price = pyo.Param(model.T, initialize=price_values)
    model.availability = pyo.Param(model.T, initialize=dict(enumerate(availability)))
    model.charge = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.discharge = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.soc = pyo.Var(
        model.S,
        bounds=(config.min_soc_mwh, config.max_soc_mwh),
        domain=pyo.NonNegativeReals,
    )
    model.charge_mode = pyo.Var(model.T, domain=pyo.Binary)
    model.discharge_mode = pyo.Var(model.T, domain=pyo.Binary)

    model.initial_soc = pyo.Constraint(expr=model.soc[0] == config.initial_soc_mwh)
    if config.final_soc_mwh is not None:
        model.final_soc = pyo.Constraint(expr=model.soc[n] == config.final_soc_mwh)
    model.soc_balance = pyo.Constraint(
        model.T,
        rule=lambda m, t: (
            m.soc[t + 1]
            == m.soc[t]
            + config.charge_efficiency * m.charge[t] * dt
            - m.discharge[t] * dt / config.discharge_efficiency
        ),
    )
    model.charge_limit = pyo.Constraint(
        model.T,
        rule=lambda m, t: (
            m.charge[t] <= config.max_charge_mw * m.availability[t] * m.charge_mode[t]
        ),
    )
    model.discharge_limit = pyo.Constraint(
        model.T,
        rule=lambda m, t: (
            m.discharge[t] <= config.max_discharge_mw * m.availability[t] * m.discharge_mode[t]
        ),
    )
    model.exclusive_mode = pyo.Constraint(
        model.T, rule=lambda m, t: m.charge_mode[t] + m.discharge_mode[t] <= 1
    )
    if config.max_equivalent_cycles_per_day is not None:
        days = max(n * dt / 24, dt / 24)
        model.cycle_limit = pyo.Constraint(
            expr=sum(model.discharge[t] * dt for t in model.T)
            <= config.max_equivalent_cycles_per_day * days * config.energy_capacity_mwh
        )
    if config.min_mode_duration_hours:
        _add_min_mode_constraints(model, n, config.min_mode_duration_hours, dt)

    spot_margin = sum(model.price[t] * (model.discharge[t] - model.charge[t]) * dt for t in model.T)
    operating_cost = sum(
        (
            config.variable_charge_cost_eur_mwh * model.charge[t]
            + (config.variable_discharge_cost_eur_mwh + config.degradation_cost_eur_mwh)
            * model.discharge[t]
        )
        * dt
        for t in model.T
    )
    model.objective = pyo.Objective(expr=spot_margin - operating_cost, sense=pyo.maximize)
    return model


def _availability(index: pd.DatetimeIndex, config: BESSConfig) -> list[float]:
    unavailable: set[pd.Timestamp] = set()
    for value in config.unavailable_timestamps:
        ts = pd.Timestamp(value)
        ts = ts.tz_localize(index.tz) if ts.tz is None else ts.tz_convert(index.tz)
        unavailable.add(ts)
    return [0.0 if timestamp in unavailable else config.availability_factor for timestamp in index]


def _add_min_mode_constraints(model: pyo.ConcreteModel, n: int, hours: int, dt: float) -> None:
    periods = max(1, math.ceil(hours / dt))
    model.charge_start = pyo.Var(model.T, domain=pyo.Binary)
    model.discharge_start = pyo.Var(model.T, domain=pyo.Binary)
    model.charge_start_rule = pyo.Constraint(
        model.T,
        rule=lambda m, t: (
            m.charge_start[t] >= m.charge_mode[t] - (m.charge_mode[t - 1] if t > 0 else 0)
        ),
    )
    model.discharge_start_rule = pyo.Constraint(
        model.T,
        rule=lambda m, t: (
            m.discharge_start[t] >= m.discharge_mode[t] - (m.discharge_mode[t - 1] if t > 0 else 0)
        ),
    )

    def min_charge(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
        available = min(periods, n - t)
        return sum(m.charge_mode[k] for k in range(t, t + available)) >= (
            available * m.charge_start[t]
        )

    def min_discharge(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
        available = min(periods, n - t)
        return sum(m.discharge_mode[k] for k in range(t, t + available)) >= (
            available * m.discharge_start[t]
        )

    model.min_charge_duration = pyo.Constraint(model.T, rule=min_charge)
    model.min_discharge_duration = pyo.Constraint(model.T, rule=min_discharge)
