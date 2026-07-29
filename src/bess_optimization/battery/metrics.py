"""Operational and financial dispatch metrics."""

from __future__ import annotations

import pandas as pd

from bess_optimization.config import BESSConfig


def dispatch_metrics(
    schedule: pd.DataFrame, prices: pd.Series, config: BESSConfig
) -> dict[str, float]:
    dt = config.timestep_hours
    charged = float(schedule["charge_mw"].sum() * dt)
    discharged = float(schedule["discharge_mw"].sum() * dt)
    charge_cost = float((schedule["charge_mw"] * prices).sum() * dt)
    discharge_revenue = float((schedule["discharge_mw"] * prices).sum() * dt)
    variable_cost = float(
        (
            schedule["charge_mw"] * config.variable_charge_cost_eur_mwh
            + schedule["discharge_mw"] * config.variable_discharge_cost_eur_mwh
        ).sum()
        * dt
    )
    degradation = discharged * config.degradation_cost_eur_mwh
    net = discharge_revenue - charge_cost - variable_cost - degradation
    cycles = discharged / config.energy_capacity_mwh
    days = max(len(schedule) * dt / 24, dt / 24)
    return {
        "energy_charged_mwh": charged,
        "energy_discharged_mwh": discharged,
        "equivalent_cycles": cycles,
        "equivalent_cycles_per_day": cycles / days,
        "charge_cost_eur": charge_cost,
        "discharge_revenue_eur": discharge_revenue,
        "gross_margin_eur": discharge_revenue - charge_cost,
        "variable_cost_eur": variable_cost,
        "degradation_cost_eur": degradation,
        "net_pnl_eur": net,
        "captured_spread_eur_mwh": (
            discharge_revenue / discharged - charge_cost / charged
            if charged > 0 and discharged > 0
            else 0.0
        ),
        "net_pnl_eur_per_mw": net / config.max_discharge_mw,
        "net_pnl_eur_per_mwh": net / config.energy_capacity_mwh,
    }
