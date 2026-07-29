"""Scenario settlement and unit-volume futures payoffs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bess_optimization.curves.hourly_shaping import is_peak
from bess_optimization.hedging.products import FutureProduct
from bess_optimization.scenarios.base import PriceScenarioSet


def build_payoff_matrix(
    scenarios: PriceScenarioSet,
    products: list[FutureProduct],
    timestep_hours: float = 1.0,
) -> pd.DataFrame:
    """Return EUR payoff per +1 MW long position for each scenario and product.

    When scenarios cover only part of a delivery period (as in the fast demo), hours and
    settlement use only the overlapping interval. Production research should use the full
    Calendar delivery horizon.
    """
    matrix: dict[str, pd.Series] = {}
    index = pd.DatetimeIndex(scenarios.prices.index)
    for product in products:
        interval = (index >= product.delivery_start) & (index <= product.delivery_end)
        if product.load_type == "peakload":
            interval &= is_peak(index)
        if not np.any(interval):
            continue
        settlement = scenarios.prices.loc[interval].mean(axis=0)
        delivery_hours = float(np.sum(interval) * timestep_hours)
        name = _unique_name(product, matrix)
        matrix[name] = delivery_hours * (settlement - product.entry_price_eur_mwh)
    if not matrix:
        raise ValueError("No futures product overlaps the scenario horizon")
    return pd.DataFrame(matrix, index=scenarios.prices.columns)


def _unique_name(product: FutureProduct, existing: dict[str, pd.Series]) -> str:
    base = f"{product.delivery_start.year}_{product.load_type}"
    if base not in existing:
        return base
    return f"{base}_{len(existing)}"
