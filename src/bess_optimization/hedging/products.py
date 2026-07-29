"""Futures product definitions and quote conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from bess_optimization.curves.forward_curve import latest_snapshot


@dataclass(frozen=True)
class FutureProduct:
    """Cash-settled Calendar future.

    A positive hedge volume is long and earns
    ``MW * delivery_hours * (settlement - entry_price)`` EUR. Negative volume is short.
    """

    name: str
    delivery_start: pd.Timestamp
    delivery_end: pd.Timestamp
    load_type: str
    entry_price_eur_mwh: float
    market: str


def products_from_quotes(quotes: pd.DataFrame, selection: str = "both") -> list[FutureProduct]:
    chosen = latest_snapshot(quotes)
    allowed = {
        "both": {"baseload", "peakload"},
        "baseload": {"baseload"},
        "peakload": {"peakload"},
    }
    if selection not in allowed:
        raise ValueError("selection must be baseload, peakload or both")
    products = []
    for _, raw_row in chosen.iterrows():
        row = cast(dict[str, Any], raw_row.to_dict())
        if row["load_type"] in allowed[selection]:
            products.append(
                FutureProduct(
                    name=str(row["product"]),
                    delivery_start=pd.Timestamp(row["delivery_start"]),
                    delivery_end=pd.Timestamp(row["delivery_end"]),
                    load_type=str(row["load_type"]),
                    entry_price_eur_mwh=float(row["price_eur_mwh"]),
                    market=str(row["market"]),
                )
            )
    return products
