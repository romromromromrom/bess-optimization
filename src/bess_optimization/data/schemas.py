"""Canonical tabular schemas and validation helpers."""

from __future__ import annotations

import pandas as pd

FORWARD_COLUMNS = [
    "observation_date",
    "delivery_start",
    "delivery_end",
    "product",
    "load_type",
    "price_eur_mwh",
    "volume_mw",
    "currency",
    "market",
]


def validate_spot_prices(prices: pd.Series) -> pd.Series:
    """Return sorted numeric spot prices with a timezone-aware unique index."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Spot prices require a DatetimeIndex")
    if prices.index.tz is None:
        raise ValueError("Spot price timestamps must be timezone-aware")
    result = pd.to_numeric(prices, errors="coerce").astype(float).sort_index()
    if result.index.has_duplicates:
        result = result.groupby(level=0).mean()
    if result.isna().any():
        raise ValueError("Spot prices contain missing or non-numeric values")
    result.name = "price_eur_mwh"
    return result


def validate_forward_quotes(frame: pd.DataFrame, timezone: str = "Europe/Paris") -> pd.DataFrame:
    """Validate and normalize user-provided licensed forward quotes."""
    missing = set(FORWARD_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing forward columns: {sorted(missing)}")
    result = frame[FORWARD_COLUMNS].copy()
    for column in ("observation_date", "delivery_start", "delivery_end"):
        parsed = pd.to_datetime(result[column], errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"Invalid dates in {column}")
        result[column] = parsed.dt.tz_localize(
            timezone, ambiguous="infer", nonexistent="shift_forward"
        )
    result["load_type"] = result["load_type"].str.lower()
    if not result["load_type"].isin(["baseload", "peakload"]).all():
        raise ValueError("load_type must contain only baseload or peakload")
    result["price_eur_mwh"] = pd.to_numeric(result["price_eur_mwh"], errors="raise")
    result["volume_mw"] = pd.to_numeric(result["volume_mw"], errors="raise")
    if (result["delivery_end"] < result["delivery_start"]).any():
        raise ValueError("delivery_end must be on or after delivery_start")
    return result.sort_values(["observation_date", "delivery_start", "load_type"]).reset_index(
        drop=True
    )
