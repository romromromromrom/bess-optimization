"""Deterministic synthetic spot and forward data for demos and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_spot_prices(
    start: str = "2022-01-01",
    end: str = "2025-12-31 23:00",
    timezone: str = "Europe/Paris",
    seed: int = 42,
) -> pd.Series:
    """Generate explicitly synthetic hourly prices with seasonality, spikes and negatives."""
    index = pd.date_range(start, end, freq="h", tz=timezone)
    rng = np.random.default_rng(seed)
    hour = index.hour.to_numpy()
    day = index.dayofyear.to_numpy()
    weekday = index.dayofweek.to_numpy() < 5
    seasonal = 12 * np.cos(2 * np.pi * (day - 15) / 365.25)
    intraday = 15 * np.exp(-(((hour - 19) / 3) ** 2)) - 7 * np.exp(-(((hour - 4) / 3) ** 2))
    weekday_effect = np.where(weekday, 5.0, -4.0)
    innovations = rng.normal(0, 8, len(index))
    noise = np.empty(len(index))
    noise[0] = innovations[0]
    for i in range(1, len(index)):
        noise[i] = 0.72 * noise[i - 1] + innovations[i]
    prices = 65 + seasonal + intraday + weekday_effect + noise
    spike_mask = rng.random(len(index)) < 0.002
    negative_mask = rng.random(len(index)) < 0.002
    prices[spike_mask] += rng.lognormal(5.0, 0.55, spike_mask.sum())
    prices[negative_mask] -= rng.uniform(70, 160, negative_mask.sum())
    return pd.Series(prices, index=index, name="price_eur_mwh")


def synthetic_forward_quotes(
    years: list[int],
    observation_dates: list[str] | None = None,
    timezone: str = "Europe/Paris",
    market: str = "FR",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic Calendar Baseload and Peakload snapshots."""
    observations = observation_dates or ["2026-06-30", "2026-07-15", "2026-07-29"]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for snap_no, observation in enumerate(observations):
        common_move = rng.normal(0, 1.5)
        for i, year in enumerate(years):
            base = 72 - 2.5 * i + common_move + snap_no * rng.normal(0, 0.4)
            peak = base + 10 + rng.normal(0, 0.8)
            for load_type, price in (("baseload", base), ("peakload", peak)):
                rows.append(
                    {
                        "observation_date": pd.Timestamp(observation, tz=timezone),
                        "delivery_start": pd.Timestamp(f"{year}-01-01", tz=timezone),
                        "delivery_end": pd.Timestamp(f"{year}-12-31 23:00", tz=timezone),
                        "product": f"Calendar {year} {load_type.title()}",
                        "load_type": load_type,
                        "price_eur_mwh": price,
                        "volume_mw": 1.0,
                        "currency": "EUR",
                        "market": market,
                    }
                )
    return pd.DataFrame(rows)
