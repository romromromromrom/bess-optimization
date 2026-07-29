"""ENTSO-E Transparency Platform connector."""

from __future__ import annotations

import os

import pandas as pd

from bess_optimization.data.schemas import validate_spot_prices


class EntsoeSpotClient:
    """Retrieve day-ahead prices using ``ENTSOE_API_KEY``."""

    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or os.getenv("ENTSOE_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "ENTSOE_API_KEY is not set. Export it or use the synthetic data mode."
            )
        self.api_key: str = resolved_key

    def fetch_day_ahead_prices(
        self,
        zone: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        timezone: str = "Europe/Paris",
        frequency: str = "1h",
    ) -> pd.Series:
        """Fetch and normalize prices; resampling supports hourly or 15-minute output."""
        try:
            from entsoe.entsoe import EntsoePandasClient
        except ImportError as exc:
            raise RuntimeError("Install entsoe-py to use the ENTSO-E connector") from exc

        start_ts = _localized_timestamp(start, timezone)
        end_ts = _localized_timestamp(end, timezone)
        raw = EntsoePandasClient(api_key=self.api_key).query_day_ahead_prices(
            zone, start=start_ts, end=end_ts
        )
        prices = raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw
        prices = prices.tz_convert(timezone)
        if frequency.lower() in {"15min", "15t"}:
            prices = prices.resample("15min").ffill()
        elif frequency.lower() in {"1h", "h", "hourly"}:
            prices = prices.resample("1h").mean()
        else:
            raise ValueError("frequency must be '1h' or '15min'")
        return validate_spot_prices(prices)


def _localized_timestamp(value: str | pd.Timestamp, timezone: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize(timezone) if ts.tz is None else ts.tz_convert(timezone)
