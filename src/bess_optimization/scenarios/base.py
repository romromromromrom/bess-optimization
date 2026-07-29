"""Common scenario interfaces and storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PriceScenarioSet:
    """Hourly prices in EUR/MWh; index=timestamps, columns=scenario identifiers."""

    prices: pd.DataFrame
    method: str
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise ValueError("Scenario prices require a DatetimeIndex")
        if self.prices.index.tz is None:
            raise ValueError("Scenario timestamps must be timezone-aware")
        if self.prices.empty or self.prices.isna().any().any():
            raise ValueError("Scenario prices must be non-empty and complete")
        if not self.prices.index.is_monotonic_increasing or self.prices.index.has_duplicates:
            raise ValueError("Scenario timestamps must be sorted and unique")

    @property
    def n_scenarios(self) -> int:
        return self.prices.shape[1]

    def expected_curve(self, quantile: float | None = None) -> pd.Series:
        if quantile is None:
            return self.prices.mean(axis=1).rename("expected_price_eur_mwh")
        return self.prices.quantile(quantile, axis=1).rename(
            f"price_q{round(quantile * 100):02d}_eur_mwh"
        )

    def to_parquet(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.prices.to_parquet(output)

    @classmethod
    def from_parquet(
        cls, path: str | Path, method: str = "loaded", seed: int = 0
    ) -> PriceScenarioSet:
        return cls(pd.read_parquet(path), method=method, seed=seed)


class ScenarioGenerator(ABC):
    """Interface shared by bootstrap, factor and future regime models."""

    @abstractmethod
    def generate(
        self,
        history: pd.Series,
        hourly_forward: pd.Series,
        n_scenarios: int,
        seed: int,
    ) -> PriceScenarioSet:
        """Generate hourly price trajectories around the observable forward curve."""


class RegimeScenarioGenerator(ScenarioGenerator):
    """Documented extension point for HMM/Markov volatility regimes."""

    def generate(
        self,
        history: pd.Series,
        hourly_forward: pd.Series,
        n_scenarios: int,
        seed: int,
    ) -> PriceScenarioSet:
        raise NotImplementedError(
            "Regime scenarios are an extension: calibrate transition probabilities, "
            "state volatilities and extreme-event tails before use."
        )
