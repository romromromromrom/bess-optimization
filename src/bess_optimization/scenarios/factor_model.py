"""PCA factor scenarios calibrated on successive forward snapshots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from bess_optimization.curves.hourly_shaping import is_peak
from bess_optimization.scenarios.base import PriceScenarioSet, ScenarioGenerator


class FactorScenarioGenerator(ScenarioGenerator):
    """Simulate level/slope/curvature shocks from a panel of forward snapshots.

    ``fit`` expects successive Calendar quotes. PCA is applied to price differences.
    Hourly idiosyncratic AR(1) noise, calibrated from spot history, preserves within-year
    variability while annual peak/off-peak factor shocks move the forward blocks.
    """

    def __init__(self, n_factors: int = 3) -> None:
        self.n_factors = n_factors
        self._pca: PCA | None = None
        self._factor_std: np.ndarray | None = None
        self._feature_years: np.ndarray | None = None
        self._feature_loads: np.ndarray | None = None

    def fit(self, quotes: pd.DataFrame) -> FactorScenarioGenerator:
        data = quotes.copy()
        data["year"] = data["delivery_start"].dt.year
        panel = data.pivot_table(
            index="observation_date",
            columns=["year", "load_type"],
            values="price_eur_mwh",
            aggfunc="mean",
        ).sort_index()
        changes = panel.diff().dropna()
        if len(changes) < 2:
            # Stable fallback remains functional for very small demo panels.
            changes = pd.DataFrame(
                [np.zeros(panel.shape[1]), np.ones(panel.shape[1])],
                columns=panel.columns,
            )
        changes = changes.fillna(0.0)
        n_components = min(self.n_factors, changes.shape[0], changes.shape[1])
        self._pca = PCA(n_components=max(n_components, 1)).fit(changes)
        scores = self._pca.transform(changes)
        self._factor_std = np.maximum(scores.std(axis=0, ddof=0), 0.25)
        self._feature_years = np.array([item[0] for item in panel.columns], dtype=int)
        self._feature_loads = np.array([item[1] for item in panel.columns], dtype=str)
        return self

    def generate(
        self,
        history: pd.Series,
        hourly_forward: pd.Series,
        n_scenarios: int,
        seed: int,
    ) -> PriceScenarioSet:
        rng = np.random.default_rng(seed)
        index = pd.DatetimeIndex(hourly_forward.index)
        output = np.empty((len(index), n_scenarios))
        innovations_std = max(float(history.diff().std()) * 0.55, 1.0)
        for scenario in range(n_scenarios):
            block_shocks = self._simulate_block_shocks(rng)
            noise = np.empty(len(index))
            noise[0] = rng.normal(0, innovations_std)
            for i in range(1, len(index)):
                noise[i] = 0.75 * noise[i - 1] + rng.normal(0, innovations_std)
            values = hourly_forward.to_numpy() + noise
            peak = is_peak(index)
            for year in set(index.year):
                for load_type, mask in (("peakload", peak), ("baseload", ~peak)):
                    group = (index.year == year) & mask
                    values[group] += block_shocks.get((year, load_type), 0.0)
            output[:, scenario] = values
        output += hourly_forward.to_numpy()[:, None] - output.mean(axis=1, keepdims=True)
        columns = [f"scenario_{i:04d}" for i in range(n_scenarios)]
        return PriceScenarioSet(
            pd.DataFrame(output, index=index, columns=columns),
            method="pca_factor",
            seed=seed,
        )

    def _simulate_block_shocks(self, rng: np.random.Generator) -> dict[tuple[int, str], float]:
        if (
            self._pca is None
            or self._factor_std is None
            or self._feature_years is None
            or self._feature_loads is None
        ):
            return {}
        factors = rng.normal(0, self._factor_std)
        shocks = self._pca.inverse_transform(factors.reshape(1, -1))[0]
        return {
            (int(year), str(load)): float(value)
            for year, load, value in zip(
                self._feature_years, self._feature_loads, shocks, strict=True
            )
        }
