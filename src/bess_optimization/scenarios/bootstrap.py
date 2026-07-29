"""Conditional historical block-bootstrap scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bess_optimization.curves.hourly_shaping import historical_residuals
from bess_optimization.scenarios.base import PriceScenarioSet, ScenarioGenerator


class ConditionalBootstrapGenerator(ScenarioGenerator):
    """Sample contiguous blocks around an ensemble centered on the forward curve."""

    def __init__(self, block_days: int = 7) -> None:
        if block_days < 1:
            raise ValueError("block_days must be positive")
        self.block_days = block_days

    def generate(
        self,
        history: pd.Series,
        hourly_forward: pd.Series,
        n_scenarios: int,
        seed: int,
    ) -> PriceScenarioSet:
        rng = np.random.default_rng(seed)
        residual = historical_residuals(history).to_numpy()
        block_size = int(round(24 * self.block_days))
        if len(residual) < block_size:
            raise ValueError("History is shorter than one bootstrap block")
        shocks = np.empty((len(hourly_forward), n_scenarios))
        for scenario in range(n_scenarios):
            sampled: list[np.ndarray] = []
            while sum(map(len, sampled)) < len(hourly_forward):
                start = int(rng.integers(0, len(residual) - block_size + 1))
                sampled.append(residual[start : start + block_size])
            shocks[:, scenario] = np.concatenate(sampled)[: len(hourly_forward)]
        # Match the observable forward in cross-sectional expectation, not scenario by
        # scenario: distinct Calendar settlements are required for hedge payoffs.
        centered_shocks = shocks - shocks.mean(axis=1, keepdims=True)
        output = hourly_forward.to_numpy()[:, None] + centered_shocks
        columns = [f"scenario_{i:04d}" for i in range(n_scenarios)]
        return PriceScenarioSet(
            pd.DataFrame(output, index=hourly_forward.index, columns=columns),
            method="conditional_block_bootstrap",
            seed=seed,
        )
