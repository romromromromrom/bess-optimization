import pandas as pd
import pytest

from bess_optimization.curves.forward_curve import build_hourly_forward_curve
from bess_optimization.data.synthetic_data import synthetic_forward_quotes, synthetic_spot_prices
from bess_optimization.scenarios.bootstrap import ConditionalBootstrapGenerator
from bess_optimization.scenarios.factor_model import FactorScenarioGenerator


def test_scenarios_are_reproducible_and_dimensionally_consistent() -> None:
    history = synthetic_spot_prices("2024-01-01", "2024-03-31 23:00", seed=5)
    index = pd.date_range("2027-01-01", periods=24 * 7, freq="h", tz="Europe/Paris")
    quotes = synthetic_forward_quotes([2027], seed=5)
    forward = build_hourly_forward_curve(quotes, index, history)
    generator = ConditionalBootstrapGenerator(block_days=2)
    first = generator.generate(history, forward, n_scenarios=4, seed=123)
    second = generator.generate(history, forward, n_scenarios=4, seed=123)
    pd.testing.assert_frame_equal(first.prices, second.prices)
    assert first.prices.shape == (24 * 7, 4)
    assert first.prices.index.equals(index)
    assert first.prices.min().min() < first.prices.max().max()


def test_scenario_means_are_rebased_to_forward_groups() -> None:
    history = synthetic_spot_prices("2024-01-01", "2024-06-30 23:00", seed=8)
    index = pd.date_range("2027-01-01", periods=24 * 14, freq="h", tz="Europe/Paris")
    forward = build_hourly_forward_curve(synthetic_forward_quotes([2027]), index, history)
    scenarios = ConditionalBootstrapGenerator(3).generate(history, forward, 3, 9)
    assert scenarios.prices.mean().mean() == pytest.approx(forward.mean(), abs=1e-8)


def test_factor_model_is_functional() -> None:
    history = synthetic_spot_prices("2024-01-01", "2024-03-31 23:00", seed=4)
    index = pd.date_range("2027-01-01", periods=48, freq="h", tz="Europe/Paris")
    quotes = synthetic_forward_quotes([2027], seed=4)
    forward = build_hourly_forward_curve(quotes, index, history)
    scenarios = FactorScenarioGenerator().fit(quotes).generate(history, forward, 4, 4)
    assert scenarios.prices.shape == (48, 4)
    assert scenarios.prices.mean(axis=1).to_numpy() == pytest.approx(forward.to_numpy())
