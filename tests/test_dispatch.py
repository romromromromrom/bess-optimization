import pandas as pd
import pytest

from bess_optimization.battery.dispatch import optimize_common_dispatch, solve_dispatch
from bess_optimization.config import BESSConfig
from bess_optimization.scenarios.base import PriceScenarioSet
from bess_optimization.valuation.pnl import value_common_dispatch


def _config() -> BESSConfig:
    return BESSConfig(
        max_charge_mw=5,
        max_discharge_mw=5,
        energy_capacity_mwh=10,
        max_soc_mwh=10,
        initial_soc_mwh=0,
        final_soc_mwh=0,
        degradation_cost_eur_mwh=0,
        variable_discharge_cost_eur_mwh=0,
        max_equivalent_cycles_per_day=None,
    )


def test_flat_positive_curve_has_zero_dispatch() -> None:
    index = pd.date_range("2027-01-01", periods=8, freq="h", tz="UTC")
    result = solve_dispatch(pd.Series(50.0, index=index), _config())
    assert result.schedule["charge_mw"].sum() == pytest.approx(0, abs=1e-7)
    assert result.schedule["discharge_mw"].sum() == pytest.approx(0, abs=1e-7)
    assert result.metrics["net_pnl_eur"] == pytest.approx(0, abs=1e-7)


def test_low_then_high_prices_generate_arbitrage() -> None:
    index = pd.date_range("2027-01-01", periods=4, freq="h", tz="UTC")
    result = solve_dispatch(pd.Series([10.0, 10.0, 100.0, 100.0], index=index), _config())
    assert result.schedule.iloc[:2]["charge_mw"].sum() > 0
    assert result.schedule.iloc[2:]["discharge_mw"].sum() > 0
    assert result.metrics["net_pnl_eur"] > 0


def test_common_dispatch_has_no_scenario_lookahead() -> None:
    index = pd.date_range("2027-01-01", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {"up": [0.0, 0.0, 100.0, 100.0], "down": [100.0, 100.0, 0.0, 0.0]},
        index=index,
    )
    scenarios = PriceScenarioSet(frame, "test", 1)
    dispatch = optimize_common_dispatch(scenarios, _config())
    pnl = value_common_dispatch(dispatch, scenarios, _config())
    assert len(pnl) == 2
    reversed_scenarios = PriceScenarioSet(frame[["down", "up"]], "test", 1)
    reversed_dispatch = optimize_common_dispatch(reversed_scenarios, _config())
    pd.testing.assert_frame_equal(dispatch.schedule, reversed_dispatch.schedule)
