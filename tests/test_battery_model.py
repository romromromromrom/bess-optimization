import pandas as pd
import pytest

from bess_optimization.battery.dispatch import solve_dispatch
from bess_optimization.config import BESSConfig


def test_soc_conservation_efficiency_limits_and_exclusive_modes() -> None:
    index = pd.date_range("2027-01-01", periods=4, freq="h", tz="Europe/Paris")
    prices = pd.Series([0.0, 0.0, 100.0, 100.0], index=index)
    config = BESSConfig(
        max_charge_mw=10,
        max_discharge_mw=10,
        energy_capacity_mwh=10,
        charge_efficiency=0.9,
        discharge_efficiency=0.9,
        min_soc_mwh=0,
        max_soc_mwh=10,
        initial_soc_mwh=0,
        final_soc_mwh=0,
        degradation_cost_eur_mwh=0,
        variable_discharge_cost_eur_mwh=0,
        max_equivalent_cycles_per_day=None,
    )
    result = solve_dispatch(prices, config)
    schedule = result.schedule
    expected_end = (
        schedule["soc_start_mwh"] + 0.9 * schedule["charge_mw"] - schedule["discharge_mw"] / 0.9
    )
    assert schedule["soc_end_mwh"].to_numpy() == pytest.approx(expected_end.to_numpy())
    assert schedule["charge_mw"].max() <= 10 + 1e-7
    assert schedule["discharge_mw"].max() <= 10 + 1e-7
    assert schedule["soc_end_mwh"].between(0, 10).all()
    assert not ((schedule["charge_mw"] > 1e-7) & (schedule["discharge_mw"] > 1e-7)).any()
    assert schedule["charge_mw"].sum() == pytest.approx(10 / 0.9)
    assert schedule["discharge_mw"].sum() == pytest.approx(9.0)


def test_unavailability_zeroes_power() -> None:
    index = pd.date_range("2027-01-01", periods=2, freq="h", tz="Europe/Paris")
    config = BESSConfig(
        energy_capacity_mwh=100,
        max_soc_mwh=100,
        unavailable_timestamps=[index[0].isoformat()],
        final_soc_mwh=50,
    )
    result = solve_dispatch(pd.Series([-100.0, 100.0], index=index), config)
    assert result.schedule.iloc[0]["charge_mw"] == pytest.approx(0)
    assert result.schedule.iloc[0]["discharge_mw"] == pytest.approx(0)
