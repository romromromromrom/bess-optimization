import pandas as pd
import pytest

from bess_optimization.battery.metrics import dispatch_metrics
from bess_optimization.config import BESSConfig
from bess_optimization.valuation.risk import risk_metrics


def test_mw_mwh_hours_and_euros_units() -> None:
    index = pd.date_range("2027-01-01", periods=2, freq="h", tz="UTC")
    schedule = pd.DataFrame(
        {
            "charge_mw": [2.0, 0.0],
            "discharge_mw": [0.0, 1.0],
            "soc_start_mwh": [0.0, 2.0],
            "soc_end_mwh": [2.0, 1.0],
        },
        index=index,
    )
    config = BESSConfig(
        energy_capacity_mwh=10,
        max_soc_mwh=10,
        initial_soc_mwh=0,
        final_soc_mwh=None,
        timestep_hours=0.5,
        degradation_cost_eur_mwh=3,
        variable_discharge_cost_eur_mwh=2,
        max_equivalent_cycles_per_day=None,
    )
    metrics = dispatch_metrics(schedule, pd.Series([10.0, 100.0], index=index), config)
    assert metrics["energy_charged_mwh"] == pytest.approx(1.0)
    assert metrics["energy_discharged_mwh"] == pytest.approx(0.5)
    assert metrics["charge_cost_eur"] == pytest.approx(10.0)
    assert metrics["discharge_revenue_eur"] == pytest.approx(50.0)
    assert metrics["net_pnl_eur"] == pytest.approx(37.5)


def test_risk_metrics_tail_conventions() -> None:
    metrics = risk_metrics(pd.Series([-100.0, 0.0, 100.0, 200.0]))
    assert metrics["expected_pnl_eur"] == pytest.approx(50.0)
    assert metrics["probability_below_zero"] == pytest.approx(0.25)
    assert metrics["var_95_eur"] > 0
