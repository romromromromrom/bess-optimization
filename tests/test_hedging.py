import numpy as np
import pandas as pd

from bess_optimization.config import HedgeConfig
from bess_optimization.hedging.optimizer import optimize_hedge


def test_synthetic_hedge_reduces_variance() -> None:
    rng = np.random.default_rng(42)
    index = pd.Index([f"s{i}" for i in range(200)])
    factor = rng.normal(0, 1000, len(index))
    payoff = pd.DataFrame(
        {"2027_baseload": factor, "2027_peakload": rng.normal(0, 500, len(index))},
        index=index,
    )
    spot_pnl = pd.Series(100_000 - 2.0 * factor + rng.normal(0, 50, len(index)), index=index)
    result = optimize_hedge(
        spot_pnl,
        payoff,
        HedgeConfig(min_volume_mw=-10, max_volume_mw=10, regularization=1e-9),
    )
    assert result.statistics["variance_reduction_pct"] > 95
    assert abs(result.volumes_mw["2027_baseload"] - 2.0) < 0.1
