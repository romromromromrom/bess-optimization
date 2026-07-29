"""Calendar Baseload/Peakload payoffs and hedge optimization."""

from bess_optimization.hedging.optimizer import HedgeResult, optimize_hedge
from bess_optimization.hedging.payoffs import build_payoff_matrix
from bess_optimization.hedging.products import FutureProduct, products_from_quotes

__all__ = [
    "FutureProduct",
    "HedgeResult",
    "build_payoff_matrix",
    "optimize_hedge",
    "products_from_quotes",
]
