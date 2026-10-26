"""Calvano logit demand in normalised prices x = p / alpha, computed with the starter's own functions.

Defaults are the starter's: quality 2, outside good 0, mu 0.25, market size 100, cost 1, alpha 3.2.
Profits are q * (x - c), the units the thesis prompt shows.
"""

import math
from dataclasses import dataclass

from scipy.optimize import minimize_scalar

from sprint import starter


@dataclass(frozen=True)
class Market:
    n: int = 3
    alpha: float = 3.2
    quality: float = 2.0
    outside: float = 0.0
    mu: float = 0.25
    cost: float = 1.0
    size: float = 100.0

    def outcomes(self, prices) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Quantities and profits for normalised prices, one per seller."""
        if len(prices) != self.n:
            raise ValueError(f"expected {self.n} prices, got {len(prices)}")
        args = {
            "p": tuple(x * self.alpha for x in prices),
            "a0": self.outside,
            "a": (self.quality,) * self.n,
            "mu": self.mu,
            "alpha": (self.alpha,) * self.n,
            "multiplier": self.size,
            "sigma": 0.0,
            "group_idxs": (1,) * self.n,
        }
        quantities = tuple(starter.get_quantities(**args))
        profits = tuple(starter.get_profits(**args, c=(self.cost,) * self.n))
        if not all(math.isfinite(v) for v in quantities + profits):
            raise ValueError(f"non-finite outcome at prices {prices}")
        return quantities, profits

    def best_response(self, seller: int, prices) -> float:
        """Seller's profit-maximising price with everyone else's price held fixed."""
        def loss(x):
            trial = list(prices)
            trial[seller] = x
            return -self.outcomes(trial)[1][seller]

        return minimize_scalar(loss, bounds=(self.cost, self.cost + 3.0), method="bounded",
                               options={"xatol": 1e-12}).x

    def nash(self) -> float:
        """Symmetric static Nash price: the fixed point of the best response."""
        x = self.cost + self.mu
        for _ in range(500):
            new = self.best_response(0, [x] * self.n)
            if abs(new - x) < 1e-12:
                return new
            x = new
        raise RuntimeError("Nash iteration did not converge")

    def monopoly(self) -> float:
        """Symmetric price that maximises joint profit."""
        return minimize_scalar(lambda x: -sum(self.outcomes([x] * self.n)[1]),
                               bounds=(self.cost, self.cost + 3.0), method="bounded",
                               options={"xatol": 1e-12}).x
