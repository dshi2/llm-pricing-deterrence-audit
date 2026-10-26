"""Economics checked against an independent logit implementation and optimality conditions,
not against stored decimals alone."""

import math

import pytest

from sprint.market import Market


def logit_profits(prices, quality=2.0, mu=0.25, cost=1.0, size=100.0):
    """Independent textbook Calvano logit with outside good a0 = 0 (no starter code)."""
    weights = [math.exp((quality - x) / mu) for x in prices]
    total = 1.0 + sum(weights)
    return [size * w / total * (x - cost) for w, x in zip(weights, prices)]


@pytest.mark.parametrize("n", [2, 3])
def test_outcomes_match_independent_logit(n):
    market = Market(n)
    for prices in ([1.5] * n, [1.2, 1.9, 1.7][:n], [2.4, 1.05, 1.6][:n]):
        assert market.outcomes(prices)[1] == pytest.approx(logit_profits(prices), rel=1e-12)


@pytest.mark.parametrize("n", [2, 3])
def test_nash_is_a_mutual_best_response(n):
    market = Market(n)
    x = market.nash()
    h = 1e-5
    up = logit_profits([x + h] + [x] * (n - 1))[0]
    down = logit_profits([x - h] + [x] * (n - 1))[0]
    assert (up - down) / (2 * h) == pytest.approx(0.0, abs=1e-5)  # first-order condition
    stay = logit_profits([x] * n)[0]
    grid = [1.0 + i / 1000 for i in range(2001)]
    assert all(logit_profits([y] + [x] * (n - 1))[0] <= stay + 1e-9 for y in grid)


@pytest.mark.parametrize("n", [2, 3])
def test_monopoly_maximises_joint_profit(n):
    market = Market(n)
    x = market.monopoly()
    best = sum(logit_profits([x] * n))
    assert all(sum(logit_profits([1.0 + i / 1000] * n)) <= best + 1e-9 for i in range(2001))


def test_benchmarks_match_values_quoted_in_the_plan():
    assert Market(2).nash() == pytest.approx(1.4729, abs=5e-5)
    assert Market(2).monopoly() == pytest.approx(1.9250, abs=5e-5)
    assert Market(3).nash() == pytest.approx(1.3702, abs=5e-5)
    assert Market(3).monopoly() == pytest.approx(2.0000, abs=5e-5)
    assert Market(2).best_response(0, [0.0, 1.925]) == pytest.approx(1.6617, abs=5e-5)
    assert Market(2).best_response(0, [0.0, 1.75]) == pytest.approx(1.5928, abs=5e-5)


def test_wrong_length_rejected():
    with pytest.raises(ValueError):
        Market(3).outcomes([1.5, 1.5])
