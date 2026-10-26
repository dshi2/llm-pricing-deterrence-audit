import asyncio
import math
from dataclasses import asdict

import pandas as pd
import pytest

from sprint import analyze
from sprint.checks import SCENARIOS, expected_gains
from sprint.engine import Journal
from sprint.experiments import MainRules, main_game
from sprint.market import Market
from sprint.scripted import as_policy
from sprint.study import export, late_price_trends, primary


def records(prices):
    return [{"game_seed": 3}] + [
        {"event": "round", "branch": "parent", "row": {"number": n, "prices": p}}
        for n, p in zip(range(100, 151), prices, strict=True)]


def test_trends_expose_cancelling_seller_drifts():
    result = late_price_trends(records([(1 + i/100, 2 - i/100) for i in range(51)])).set_index("series")
    assert result.loc["market_mean", "slope_per_round"] == pytest.approx(0)
    assert result.loc["seller_0", "slope_per_round"] == pytest.approx(0.01)
    assert result.loc["seller_1", "fitted_change_50_rounds"] == pytest.approx(-0.5)
    assert result.loc["seller_0", "late_minus_early_mean"] == pytest.approx(0.26)


def test_flat_slope_does_not_hide_oscillation():
    result = late_price_trends(records([(1 + i % 2,) for i in range(51)])).iloc[0]
    assert result.slope_per_round == pytest.approx(0)
    assert result.sd > 0.49
    assert result.mean_absolute_step == 1


def test_incomplete_or_duplicate_window_is_not_zero_trend():
    events = records([(1.75,)] * 51)
    for invalid in (events[:-1], events + [events[-1]]):
        result = late_price_trends(invalid)
        assert result.iloc[0].status == "window incomplete"
        assert "slope_per_round" not in result
    result = late_price_trends(events).iloc[0]
    assert result.slope_per_round == 0
    assert result.sd == 0


def test_s_minus_j_is_a_paired_primary_interval():
    table = pd.DataFrame([{"run": "test", "setup": "fixed", "model": "stub", "status": "eligible",
                           "G10": 6, "G01": 2, "G11": -1},
                          {"run": "test", "setup": "fixed", "model": "stub", "status": "eligible",
                           "G10": 4, "G01": 4, "G11": 1}])
    result = primary(table, "pooled").set_index("outcome").loc["S_minus_J"]
    assert result["mean"] == pytest.approx(4)  # per game: 5 and 3
    assert result.level == pytest.approx(1 - 0.05 / 16)
    assert "16 planned" in result.method


def table(triples):
    return pd.DataFrame([{"run": "pooled", "setup": "fixed", "model": "test", "status": "eligible",
                          "G10": b, "G01": c, "G11": j} for b, c, j in triples])


def test_primary_does_not_turn_average_into_joint_only_claim():
    result = primary(table([(-10, 30, -20), (-10.5, 30.5, -20.5), (-9.5, 29.5, -19.5)]), "pooled").set_index("outcome")
    assert result.loc["S", "mean"] == 10
    assert result.loc["M", "mean"] == -10
    assert result.loc["M", "interpretation"] == "negative M does not establish individual deterrence"
    assert all(result.level == 1 - 0.05 / 16)
    assert result.loc["S", "interpretation"] == "uncertainty unavailable"  # zero variance
    assert not any("joint-only" in x for x in result.interpretation)
    counterexample = primary(table([(1, 1, 1), (-0.1, 100, -100)]), "pilot").set_index("outcome")
    assert counterexample.loc["M", "mean"] == pytest.approx(0.45)
    assert counterexample.loc["J", "mean"] == -49.5
    assert set(counterexample.scope) == {"exploratory"}


def test_primary_interval_matches_closed_form_t_with_two_degrees_of_freedom():
    result = primary(table([(1, 1, -1), (2, 2, -2), (3, 3, -3)]), "pooled").set_index("outcome")
    # For t(df=2), CDF(x) = 1/2 + x/(2*sqrt(x*x+2)); invert independently of scipy.
    u = 2 * (1 - 0.05 / 32) - 1
    critical = math.sqrt(2) * u / math.sqrt(1 - u*u)
    radius = critical / math.sqrt(3)  # sample SD of [1, 2, 3] is 1
    assert result.loc["S", "lo"] == pytest.approx(2 - radius)
    assert result.loc["S", "hi"] == pytest.approx(2 + radius)
    assert result.loc["S", "leave_one_out_min"] == 1.5
    assert result.loc["S", "leave_one_out_max"] == 2.5


def test_export_keeps_failed_and_unstarted_games(tmp_path, monkeypatch):
    from sprint import analyze

    monkeypatch.setattr(analyze, "games", lambda paths: pd.DataFrame([
        {"run": "main_test", "setup": "fixed", "model": "test", "seed": 1, "status": "failed"}]))
    games, estimates = export([tmp_path / "main_test"], tmp_path / "reports", "pooled", {"main_test": [1, 2, 3]})
    assert games.status.value_counts().to_dict() == {"not_started": 2, "failed": 1}
    assert (estimates.eligible == 0).all()
    assert estimates["mean"].isna().all()
    assert (tmp_path / "reports/accounting.csv").exists()


def test_controls_give_known_s_j_m_through_runner_and_analysis(tmp_path):
    """Positive and negative controls from saved journals to the primary table, against independent answers."""
    rules, expected = MainRules(), {}
    for name, strategies in SCENARIOS.items():
        async def policy(request, label, strategies=strategies):
            return await as_policy(strategies[request.seller])(request, label)

        for seed in (7, 8):
            head = {"experiment": "main", "game_seed": seed, "rules": asdict(rules), "market": asdict(Market(3)),
                    "model": "scripted", "code_sha": "scripted"}
            asyncio.run(main_game(policy, seed, Journal(tmp_path / name / f"game_{seed}.jsonl", head), Market(3), rules))
            roles = analyze._events(tmp_path / name / f"game_{seed}.jsonl")[-1]["result"]["roles"]
            g = expected_gains(strategies, roles, rules)
            expected.setdefault(name, []).append({"S": (g["10"] + g["01"]) / 2, "J": g["11"], "M": min(g["10"], g["01"]),
                                                  "S_minus_J": (g["10"] + g["01"]) / 2 - g["11"]})
    _, result = export([tmp_path / name for name in SCENARIOS], tmp_path / "out", "pooled")
    means = result.pivot(index="run", columns="outcome", values="mean")
    for name, games in expected.items():
        assert means.loc[name].to_dict() == pytest.approx(pd.DataFrame(games).mean().to_dict(), abs=1e-9)
    assert means.loc["no_response", "J"] > 0 and means.loc["no_response", "S_minus_J"] == pytest.approx(0)
    assert means.loc["one_responder", "M"] < 0  # one rival is enough
    assert means.loc["all_respond_strong", ["S", "J"]].lt(0).all()
    mild = means.loc["all_respond_mild"]
    assert mild.S > 0 and mild.M > 0 and mild.J < 0 and mild.S_minus_J > 0  # only both together deter
