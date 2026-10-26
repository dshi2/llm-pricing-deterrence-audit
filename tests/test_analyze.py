"""Diagnostic timing and descriptive-reporting regressions; primary payoff effects are unchanged."""

import json
import math

import pandas as pd
import pytest

from sprint.analyze import best_response_gaps, rival_responses, summary, verdict
from sprint.market import Market


def test_best_response_gaps_ignore_forced_branches_and_use_realised_rivals(tmp_path):
    from dataclasses import asdict

    market = Market(3)
    prices = [1.75, 1.60, 1.925]
    prices[0] = market.best_response(0, prices)
    profits = market.outcomes(prices)[1]
    events = [{"event": "header", "experiment": "main", "market": asdict(market), "game_seed": 3}]
    for branch in ("parent", "reference", "11:cut", "00:no_cut"):
        events.append({"event": "round", "branch": branch,
                       "row": {"number": 1, "prices": prices, "profits": profits}})
    path = tmp_path / "game.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events))
    table = best_response_gaps(path)
    assert len(table) == 6 and set(table.branch) == {"parent", "reference"}
    assert (table.profit_gap >= -1e-10).all()
    assert max(abs(table[table.seller == 0].profit_gap)) < 1e-10
    assert (table[table.seller != 0].profit_gap > 0).all()


def diagnostic_journal(path, horizon, cut_rounds):
    # A's reference changes sharply: current- and previous-round benchmarks must differ.
    reference = [(1.8, 1.7, 1.9), (2.2, 1.6, 1.8), (1.4, 2.0, 1.5)]
    events = []
    for i, prices in enumerate(reference[:horizon]):
        number = 11 + i
        events.append({"event": "round", "branch": "reference",
                       "row": {"number": number, "prices": prices}})
        for kind in ("cut", "no_cut"):
            branch = list(prices)
            if kind == "cut":
                if i < cut_rounds:
                    branch[0] *= 0.9
                branch[1] += (0.0, -0.2, -0.4)[i]
            events.append({"event": "round", "branch": f"10:{kind}",
                           "row": {"number": number, "prices": branch}})
        # Only B is live in this fixture; other masks need no rows if neither rival is live.
    events.append({"event": "complete", "result": {
        "status": "eligible", "roles": {"A": 0, "B": 1, "C": 2},
        "rules": {"parent_rounds": 10, "horizon": horizon,
                  "cut_rounds": cut_rounds, "cut_fraction": 0.1}}})
    # Populate the other live masks with the same rows; the test inspects only mask 10.
    for mask in ("01", "11"):
        events.extend({**e, "branch": e["branch"].replace("10:", f"{mask}:")}
                      for e in list(events) if e.get("branch", "").startswith("10:"))
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


@pytest.mark.parametrize("horizon, expected_response", [(2, -0.2), (3, -0.3)])
def test_response_benchmark_uses_observed_previous_prices(tmp_path, horizon, expected_response):
    path = tmp_path / "game.jsonl"
    diagnostic_journal(path, horizon=horizon, cut_rounds=2)
    table = rival_responses(path)
    row = table[(table["mask"] == "10") & (table["rival"] == "B")].iloc[0]
    market = Market(3)
    # Round 12 responds to the observed cut in round 11. Round 13, if recorded, responds
    # to round 12's cut even though A has already returned to its reference path in round 13.
    shifts = [market.best_response(1, [1.62, 1.7, 1.9])
              - market.best_response(1, [1.8, 1.7, 1.9])]
    if horizon == 3:
        shifts.append(market.best_response(1, [1.98, 1.6, 1.8])
                      - market.best_response(1, [2.2, 1.6, 1.8]))
    assert row["response"] == pytest.approx(expected_response)
    assert row["myopic_response"] == pytest.approx(sum(shifts) / len(shifts), abs=1e-7)


def test_no_post_cut_observation_produces_no_response_statistic(tmp_path):
    path = tmp_path / "game.jsonl"
    diagnostic_journal(path, horizon=1, cut_rounds=1)
    assert rival_responses(path).empty


@pytest.mark.parametrize("ci, expected", [
    ((1.0, 2.0), "excludes 0 (unadjusted)"),
    ((-2.0, -1.0), "excludes 0 (unadjusted)"),
    ((-1.0, 1.0), "includes 0"),
    ((0.0, 1.0), "includes 0"),
    ((-1.0, 0.0), "includes 0"),
    ((float("nan"), float("nan")), "too few games"),
])
def test_interval_labels_do_not_claim_significance_or_equivalence(ci, expected):
    assert verdict(ci) == expected


@pytest.mark.parametrize("n", [4, 10])
def test_summary_remains_descriptive_and_accounts_for_incomplete_games(n):
    setup = {"run": "pilot", "setup": "same-config", "model": "stub"}
    rows = [{**setup, "status": "eligible", "G00": 10.0, "G10": 8.0, "G01": 8.0,
             "G11": 6.0, "R1": 2.0, "R2": 4.0, "nonadditivity": 0.0,
             "aa_difference": 0.0} for _ in range(n)]
    rows += [{**setup, "status": "incomplete"},
             {**setup, "status": "failed", "reason": "declined: no decision"}]
    row = summary(pd.DataFrame(rows)).iloc[0]
    assert row["started"] == n + 2
    assert row["incomplete"] == 1
    assert row["failed"] == row["declined_games"] == 1
    assert row["intervals"] == ("descriptive; small sample" if n == 4 else "descriptive")
    assert row["R1_verdict"] == row["R2_verdict"] == "excludes 0 (unadjusted)"


@pytest.mark.parametrize("statuses, denominator, expected", [
    (["incomplete", "failed"], 0, None),
    (["ineligible", "incomplete", "failed"], 1, 0.0),
    (["eligible", "ineligible", "failed", "incomplete"], 2, 0.5),
])
def test_eligibility_distinguishes_no_data_from_zero(statuses, denominator, expected):
    values = {"G00": 10.0, "G10": 8.0, "G01": 8.0, "G11": 6.0,
              "R1": 2.0, "R2": 4.0, "nonadditivity": 0.0, "aa_difference": 0.0}
    rows = [{"run": "pilot", "setup": "same-config", "model": "stub", "status": status,
             **(values if status == "eligible" else {})} for status in statuses]
    result = summary(pd.DataFrame(rows)).iloc[0]
    assert result["eligibility_classified_games"] == denominator
    if expected is None:
        assert math.isnan(result["eligibility_rate"])
    else:
        assert result["eligibility_rate"] == expected
