"""Engine and main-study checks with scripted policies: independent answers, exact resume after
interruption, information isolation, and journal integrity."""

import asyncio
import json
import random
from dataclasses import replace

import pytest

from sprint.checks import scripted_check
from sprint.engine import Decision, Journal
from sprint.experiments import MainRules, main_game
from sprint.market import Market

RULES = MainRules()
HEADER = {"test": "engine"}


async def noisy(request, label):
    """Deterministic in the request: noise from the sampling seed, punishes a visible undercut."""
    last = request.history[-1].prices if request.history else None
    if last and any(p < 1.70 for i, p in enumerate(last) if i != request.seller):
        return Decision(1.40, f"plans {request.seller}", "punish")
    return Decision(1.76 + 0.02 * random.Random(request.seed).random(), f"plans {request.seller}", "calm")


def run(policy, journal, seed=3):
    return asyncio.run(main_game(policy, seed, journal, Market(3), RULES))


def committed(path):
    return [json.dumps(e, sort_keys=True) for e in map(json.loads, path.read_text().splitlines())
            if e["event"] == "round"]


def test_scripted_scenarios_match_independent_answers(tmp_path):
    table = scripted_check(tmp_path).set_index("scenario")
    assert (table["max_error"] < 1e-9).all()
    assert table.loc["no_response", "R1"] == pytest.approx(0) and table.loc["no_response", "R2"] == pytest.approx(0)
    mild = table.loc["all_respond_mild"]
    assert mild["G10"] > 0 and mild["G01"] > 0 and mild["G11"] < 0  # only both together deter


def test_call_count_matches_plan(tmp_path):
    calls = []

    async def counting(request, label):
        calls.append(label)
        return await noisy(request, label)

    result = run(counting, Journal(tmp_path / "g.jsonl", HEADER))
    assert result["status"] == "eligible"
    assert len(calls) == RULES.planned_calls()["eligible"] == 710


@pytest.mark.parametrize("stop_after", [1, 200, 452, 530, 600, 709])
def test_interrupted_game_resumes_to_the_identical_result(tmp_path, stop_after):
    control_path = tmp_path / "control.jsonl"
    control = run(noisy, Journal(control_path, HEADER))

    calls = []

    async def flaky(request, label):
        calls.append((label, request))
        if len(calls) == stop_after:
            raise ConnectionError("simulated disconnect")
        return await noisy(request, label)

    path = tmp_path / "resumed.jsonl"
    with pytest.raises(ConnectionError):
        run(flaky, Journal(path, HEADER))
    before = len(calls)
    resumed_calls = []

    async def recording(request, label):
        resumed_calls.append(label)
        return await noisy(request, label)

    result = run(recording, Journal(path, HEADER))
    assert result == control
    assert committed(path) == committed(control_path)
    # Only the failed request and the ones never made are sent again; saved decisions are reused.
    assert before - 1 + len(resumed_calls) == 710
    assert calls[stop_after - 1][0] in resumed_calls


def test_partial_round_keeps_saved_decisions(tmp_path):
    async def seller1_fails_round_5(request, label):
        if request.seller == 1 and request.round_number == 5:
            raise TimeoutError("simulated timeout")
        return await noisy(request, label)

    path = tmp_path / "g.jsonl"
    with pytest.raises(TimeoutError):
        run(seller1_fails_round_5, Journal(path, HEADER))
    saved = [e for e in map(json.loads, path.read_text().splitlines()) if e["event"] == "decision" and e["round"] == 5]
    assert sorted(e["seller"] for e in saved) == [0, 2]
    retried = []

    async def recording(request, label):
        if request.round_number == 5:
            retried.append(request.seller)
        return await noisy(request, label)

    run(recording, Journal(path, HEADER))
    assert retried == [1]


def test_cancelled_round_keeps_decisions_already_received(tmp_path):
    """Seller 0 has answered and seller 2 is still waiting when the process is cancelled (Colab stop,
    or another game's fatal error): seller 0's decision must already be on disk."""
    path, stuck = tmp_path / "g.jsonl", asyncio.Event()

    async def seller2_hangs_in_round_3(request, label):
        if request.round_number == 3 and request.seller == 2:
            await stuck.wait()  # never set
        if request.round_number == 3 and request.seller == 1:
            await asyncio.sleep(0.05)  # answers late, after the cancellation
        return await noisy(request, label)

    async def cancel_once_seller0_answered():
        task = asyncio.create_task(main_game(seller2_hangs_in_round_3, 3, Journal(path, HEADER), Market(3), RULES))
        while '"round": 3, "seller": 0' not in path.read_text():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_once_seller0_answered())
    saved = [e["seller"] for e in map(json.loads, path.read_text().splitlines())
             if e["event"] == "decision" and e["round"] == 3]
    assert saved == [0]
    asked = []

    async def recording(request, label):
        if request.round_number == 3:
            asked.append(request.seller)
        return await noisy(request, label)

    run(recording, Journal(path, HEADER))
    assert sorted(asked) == [1, 2]


def test_live_sellers_see_only_their_own_past(tmp_path):
    requests = []

    async def recording(request, label):
        requests.append((label, request))
        return await noisy(request, label)

    result = run(recording, Journal(tmp_path / "g.jsonl", HEADER))
    a = result["roles"]["A"]
    for label, r in requests:
        assert all(o.number < r.round_number for o in r.history)  # no future rounds
        assert len(r.history) <= RULES.window
        if r.round_number > 1:
            assert r.plans == f"plans {r.seller}"  # its own notes from the previous round
        if label.startswith(("00:", "10:", "01:", "11:")):
            assert r.seller != a  # A is programmed in every response branch
    first = {}
    for label, r in requests:  # at the first branch round, a seller's information is identical everywhere
        if r.round_number == RULES.parent_rounds + 1:
            first.setdefault(r.seller, set()).add(r)
    assert all(len(v) == 1 for v in first.values())


def test_ineligible_game_runs_no_branches(tmp_path):
    nash = Market(3).nash()
    calls = []

    async def competitive(request, label):
        calls.append(label)
        return Decision(nash)

    result = run(competitive, Journal(tmp_path / "g.jsonl", HEADER))
    assert result["status"] == "ineligible"
    assert len(calls) == RULES.planned_calls()["ineligible"]


def test_failed_game_is_reported_not_rerun(tmp_path):
    journal = Journal(tmp_path / "g.jsonl", HEADER)
    journal.fail("content attempts exhausted")

    async def must_not_run(request, label):
        raise AssertionError("a failed game must not be silently rerun")

    assert run(must_not_run, Journal(tmp_path / "g.jsonl", HEADER))["status"] == "failed"


def test_torn_final_line_is_dropped_and_game_resumes(tmp_path):
    path = tmp_path / "g.jsonl"
    control = run(noisy, Journal(tmp_path / "control.jsonl", HEADER))
    calls = []

    async def flaky(request, label):
        calls.append(label)
        if len(calls) == 50:
            raise ConnectionError
        return await noisy(request, label)

    with pytest.raises(ConnectionError):
        run(flaky, Journal(path, HEADER))
    with path.open("a") as f:
        f.write('{"event": "decision", "branch": "par')  # crash mid-write
    assert run(noisy, Journal(path, HEADER)) == control
    assert any(json.loads(line)["event"] == "torn_line_dropped" for line in path.read_text().splitlines())


def test_journal_refuses_other_configuration(tmp_path):
    Journal(tmp_path / "g.jsonl", HEADER)
    with pytest.raises(ValueError, match="different configuration"):
        Journal(tmp_path / "g.jsonl", {"test": "other"})


def test_invalid_decisions_and_rules_rejected():
    for bad in (float("nan"), float("inf"), -0.1, True, "1.5"):
        with pytest.raises(ValueError):
            Decision(bad)
    with pytest.raises(ValueError):
        replace(RULES, cut_rounds=21)
