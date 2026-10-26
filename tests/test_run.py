"""Launch safeguards: the budget is a total per run directory, the context probe is long, and the pilot
launch checks pass on a clean canary and fail on a bad one."""

import asyncio
import json

import pytest
from test_client import CONFIG, FakeClient, ok

from sprint.analyze import launch_checks
from sprint.client import Budget, Interrupted, LLMPolicy
from sprint.engine import Journal
from sprint.experiments import MainRules, main_game
from sprint.market import Market
from sprint.prompt import messages
from sprint.run import attempts_logged, header, probe_request

CANARY = MainRules(parent_rounds=5, horizon=2, cut_rounds=1)


def test_budget_counts_attempts_already_logged(tmp_path):
    for name, rows in (("game_1.calls.jsonl", 4), ("probe_x.calls.jsonl", 2)):
        (tmp_path / name).write_text("{}\n" * rows)
    (tmp_path / "game_1.jsonl").write_text("{}\n" * 50)  # journals are not attempts
    budget = Budget(7, used=attempts_logged(tmp_path))
    budget.take()
    with pytest.raises(Interrupted, match="budget"):
        budget.take()


def test_probe_prompt_has_a_full_window_and_reply_sized_notes():
    config = {"rules": MainRules(), "model": CONFIG}
    request = probe_request(config)
    assert len(request.history) == MainRules().window and len(request.history[0].prices) == 3
    assert len(request.plans) >= 4 * CONFIG.max_tokens - 1 and len(request.insights) >= 4 * CONFIG.max_tokens - 1
    assert request.plans.startswith("Hold near 5.60") and request.insights.startswith("Rounds at 5.55")
    body = messages(request, Market(3))[2]["content"]
    assert body.count("Round ") == MainRules().window and request.plans in body and request.insights in body


def canary(tmp_path, provider="DeepInfra"):
    """A canary game and a probe answered by a fake client; returns the two run directories."""
    canary_dir, probe_dir = tmp_path / "canary", tmp_path / "probe"
    policy = LLMPolicy(CONFIG, Market(3), Budget(100), canary_dir / "game_9.calls.jsonl",
                       client=FakeClient([ok(provider=provider)] * 100))
    config = {"experiment": "main", "rules": CANARY, "model": CONFIG}
    asyncio.run(main_game(policy, 9, Journal(canary_dir / "game_9.jsonl", header(config, 9, stub=False)),
                          Market(3), CANARY))
    probe = LLMPolicy(CONFIG, Market(3), Budget(1), probe_dir / "probe_1.calls.jsonl", client=FakeClient([ok()]))
    asyncio.run(probe(probe_request({"rules": MainRules(), "model": CONFIG}), "context_probe"))
    return canary_dir, probe_dir


def test_launch_checks_pass_on_a_clean_canary(tmp_path):
    checks = launch_checks(*canary(tmp_path))
    assert all(checks.values()), checks


@pytest.mark.parametrize("bad, failing", [
    ({"outcome": "declined"}, "no fatal, declined or error attempts"),
    ({"outcome": "ok", "provider_served": "SiliconFlow"}, "every reply served by the pinned provider"),
    ({"outcome": "ok", "cost": None}, "cost recorded for every reply"),
])
def test_launch_checks_fail_on_a_bad_attempt(tmp_path, bad, failing):
    canary_dir, probe_dir = canary(tmp_path)
    log = canary_dir / "game_9.calls.jsonl"
    row = json.loads(log.read_text().splitlines()[0])
    with log.open("a") as f:
        f.write(json.dumps(row | bad) + "\n")
    checks = launch_checks(canary_dir, probe_dir)
    assert [name for name, passed in checks.items() if not passed] == [failing]


def test_launch_checks_fail_without_a_finished_canary_or_probe(tmp_path):
    canary_dir, probe_dir = canary(tmp_path)
    for path in probe_dir.iterdir():
        path.unlink()
    assert not launch_checks(canary_dir, probe_dir)["context probe answered and parsed"]
    journal = canary_dir / "game_9.jsonl"
    journal.write_text("".join(line + "\n" for line in journal.read_text().splitlines()
                               if json.loads(line)["event"] != "complete"))
    assert not launch_checks(canary_dir, probe_dir)["canary game completed"]
