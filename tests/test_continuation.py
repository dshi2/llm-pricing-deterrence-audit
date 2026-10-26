# ruff: noqa: S102 -- Execute our generated notebook cells with fake I/O; no API or network access.
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from test_client import CONFIG
from test_run import CANARY, canary

from scripts.build_parallel_notebooks import CUTOFF, build
from sprint.continuation import FAMILIES, allocate, config_text, measure, pilot_status, prepare, reply_rounds
from sprint.engine import Journal
from sprint.experiments import MainRules, main_game
from sprint.market import Market
from sprint.run import header, load_config

FAMILY = "qwen38"


def cutoff(hours):
    return (datetime.now(UTC) + timedelta(hours=hours)).isoformat()


def age(root, minutes):
    then = time.time() - 60 * minutes
    for path in root.rglob("*"):
        os.utime(path, (then, then))


def finished_pilot(tmp_path):
    """A pilot that finished an hour ago: canary, probe and one eligible-or-ineligible game (seed 9)."""
    source, probe = canary(tmp_path / "fixture")
    root = tmp_path / "runs/parallel_v1/all"
    for name in (f"canary_{FAMILY}", f"main_{FAMILY}"):
        shutil.copytree(source, root / name)
    shutil.copytree(probe, root / f"main_{FAMILY}_probe")
    configs = tmp_path / "configs"
    configs.mkdir()
    pilot = {"name": f"main_{FAMILY}", "rules": CANARY, "model": CONFIG}
    (configs / f"deterrence_pilot_{FAMILY}.toml").write_text(config_text(pilot, [9], 4))
    age(root, 60)
    return root, configs


@pytest.mark.parametrize("family", FAMILIES)
def test_continuation_is_the_pilot_protocol_in_the_same_folder(tmp_path, family):
    pilot = load_config(f"configs/deterrence_pilot_{family}.toml")
    path = tmp_path / "continue.toml"
    path.write_text(config_text(pilot, list(range(1, 13))))
    more = load_config(path)
    assert more["name"] == pilot["name"]  # same run directory, so resume skips the finished pilot games
    assert more["seeds"][:4] == pilot["seeds"] == [1, 2, 3, 4]
    assert {k: v for k, v in more.items() if k not in ("seeds", "concurrency")} == {
        k: v for k, v in pilot.items() if k not in ("seeds", "concurrency")}
    assert header(more, 1, False) == header(pilot, 1, False)
    assert more["rules"].planned_calls() == {"eligible": 710, "ineligible": 510}


def test_allocation_takes_turns_cheapest_first_within_caps():
    seeds, spend = allocate({"a": 1, "b": 3}, {"a": 6, "b": 10}, 10, {"a": 4, "b": 4})
    assert seeds == {"a": 6, "b": 6}
    assert spend == 8
    assert allocate({"a": 1}, {"a": 4}, 100, {"a": 4}) == ({"a": 4}, 0.0)


def test_status_waits_for_running_or_settling_pilots(tmp_path):
    assert pilot_status(tmp_path / "not_started", [FAMILY], "configs")["state"] == "running"
    root, configs = finished_pilot(tmp_path)
    assert pilot_status(root, [FAMILY], configs)["state"] == "finished"
    (root / f"{FAMILY}.console.log").write_text("pilot report still writing")
    assert pilot_status(root, [FAMILY], configs)["state"] == "running"
    journal = root / f"main_{FAMILY}/game_9.jsonl"
    journal.write_text("".join(line + "\n" for line in journal.read_text().splitlines()
                               if json.loads(line)["event"] != "complete"))
    age(root, 12)
    assert pilot_status(root, [FAMILY], configs)["state"] == "running"  # unfinished and recently active
    age(root, 20)
    status = pilot_status(root, [FAMILY], configs)
    assert status["state"] == "stopped" and status["unfinished"] == [FAMILY]
    with pytest.raises(ValueError, match="Pilots not finished"):
        prepare(root, [FAMILY], 50, cutoff(10), configs)


def test_plan_is_frozen_once_and_never_resized(tmp_path):
    root, configs = finished_pilot(tmp_path)
    plan = prepare(root, [FAMILY], 50, cutoff(10), configs)
    entry = plan["families"][FAMILY]
    assert entry["status"] == "continued"
    assert entry["seeds"][0] == 9 and entry["seeds"][1:4] == [10, 11, 12]
    assert entry["attempt_budget"] == 875 * len(entry["seeds"])
    assert plan["projected_new_cost_usd_not_guaranteed"] <= 0.9 * 50
    assert prepare(root, [FAMILY], 0.01, cutoff(-1), configs) == plan
    frozen = root / f"configs/main_continue_{FAMILY}.toml"
    assert load_config(frozen)["seeds"] == entry["seeds"]
    frozen.write_text(frozen.read_text().replace("temperature", "top_p"))
    with pytest.raises(ValueError, match="Frozen config modified"):
        prepare(root, [FAMILY], 50, cutoff(10), configs)


@pytest.mark.parametrize("hours, money, status", [(-1, 50, "pilot only"), (10, 0.0001, "pilot only")])
def test_no_time_or_money_leaves_pilot_only(tmp_path, hours, money, status):
    root, configs = finished_pilot(tmp_path)
    entry = prepare(root, [FAMILY], money, cutoff(hours), configs)["families"][FAMILY]
    assert entry["status"].startswith(status) and entry["seeds"] == [9]


def test_failed_pilot_game_or_changed_protocol_is_not_continued(tmp_path):
    root, configs = finished_pilot(tmp_path)
    journal = root / f"main_{FAMILY}/game_9.jsonl"
    lines = [line for line in journal.read_text().splitlines() if json.loads(line)["event"] != "complete"]
    journal.write_text("\n".join([*lines, json.dumps({"event": "failed", "reason": "unusable"})]) + "\n")
    age(root, 60)
    entry = prepare(root, [FAMILY], 50, cutoff(10), configs)["families"][FAMILY]
    assert entry["status"] == "excluded: failed pilot game"
    root, configs = finished_pilot(tmp_path / "changed")
    path = configs / f"deterrence_pilot_{FAMILY}.toml"
    path.write_text(re.sub(r"(?m)^temperature = .*$", "temperature = 0.123", path.read_text()))
    with pytest.raises(ValueError, match="not the same protocol"):
        prepare(root, [FAMILY], 50, cutoff(10), configs)



@pytest.mark.parametrize("throttled, concurrency", [(False, 8), (True, 4)])
def test_full_run_doubles_concurrency_only_if_the_pilot_was_not_rate_limited(tmp_path, throttled, concurrency):
    root, configs = finished_pilot(tmp_path)
    log = root / f"main_{FAMILY}/game_9.calls.jsonl"
    calls = log.read_text().splitlines()
    if throttled:  # about 9% of attempts answered 429
        log.write_text(log.read_text() + (json.dumps(json.loads(calls[-1]) | {"http_status": 429, "cost": None}) + "\n")
                       * (len(calls) // 10 + 1))
        age(root, 60)
    plan = prepare(root, [FAMILY], 50, cutoff(10), configs)
    assert (plan["pilot_measurements"][FAMILY]["rate_limited_share"] > 0.01) == throttled
    assert load_config(root / f"configs/main_continue_{FAMILY}.toml")["concurrency"] == concurrency


def test_game_time_is_a_full_eligible_game_at_the_slowest_pilot_pace(tmp_path):
    async def policy(request, label):
        from sprint.checks import SCENARIOS
        from sprint.scripted import as_policy
        return await as_policy(SCENARIOS["all_respond_mild"][request.seller])(request, label)

    rules = MainRules()
    head = {"experiment": "main", "game_seed": 7, "rules": asdict(rules), "market": asdict(Market(3)), "model": "s"}
    asyncio.run(main_game(policy, 7, Journal(tmp_path / "game.jsonl", head), Market(3), rules))
    events = [json.loads(line) for line in (tmp_path / "game.jsonl").read_text().splitlines()]
    assert events[-1]["result"]["status"] == "eligible"
    assert reply_rounds(rules) == len({(e["branch"], e["round"]) for e in events if e["event"] == "decision"}) == 310
    root, configs = finished_pilot(tmp_path / "pilot")  # a short pilot game is scaled up, never taken as is
    pilot = load_config(configs / f"deterrence_pilot_{FAMILY}.toml")
    played = {(e["branch"], e["round"]) for e in map(json.loads, (root / f"main_{FAMILY}/game_9.jsonl").read_text().splitlines())
              if e["event"] == "decision"}
    m = measure(root, pilot)
    assert m["game_hours"] == pytest.approx(m["slowest_pilot_game_hours"] / len(played) * reply_rounds(pilot["rules"]))
    assert m["game_hours"] >= m["slowest_pilot_game_hours"]

def notebook_cells():
    notebook = build(json.loads(Path("notebooks/run_models.ipynb").read_text()), "all", "a" * 40, "continue")
    return ["".join(c["source"]) for c in notebook["cells"]]


def test_continue_notebook_reuses_pilot_folder_key_and_gates():
    cells = notebook_cells()
    source = "\n".join(cells)
    assert "runs/parallel_v1/all" in source and "collusion-continue.bundle" in source
    assert "userdata.get('COLLUSION_OPENROUTER_API_KEY')" in source
    assert f"COLLECTION_CUTOFF = '{CUTOFF}'" in source
    assert source.index("prepare(OUT") < source.index("pool.submit") < source.index("'pooled'")
    assert "if not globals().get('OFFLINE_PASSED') or not globals().get('BUDGET_VERIFIED')" in cells[6]
    with pytest.raises(ValueError):
        build(json.loads(Path("notebooks/run_models.ipynb").read_text()), FAMILY, "a" * 40, "continue")



def test_allowance_is_read_after_the_pilots_stop_spending(tmp_path, monkeypatch):
    from sprint import continuation
    allowance, seen, states = {"usd": 90}, {}, iter(["running", "finished"])

    def status(root, families):
        state = next(states)
        if state == "finished":
            allowance["usd"] = 50  # the pilot spent $40 while this notebook waited
        return {"state": state, "families": {}}

    def freeze(root, families, usd, cutoff):
        seen["usd"] = usd
        return {"families": {}, "projected_new_cost_usd_not_guaranteed": 0}

    monkeypatch.setattr(continuation, "pilot_status", status)
    monkeypatch.setattr(continuation, "prepare", freeze)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    exec(notebook_cells()[5], {"OUT": tmp_path, "FAMILIES": [FAMILY], "COLLECTION_CUTOFF": cutoff(10),
                               "key_budget": lambda: {"remaining_allowance_usd": allowance["usd"]},
                               "print": lambda *args, **kwargs: None})
    assert seen["usd"] == 50


def complete_games(root, seeds):
    """Stand-in for a finished sprint.run: each new seed replays the pilot game's saved journal."""
    lines = (root / f"main_{FAMILY}/game_9.jsonl").read_text().splitlines()
    for seed in seeds:
        path = root / f"main_{FAMILY}/game_{seed}.jsonl"
        if not path.exists():
            path.write_text("\n".join([json.dumps(json.loads(lines[0]) | {"game_seed": seed}), *lines[1:]]) + "\n")

def test_freeze_launch_report_and_resume_run_end_to_end(tmp_path, monkeypatch):
    root, _ = finished_pilot(tmp_path)
    freeze, launch, report = notebook_cells()[5:]
    report = report.replace("from IPython.display import Image\n", "Image = dict\n")  # Colab-only import
    monkeypatch.chdir(tmp_path)  # prepare reads configs/deterrence_pilot_*.toml from here
    calls, snapshots = [], []

    def run(args, **kwargs):  # the first pass exits normally with nothing done; the resume finishes
        calls.append(args)
        if len(calls) == 2:
            complete_games(root, scope["PLAN"]["families"][FAMILY]["seeds"])

    scope = {"OUT": root, "FAMILIES": [FAMILY], "key_budget": lambda: {"remaining_allowance_usd": 50}, "Path": Path,
             "COLLECTION_CUTOFF": cutoff(10), "OFFLINE_PASSED": True, "BUDGET_VERIFIED": True,
             "sh": run, "sys": sys, "subprocess": subprocess,
             "load_config": load_config, "endpoint_snapshot": lambda config: snapshots.append(config["name"]),
             "display": lambda *args: None, "print": lambda *args, **kwargs: None}
    exec(freeze, scope)
    rehearsal = pd.read_csv(root / "analysis_pilot_rehearsal/games.csv")
    assert list(rehearsal.seed) == [9]  # the full-run report, rehearsed on the real pilot before any new game
    assert list((root / "figures_pilot_rehearsal").glob("deterrence_primary_*.png"))
    exec(launch, scope)
    assert "Rerun to resume" in scope["failures"][FAMILY]  # a normal exit is not completion
    with pytest.raises(RuntimeError, match="Some families stopped"):
        exec(report, scope)
    games = scope["table"].set_index("seed").status
    assert games[9] in ("eligible", "ineligible")
    assert set(games.drop(9)) == {"not_started"}  # reported, not hidden
    exec(freeze, scope)  # resume: same plan, same caps
    exec(launch, scope)
    assert scope["failures"] == {}
    budget = str(scope["PLAN"]["families"][FAMILY]["attempt_budget"])
    assert len(calls) == 2 and all(call[call.index("--budget") + 1] == budget for call in calls)
    assert all(str(root / f"configs/main_continue_{FAMILY}.toml") in call for call in calls)
    assert snapshots == [f"main_{FAMILY}"] * 4
    exec(report, scope)
    assert len(scope["table"]) == len(scope["PLAN"]["families"][FAMILY]["seeds"])
    assert set(scope["table"].status) <= {"eligible", "ineligible"}
    assert (root / "analysis_pooled/analysis.json").exists()
    assert list((root / "figures_pooled").glob("design_deterrence.png"))
    assert asdict(load_config(root / f"configs/main_continue_{FAMILY}.toml")["model"]) == asdict(CONFIG)
