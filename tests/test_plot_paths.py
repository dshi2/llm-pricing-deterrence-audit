import asyncio
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from scripts.check_fallback1_controls import control_policy
from scripts.plot_paths import export
from sprint.checks import SCENARIOS
from sprint.engine import Journal
from sprint.experiments import MainRules, main_game
from sprint.fallback1 import ReplacementRules, replacement_game
from sprint.market import Market


def test_main_paths_and_profit_gain_render_from_scripted_game(tmp_path):
    async def policy(request, _label):
        from sprint.scripted import as_policy
        return await as_policy(SCENARIOS["all_respond_mild"][request.seller])(request, _label)

    run = tmp_path / "main"
    run.mkdir()
    rules, market = MainRules(), Market(3)
    header = {"experiment": "main", "game_seed": 7, "rules": asdict(rules),
              "market": asdict(market), "model": "scripted", "code_sha": "scripted"}
    asyncio.run(main_game(policy, 7, Journal(run / "game_7.jsonl", header), market=market, rules=rules))
    out = tmp_path / "figures"
    export("main", [run], out, "SCRIPTED CONTROL — not live results")
    assert list(out.glob("main_paths_*.png"))
    assert list(out.glob("main_profit_gain_*.png"))
    assert (out / "late_profit_gain.csv").exists()


def test_f1_paths_cells_and_observation_access_render_from_scripted_game(tmp_path):
    run = tmp_path / "f1"
    run.mkdir()
    rules = ReplacementRules(parent_rounds=10, restraint_window=5, horizon=6)
    header = {"experiment": "replacement", "game_seed": 8, "rules": asdict(rules),
              "market": asdict(Market(2)), "model": "scripted", "code_sha": "scripted"}
    asyncio.run(replacement_game(control_policy("price_responsive", rules), 8,
                                 Journal(run / "game_8.jsonl", header), rules=rules))
    out = tmp_path / "figures"
    export("f1", [run], out, "SCRIPTED CONTROL — not live results")
    assert list(out.glob("f1_paths_*.png"))
    assert list(out.glob("f1_mean_paths_*.png"))
    assert list(out.glob("f1_cells_*.png"))
    assert (out / "observation_opportunity.csv").exists()



def test_f1_mean_paths_leave_out_an_interrupted_path(tmp_path):
    run = tmp_path / "f1"
    run.mkdir()
    rules = ReplacementRules(parent_rounds=10, restraint_window=5, horizon=6)
    header = {"experiment": "replacement", "game_seed": 8, "rules": asdict(rules),
              "market": asdict(Market(2)), "model": "scripted", "code_sha": "scripted"}
    journal = run / "game_8.jsonl"
    asyncio.run(replacement_game(control_policy("price_responsive", rules), 8, Journal(journal, header), rules=rules))
    kept, branch_rounds = [], 0
    for line in journal.read_text().splitlines():  # interrupted after 3 of 6 rounds of the first arm
        event = json.loads(line)
        branch_rounds += event["event"] == "round" and event["branch"] != "parent"
        if branch_rounds > 3:
            break
        kept.append(line)
    journal.write_text("\n".join(kept) + "\n")
    out = tmp_path / "figures"
    export("f1", [run], out, "SCRIPTED CONTROL — not live results")
    paths = pd.read_csv(out / "price_paths.csv")
    assert list(paths.branch.unique()) == ["fresh_fixed"] and len(paths) == 3 and not paths.complete.any()
    assert list(out.glob("f1_paths_*.png"))  # the partial path stays visible per game
    assert not list(out.glob("f1_mean_paths_*.png"))  # but is never averaged as a complete one

@pytest.mark.parametrize("script", ["plot_paths", "report_figures", "check_covert_controls", "check_fallback1_controls"])
def test_scripts_run_directly_from_any_folder(tmp_path, script):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{script}.py"
    subprocess.run([sys.executable, str(path), "--help"], cwd=tmp_path, check=True, capture_output=True)
