import asyncio
import json
from dataclasses import asdict

import pandas as pd
import pytest

from scripts.run_saved_report import hashes, run_report


@pytest.mark.parametrize("kind", ["main", "f1", "f2"])
def test_saved_report_end_to_end(tmp_path, kind):
    from scripts.check_fallback1_controls import control_policy
    from sprint.checks import SCENARIOS
    from sprint.covert import cases
    from sprint.engine import Journal
    from sprint.experiments import MainRules, main_game
    from sprint.fallback1 import ReplacementRules, replacement_game
    from sprint.market import Market
    from sprint.scripted import as_policy

    source = tmp_path / kind
    source.mkdir()
    if kind == "f2":
        planned = cases([101, 102], "affine")
        (source / "manifest.json").write_text(json.dumps({"model": {"model": "scripted"}, "cases": planned}))
        # One unanswered case must remain in the planned denominator.
        for case in planned[1:]:
            (source / f"case_{case['id']}.jsonl").write_text(json.dumps(
                {"event": "complete", "result": {"reply": case["expected"]}}) + "\n")
    else:
        rules = MainRules() if kind == "main" else ReplacementRules(parent_rounds=10, restraint_window=5, horizon=6)
        market = Market(3 if kind == "main" else 2)
        header = {"experiment": "main" if kind == "main" else "replacement", "game_seed": 1,
                  "rules": asdict(rules), "market": asdict(market), "model": "scripted", "code_sha": "scripted"}
        journal = Journal(source / "game_1.jsonl", header)
        if kind == "main":
            async def policy(request, label):
                return await as_policy(SCENARIOS["all_respond_mild"][request.seller])(request, label)
            asyncio.run(main_game(policy, 1, journal, market=market, rules=rules))
        else:
            asyncio.run(replacement_game(control_policy("price_responsive", rules), 1, journal, rules=rules))
    before = hashes([source])
    out = tmp_path / "report"
    run_report(kind, [source], out, label="SCRIPTED TEST DATA")
    manifest = json.loads((out / "report_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["inputs"] == before == hashes([source])
    assert set(manifest["environment"]["packages"]) == {"numpy", "pandas", "scipy", "matplotlib"}
    assert list((out / "figures").rglob("*.png"))
    assert list((out / "tables").glob("*.csv"))
    if kind == "f2":
        table = pd.read_csv(out / "tables" / "judgments.csv")
        assert len(table) == 80 and table.answered.sum() == 78
        assert list((out / "figures" / "metrics").glob("*.pdf"))
    with pytest.raises(FileExistsError):
        run_report(kind, [source], out)


def test_report_rejects_overlapping_input_output(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        run_report("main", [tmp_path], tmp_path / "output")
