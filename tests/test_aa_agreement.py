import asyncio
import json
from dataclasses import asdict

import pandas as pd

from scripts.aa_agreement import REPEAT, agreement, summary
from sprint.checks import SCENARIOS
from sprint.engine import Journal
from sprint.experiments import MainRules, main_game
from sprint.market import Market


def scripted_game(run):
    async def policy(request, _label):
        from sprint.scripted import as_policy
        return await as_policy(SCENARIOS["all_respond_mild"][request.seller])(request, _label)

    run.mkdir()
    rules, market = MainRules(), Market(3)
    header = {"experiment": "main", "game_seed": 7, "rules": asdict(rules),
              "market": asdict(market), "model": "scripted", "code_sha": "scripted"}
    asyncio.run(main_game(policy, 7, Journal(run / "game_7.jsonl", header), market=market, rules=rules))
    return run / "game_7.jsonl"


def test_deterministic_repeat_matches_throughout(tmp_path):
    scripted_game(tmp_path / "main_x")
    table = agreement([tmp_path / "main_x"])
    row = table.iloc[0]
    assert row.rivals == 2 and row.rounds_compared == MainRules().horizon
    assert row.identical_rounds_before_first_difference == MainRules().horizon
    assert pd.isna(row.first_difference_round)
    assert summary(table).first_round_identical_reply_share.item() == 1.0


def test_comparison_stops_at_first_differing_reply(tmp_path):
    path = scripted_game(tmp_path / "main_x")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    rounds = sorted({e["round"] for e in events if e.get("branch") == REPEAT and e["event"] == "decision"})
    for e in events:  # one rival's notes differ in the third repeat round; its price does not
        if e.get("branch") == REPEAT and e["event"] == "decision" and e["round"] == rounds[2]:
            e["decision"]["plans"] += " changed"
            break
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    row = agreement([tmp_path / "main_x"]).iloc[0]
    assert row.first_difference_round == 3
    assert row.identical_rounds_before_first_difference == 2
    assert row.first_round_identical_replies == 2


def test_game_cut_off_in_first_repeat_round_is_skipped(tmp_path):
    path = scripted_game(tmp_path / "main_x")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    cut = next(i for i, e in enumerate(events) if e.get("branch") == REPEAT and e["event"] == "decision")
    path.write_text("".join(json.dumps(e) + "\n" for e in events[:cut + 1]))  # one rival replied, then credit ran out
    table = agreement([tmp_path / "main_x"])
    assert table.empty and summary(table).empty
