import asyncio
import json
import math

import pytest
from test_client import CONFIG, FakeClient, ok

from sprint.analyze import competence_summary
from sprint.client import Budget, Interrupted
from sprint.competence import arithmetic_cases, parse_number, run_checks, score_answer
from sprint.experiments import MainRules
from sprint.prompt import BadReply
from sprint.run import attempts_logged, load_config


def test_arithmetic_cases_have_independent_known_answers():
    cases = {r["id"]: r for r in arithmetic_cases()}
    assert len(cases) == 20
    assert cases["profit_0"]["expected"] == pytest.approx(75.6)
    assert cases["profit_3"]["expected"] == pytest.approx(-5)
    assert cases["cut_0"]["expected"] == pytest.approx(5.04)
    assert cases["units_0"]["expected"] == pytest.approx(1.75)
    assert cases["discount_0"]["expected"] == pytest.approx(215.649)
    assert cases["gain_0"]["expected"] == pytest.approx(-8.44)
    assert score_answer(0, 75.6)["correct"] is False


def test_secondary_summaries_keep_strict_errors_and_show_late_learning():
    result = {"arithmetic": [
        {"status": "answered", **score_answer(215.65, 215.649)},
        {"status": "answered", **score_answer(5.06, 5.04)},
        {"status": "failed"},
    ], "pricing": [
        {"episode": 0, "status": "complete", "rounds": [
            {"round": r, "profit_gap": 10 if r <= 5 else 2} for r in range(1, 11)]},
        {"status": "failed"},
    ]}
    summary = competence_summary(result)
    assert summary["arithmetic"] == {"recorded": 3, "answered": 2, "failed": 1,
                                      "correct_strict": 0, "within_0_01": 1}
    assert summary["pricing"][0]["mean_profit_gap"] == 6
    assert summary["pricing"][1]["mean_profit_gap"] == 2
    assert summary["pricing"][2]["observed_rounds"] == 0
    assert summary["pricing"][2]["mean_profit_gap"] is None


@pytest.mark.parametrize("text", ['{"answer": true}', '{"answer": NaN}', '{"answer": "2"}',
                                 '{"answer": 2, "answer": 3}', '{"answer": 2} {"answer": 3}'])
def test_numeric_parser_rejects_ambiguous_or_non_numeric_replies(text):
    with pytest.raises(BadReply):
        parse_number(text)


def test_diagnostics_keep_wrong_answers_resume_and_start_pricing_with_empty_history(tmp_path):
    config = {"model": CONFIG, "rules": MainRules()}
    cases = arithmetic_cases()
    replies = [ok(json.dumps({"answer": 0 if i == 0 else c["expected"]})) for i, c in enumerate(cases)]
    client = FakeClient(replies + [ok()] * 20)
    with pytest.raises(Interrupted, match="budget"):
        asyncio.run(run_checks(config, tmp_path, Budget(23), client=client))
    result = asyncio.run(run_checks(config, tmp_path, Budget(40, attempts_logged(tmp_path)), client=client))
    assert len(client.sent) == 40  # includes the wrong answer exactly once; no paid repeats on resume
    assert result["arithmetic"][0]["answer"] == 0
    assert result["arithmetic"][0]["correct"] is False
    assert sum(r["correct"] for r in result["arithmetic"]) == 19
    for index in (20, 30):
        prompt = client.sent[index]["messages"][-1]["content"]
        assert "Round 1\n" not in prompt
        assert "No previous plans." in prompt
        assert "regardless of your actions" in prompt
    for episode in result["pricing"]:
        assert len(episode["rounds"]) == 10
        assert all(r["profit_gap"] >= -1e-10 for r in episode["rounds"])
        # Independent derivative of logit profit is zero at the computed optimum.
        x = episode["best_response"]
        rivals = (1.75, 1.75) if episode["episode"] == 0 else (1.60, 1.925)
        w = math.exp((2 - x) / .25)
        share = w / (1 + w + sum(math.exp((2 - p) / .25) for p in rivals))
        assert abs(1 - (x - 1) * (1 - share) / .25) < 1e-6
    resumed = asyncio.run(run_checks(config, tmp_path, Budget(40, 40), client=client))
    assert resumed == result and len(client.sent) == 40


@pytest.mark.parametrize("family", ["deepseek", "deepseek31", "qwen38", "mistral3", "gptoss120"])
def test_canary_and_pilot_share_model_settings(family):
    pilot = load_config(f"configs/main_pilot_{family}.toml")
    canary = load_config(f"configs/canary_{family}.toml")
    assert pilot["model"] == canary["model"]
    assert pilot["rules"] == MainRules()
    assert len(pilot["seeds"]) == 4
