import asyncio
import json
from dataclasses import replace

import pytest
from test_client import CONFIG, FakeClient, ok, status

from scripts.check_covert_controls import compatible_answers, run_controls
from sprint.client import Budget, Fatal, Interrupted
from sprint.covert import INPUTS, QUESTIONS, VIEWS, cases, evidence, execute, messages, parse_reply
from sprint.covert_analyze import export_tables, judgments, launch_checks, summary
from sprint.covert_run import run_batch
from sprint.engine import Journal
from sprint.prompt import BadReply
from sprint.run import attempts_logged


def config():
    return {"model": replace(CONFIG, max_tokens=2000), "seeds": [99],
            "scheme": "direct", "concurrency": 1}


VALID = json.dumps({"recovered": "contradicted", "dependent": "supported", "evidence": []})


@pytest.mark.parametrize("scheme", ["direct", "affine"])
def test_matched_pair_does_not_leak_hidden_use(scheme):
    for seed in range(101, 121):
        used = execute(seed, scheme, "used")
        ignored = execute(seed, scheme, "ignored_matching")
        assert used["truth"]["dependent"] and not ignored["truth"]["dependent"]
        for view in ("public", "decoder"):
            assert messages(evidence(used, view)) == messages(evidence(ignored, view))
        assert [r["input"] for r in used["trials"]] == [r["input"] for r in ignored["trials"]]
        assert all(type(r["A_minor"]) is int for r in used["trace"])


def test_labels_independently_follow_compatible_mechanisms(tmp_path):
    table = run_controls(tmp_path)
    assert table.correct.count() == 1600
    for case in cases(range(101, 105), "affine"):
        assert compatible_answers(case["record"]) == case["expected"]
    cases_partial = [c for c in cases(range(101, 121), "affine")
                     if c["view"] == "partial_execution" and c["kind"] == "used"]
    assert {c["expected"]["dependent"] for c in cases_partial} == {"supported", "insufficient_evidence"}
    assert all({r["input"] for r in c["record"]["policy_trials"]} == set(INPUTS) for c in cases_partial)


def test_wrong_answers_are_valid_and_labels_absent_from_prompt():
    assert parse_reply(VALID)["dependent"] == "supported"
    for case in cases([99], "direct"):
        rendered = json.dumps(messages(case["record"]))
        assert case["id"] not in rendered
        assert '"truth"' not in rendered and '"expected"' not in rendered


@pytest.mark.parametrize("text", [
    '{}', '{"recovered":"yes","dependent":"supported","evidence":[]}',
    VALID[:-1] + ',"extra":0}', VALID[:-1] + ',"dependent":"contradicted"}',
    VALID + VALID, VALID.replace('"evidence": []', '"evidence": "x"'),
])
def test_parser_rejects_ambiguous_or_malformed_reply(text):
    with pytest.raises(BadReply):
        parse_reply(text)


def test_budget_resume_and_failed_reply_accounting(tmp_path):
    client = FakeClient([ok(VALID)] * 20)
    first = asyncio.run(run_batch(config(), tmp_path, Budget(2), client=client))
    assert sum(r["status"] == "answered" for r in first) == 2
    before = judgments([tmp_path])
    assert len(before) == 40 and before.answered.sum() == 4
    resumed = asyncio.run(run_batch(config(), tmp_path, Budget(20, attempts_logged(tmp_path)), client=client))
    assert all(r["status"] == "answered" for r in resumed)
    assert len(client.sent) == 20
    assert all(launch_checks(tmp_path).values())
    # No success-dependent resampling: knowingly wrong judgments still used only one call each.
    table = export_tables([tmp_path], tmp_path)
    assert table.correct.sum() < 40
    assert len(summary(table)) == len(VIEWS) * len(QUESTIONS) * 7
    assert summary(table).blocks.max() == 1
    assert table.drop_duplicates("case_id").cost.sum() == pytest.approx(.008)
    asyncio.run(run_batch(config(), tmp_path, Budget(20, 20), client=client))
    assert len(client.sent) == 20
    with pytest.raises(ValueError, match="configuration changed"):
        asyncio.run(run_batch(config() | {"seeds": [100]}, tmp_path, Budget(20), client=client))


def test_saved_api_answer_reused_after_interruption(tmp_path, monkeypatch):
    original = Journal.finish
    interrupted = False

    def fail_once(self, result):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise Interrupted("injected between call log and journal completion")
        return original(self, result)

    monkeypatch.setattr(Journal, "finish", fail_once)
    client = FakeClient([ok(VALID)] * 20)
    asyncio.run(run_batch(config(), tmp_path, Budget(20), client=client))
    assert judgments([tmp_path]).answered.sum() == 38
    asyncio.run(run_batch(config(), tmp_path, Budget(20, 20), client=client))
    assert judgments([tmp_path]).answered.all()
    assert len(client.sent) == 20


def test_refusal_is_retained_without_retry(tmp_path):
    client = FakeClient([ok("I cannot participate.")] + [ok(VALID)] * 19)
    result = asyncio.run(run_batch(config(), tmp_path, Budget(20), client=client))
    assert sum(r["status"] == "failed" for r in result) == 1
    assert len(client.sent) == 20
    assert not all(launch_checks(tmp_path).values())
    assert (judgments([tmp_path]).status == "failed").sum() == 2


@pytest.mark.parametrize("code", [401, 402])
def test_fatal_keeps_planned_and_unstarted_cases(tmp_path, code):
    client = FakeClient([status(code)] + [ok(VALID)] * 20)
    with pytest.raises(Fatal):
        asyncio.run(run_batch(config(), tmp_path, Budget(20), client=client))
    table = judgments([tmp_path])
    assert len(table) == 40
    assert not table.answered.all()
    assert attempts_logged(tmp_path) >= 1


def test_stub_end_to_end_and_torn_last_line(tmp_path):
    asyncio.run(run_batch(config(), tmp_path, Budget(0), stub="reference"))
    path = next(tmp_path.glob("case_*.jsonl"))
    with path.open("a") as log:
        log.write('{"event":')
    table = export_tables([tmp_path], tmp_path)
    assert len(table) == 40 and table.correct.all()
    assert (tmp_path / "case_type_scores.csv").exists()
    assert summary(table).ci_low.isna().all()


@pytest.mark.parametrize("stub", ["always_accuse", "always_abstain"])
def test_negative_controls_fail_through_runner_and_analysis(tmp_path, stub):
    asyncio.run(run_batch(config(), tmp_path, Budget(0), stub=stub))
    table = export_tables([tmp_path], tmp_path)
    assert len(table) == 40 and not table.correct.all()
