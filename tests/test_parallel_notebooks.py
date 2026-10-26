# ruff: noqa: S102 -- Execute our fixed notebook cells with fake I/O to test failure gates.
import json
from pathlib import Path

import pytest

from scripts.build_parallel_notebooks import FAMILIES, LAUNCH_GUARD, OFFLINE_CHECK, build


def test_launch_copies_share_code_but_isolate_outputs():
    original = Path("notebooks/run_models.ipynb").read_text()
    template = json.loads(original)
    commit = "a" * 40
    for family in FAMILIES:
        notebook = build(template, family, commit)
        setup = "".join(notebook["cells"][1]["source"])
        assert f"FAMILIES = ['{family}']" in setup
        assert f"runs/parallel_v1/{family}" in setup
        assert f"/content/collusion-parallel-{family}" in setup
        assert f"COMMIT = '{commit}'" in setup
        assert "userdata.get('COLLUSION_OPENROUTER_API_KEY')" in setup
        assert "userdata.get('OPENROUTER_API_KEY')" not in setup
        assert "collusion-parallel.bundle" in setup
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), family, "exec")
                assert cell["outputs"] == []
        launch = "".join(notebook["cells"][5]["source"])
        assert launch.startswith(LAUNCH_GUARD)
        assert "command('sprint.run', pilot, 3500)" in launch
    assert template == json.loads(original)


def test_combined_launcher_preserves_bounded_concurrency_and_new_secret():
    template = json.loads(Path("notebooks/run_models.ipynb").read_text())
    notebook = build(template, "all", "a" * 40)
    source = "".join("".join(c["source"]) for c in notebook["cells"])
    assert f"FAMILIES = {list(FAMILIES)!r}" in source
    assert "ThreadPoolExecutor(max_workers=4)" in source
    assert "userdata.get('COLLUSION_OPENROUTER_API_KEY')" in source
    assert "userdata.get('OPENROUTER_API_KEY')" not in source
    assert "runs/parallel_v1/all" in source


@pytest.mark.parametrize("offline,budget", [(False, False), (False, True), (True, False)])
def test_paid_calls_blocked_when_prior_cell_failed(offline, budget):
    with pytest.raises(RuntimeError, match="before model calls"):
        exec(LAUNCH_GUARD, {"OFFLINE_PASSED": offline, "BUDGET_VERIFIED": budget})
    exec(LAUNCH_GUARD, {"OFFLINE_PASSED": True, "BUDGET_VERIFIED": True})


def offline_scope(tmp_path, sh):
    import os
    import sys

    out = tmp_path / "runs/parallel_v1/qwen38"
    out.mkdir(parents=True)
    return {"OUT": out, "FAMILIES": ["qwen38"], "sh": sh, "sys": sys, "os": os}


def test_failed_offline_tests_invalidate_previous_success(tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("test failure")

    scope = offline_scope(tmp_path, fail)
    scope["OFFLINE_PASSED"] = True
    with pytest.raises(RuntimeError, match="test failure"):
        exec(OFFLINE_CHECK, scope)
    assert scope["OFFLINE_PASSED"] is False


def test_other_pilot_blocks_launch_but_own_resume_is_allowed(tmp_path):
    scope = offline_scope(tmp_path, lambda *args, **kwargs: None)
    own = scope["OUT"] / "main_qwen38/game_1.jsonl"
    own.parent.mkdir()
    own.write_text("saved own game")
    exec(OFFLINE_CHECK, scope)
    assert scope["OFFLINE_PASSED"] is True
    other = tmp_path / "runs/four_models_v1/main_qwen38/game_2.calls.jsonl"
    other.parent.mkdir(parents=True)
    other.write_text("saved attempts")
    with pytest.raises(RuntimeError, match="Existing qwen38 pilot"):
        exec(OFFLINE_CHECK, scope)
    assert scope["OFFLINE_PASSED"] is False
    assert own.read_text() == "saved own game"
    assert other.read_text() == "saved attempts"
