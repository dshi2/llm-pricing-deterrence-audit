"""The new family uses the existing request path and a separate collection directory."""

import json
from pathlib import Path

from test_client import FakeClient, call, ok

from sprint.client import Budget, LLMPolicy
from sprint.market import Market
from sprint.run import load_config


def test_new_family_preserves_experiment_and_canary_settings():
    for kind in ("deterrence_pilot", "canary"):
        old = load_config(f"configs/{kind}_deepseek31.toml")
        new = load_config(f"configs/{kind}_deepseek41.toml")
        for key in ("experiment", "seeds", "concurrency", "rules"):
            assert new[key] == old[key]
        assert new["model"].prefix == old["model"].prefix
        assert new["model"].units == old["model"].units
        assert new["model"].temperature == old["model"].temperature
        assert new["model"].reasoning_effort == "high"
        assert new["model"].reasoning_max_tokens is None


def test_new_family_request_is_pinned_and_explicit(tmp_path):
    config = load_config("configs/deterrence_pilot_deepseek41.toml")["model"]
    client = FakeClient([ok()])
    call(LLMPolicy(config, Market(3), Budget(1), tmp_path / "calls.jsonl", client=client))
    request = client.sent[0]
    assert request["model"] == "deepseek/deepseek-v4.1-flash-1017"
    assert request["max_tokens"] == 8192 and request["temperature"] == 0.7
    assert request["extra_body"]["reasoning"] == {"effort": "high"}
    assert request["extra_body"]["provider"] == {
        "order": ["deepinfra/fp8"], "allow_fallbacks": False,
        "require_parameters": True, "quantizations": ["fp8"]}


def test_notebook_isolated_from_running_families():
    notebook = json.loads(Path("notebooks/run_deepseek41.ipynb").read_text())
    source = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "FAMILIES = ['deepseek41']" in source
    assert "runs/deepseek41_v1" in source and "runs/four_models_v1" not in source
    assert "/content/collusion-deepseek41" in source
    assert "collusion-deepseek41.bundle" in source
    assert "userdata.get('OPENROUTER_API_KEY')" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "run_deepseek41.ipynb", "exec")
