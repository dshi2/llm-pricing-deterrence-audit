"""Check notebook code and launch gating offline; this does not validate Colab or Drive."""

import ast
import json
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from sprint.run import load_config

NOTEBOOK = Path("notebooks/run_models.ipynb")


def notebook_function(name, namespace):
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)  # noqa: S102 -- own notebook, offline tests
                    return namespace[name]
    raise AssertionError(f"missing function: {name}")


def test_notebook_code_compiles_without_magics():
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")


@pytest.mark.parametrize("changes", [{"limit": None}, {"limit": 101}, {"limit_reset": "daily"},
                                     {"limit_remaining": 0}, {"limit_remaining": None}])
def test_budget_gate_rejects_missing_oversized_or_resetting_caps(changes):
    check = notebook_function("check_key_budget", {})
    data = {"limit": 100, "limit_reset": None, "limit_remaining": 40}
    assert check(data)["remaining_allowance_usd"] == 40
    with pytest.raises(RuntimeError):
        check(data | changes)


def test_failed_canary_gate_never_launches_competence_or_pilot(tmp_path):
    sent = []
    launch = notebook_function("run_family", {
        "OUT": tmp_path, "sys": SimpleNamespace(executable="python"),
        "subprocess": SimpleNamespace(STDOUT=-2), "sh": lambda args, **kwargs: sent.append(args),
        "analyze": SimpleNamespace(launch_checks=lambda *args: {"provider verified": False}),
        "load_config": lambda path: {}, "endpoint_snapshot": lambda config: None,
    })
    with pytest.raises(RuntimeError, match="launch checks failed"):
        launch("qwen38")
    assert len(sent) == 2
    assert "canary_qwen38.toml" in sent[0][4]
    assert "--context-probe" in sent[1]


def test_endpoint_snapshot_preserves_anchor_and_rejects_drift(tmp_path):
    endpoint = {"tag": "deepinfra/bf16", "provider_name": "DeepInfra", "quantization": "bf16",
                "context_length": 131072, "max_completion_tokens": 100000,
                "supported_parameters": ["seed", "temperature", "max_tokens", "response_format", "reasoning"]}
    def urlopen(*args, **kwargs):
        return StringIO(json.dumps({"data": {"endpoints": [endpoint]}}))
    snapshot = notebook_function("endpoint_snapshot", {
        "OUT": tmp_path, "urllib": SimpleNamespace(request=SimpleNamespace(urlopen=urlopen)),
        "json": json, "datetime": datetime, "UTC": UTC,
    })
    config = load_config("configs/main_pilot_gptoss120.toml")
    snapshot(config)
    anchor = (tmp_path / "main_gptoss120.endpoint.json").read_text()
    endpoint["supported_parameters"].reverse()
    snapshot(config)  # order-only changes are immaterial
    endpoint["supported_parameters"].append("tools")
    snapshot(config)  # a newly advertised, unused parameter does not change our requests
    endpoint["context_length"] = 8192
    with pytest.raises(RuntimeError, match="metadata changed"):
        snapshot(config)
    assert (tmp_path / "main_gptoss120.endpoint.json").read_text() == anchor
    assert len((tmp_path / "main_gptoss120.endpoint_history.jsonl").read_text().splitlines()) == 4
    endpoint["context_length"] = 131072
    endpoint["supported_parameters"].remove("seed")
    with pytest.raises(RuntimeError, match="required parameters"):
        snapshot(config)
    endpoint["quantization"] = "fp8"
    with pytest.raises(RuntimeError, match="quantization changed"):
        snapshot(config)
