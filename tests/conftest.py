from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def repo_root(monkeypatch):
    """Tests read configs and notebooks by relative path, so run them from the repository root."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
