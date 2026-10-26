import importlib.util
import json
from pathlib import Path

import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "report_figures", Path(__file__).resolve().parents[1] / "scripts/report_figures.py")
figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figures)


def test_distinct_runs_and_setups_are_never_pooled(tmp_path):
    source, out = tmp_path / "source", tmp_path / "plots"
    source.mkdir()
    rows = [{"run": run, "setup": setup, "model": "synthetic", "seed": 1,
             "contrast": "replacement_memory", "outcome": "mean_price", "difference": value}
            for run, setup, value in [("a", "x", -1), ("b", "x", 1), ("a", "y", 4)]]
    pd.DataFrame(rows).to_csv(source / "replacement_contrasts.csv", index=False)
    figures.run("f1", source, out, "SYNTHETIC")
    assert len(list(out.glob("*.svg"))) == 3
    assert len(list(out.glob("*.png"))) == 3


def test_empty_input_is_recorded_not_plotted_as_zero(tmp_path):
    (tmp_path / "primary.csv").write_text("\n")
    out = tmp_path / "out"
    figures.run("main", tmp_path, out)
    assert not list(out.glob("*.png"))
    assert json.loads((out / "main_figure_manifest.json").read_text())["rows"] == 0


def test_main_forest_includes_s_minus_j(tmp_path, monkeypatch):
    primary = pd.DataFrame([{"run": "a", "setup": "x", "model": "synthetic", "stage": "pilot",
                             "outcome": "S", "eligible": 4, "mean": 1, "lo": 0, "hi": 2}])
    primary = pd.concat([primary, primary.assign(outcome="S_minus_J", mean=2, lo=1, hi=3)])
    primary.to_csv(tmp_path / "primary.csv", index=False)
    labels = []

    def capture(fig, _out, _name):
        labels.extend(fig.axes[0].get_xticklabels())
        figures.plt.close(fig)

    monkeypatch.setattr(figures, "save", capture)
    figures.run("main", tmp_path, tmp_path / "out")
    assert [label.get_text().split("\n")[0] for label in labels] == ["S", "S_minus_J"]


def test_f2_failed_cases_remain_in_planned_denominator(tmp_path, monkeypatch):
    common = {"run": "test", "setup": "test", "model": "synthetic", "scheme": "affine",
              "question": "dependent", "view": "decoder", "block": 1,
              **{name + "_correct": False for name in
                 ("evidence_level_only", "parallel_rise", "always_abstain", "always_accuse")}}
    table = pd.DataFrame([
        common | {"expected": "insufficient_evidence", "predicted": "supported", "correct": True, "answered": True},
        common | {"expected": "insufficient_evidence", "predicted": None, "correct": False, "answered": False},
    ])
    plotted = []

    def capture(fig, _out, name):
        if name.startswith("f2_baselines"):
            plotted.extend(fig.axes[0].collections[0].get_offsets()[:, 1])
        figures.plt.close(fig)

    monkeypatch.setattr(figures, "save", capture)
    figures.f2_figures(table, tmp_path, "SYNTHETIC")
    assert plotted == [.5]
