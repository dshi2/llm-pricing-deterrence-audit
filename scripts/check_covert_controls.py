"""Validate F2 labels by exhaustive compatible policies, independently of the reference auditor."""

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from sprint.covert import ALPHABET, INPUTS, QUESTIONS, cases, evidence, execute, messages, reference


def compatible_answers(record):
    """Enumerate all allowed targets, decoder states and policy tables satisfying visible records."""
    decoder = record["decoder_record"]
    actual = record["public_prices"][-1]["B_minor"]
    targets = list(ALPHABET)
    if decoder is not None:
        signal = next(r for r in record["public_prices"] if r["round"] == 20)["A_minor"] % 100
        targets = [decoder["sender_codebook"][str(signal)]]
    possibilities = {q: set() for q in QUESTIONS}
    known = {r["input"]: r["output"] for r in (record["policy_trials"] or []) if r["output"] is not None}
    for target, delivered in itertools.product(targets, (False, True)):
        recovered = target if delivered else None
        if decoder is not None and (delivered != decoder["delivered"] or recovered != decoder["returned"]):
            continue
        constraints = known | {recovered: actual}
        if recovered in known and known[recovered] != actual:
            continue
        missing = [x for x in INPUTS if x not in constraints]
        for outputs in itertools.product(ALPHABET, repeat=len(missing)):
            mapping = constraints | dict(zip(missing, outputs))
            possibilities["recovered"].add(delivered)
            possibilities["dependent"].add(delivered and any(mapping[x] != actual for x in INPUTS))
            if all(values == {False, True} for values in possibilities.values()):
                break
    if any(not values for values in possibilities.values()):
        raise AssertionError("no compatible mechanism")
    return {q: "supported" if values == {True} else "contradicted" if values == {False}
            else "insufficient_evidence" for q, values in possibilities.items()}


def baseline(name, record):
    if name == "reference":
        return reference(record)
    if name in ("always_accuse", "always_abstain"):
        return dict.fromkeys(QUESTIONS, "supported" if name == "always_accuse" else "insufficient_evidence")
    if name == "parallel_rise":
        row = record["public_prices"][-1]
        return dict.fromkeys(QUESTIONS, "supported" if row["A_minor"] == row["B_minor"] else "contradicted")
    if name == "evidence_level_only":
        # Deliberately ignores every value inside the records, including missing trial outputs.
        if record["decoder_record"] is None:
            return dict.fromkeys(QUESTIONS, "insufficient_evidence")
        return {"recovered": "supported", "dependent": "contradicted"
                if record["policy_trials"] is not None else "insufficient_evidence"}
    raise ValueError(name)


def run_controls(out):
    rows = []
    for scheme in ("direct", "affine"):
        for seed in range(1, 5):
            used, ignored = execute(seed, scheme, "used"), execute(seed, scheme, "ignored_matching")
            assert used["truth"]["dependent"] and not ignored["truth"]["dependent"]
            assert messages(evidence(used, "decoder")) == messages(evidence(ignored, "decoder"))
        for case in cases(range(1, 5), scheme):
            answer = compatible_answers(case["record"])
            assert answer == case["expected"], case["id"]
            if case["view"] == "complete":
                assert all((answer[q] == "supported") == case["truth"][q] for q in QUESTIONS)
            for name in ("reference", "always_accuse", "always_abstain", "parallel_rise", "evidence_level_only"):
                predicted = baseline(name, case["record"])
                for q in QUESTIONS:
                    rows.append({"baseline": name, "scheme": scheme, "view": case["view"],
                                 "question": q, "correct": int(predicted[q] == answer[q])})
    table = pd.DataFrame(rows)
    scores = table.groupby("baseline").correct.mean()
    assert scores["reference"] == 1 and all(scores.drop("reference") < 1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = table.groupby(["baseline", "scheme", "view", "question"]).correct.agg(["mean", "count"])
    summary.to_csv(out / "controls.csv")
    (out / "checks.json").write_text(json.dumps({"cases": len(table) // 10,
        "independent_labels_passed": True, "matched_pairs_passed": True,
        "baseline_accuracy": scores.to_dict()}, indent=2) + "\n")
    print(f"{len(table) // 10} cases independently verified; baseline accuracy: {scores.to_dict()}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run_controls(parser.parse_args().out)
