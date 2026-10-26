"""Offline audit of explicit numeric equalities in saved decisions; never executes model text.

This checks internal arithmetic, not factual operands, forecasts or optimal pricing. Unsupported
numeric prose stays in the review table. Repeated notes/branches are not independent observations.
"""

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal, DecimalException, localcontext
from pathlib import Path

FIELDS = ("observations", "plans", "insights")
STATUSES = ("consistent", "rounding_compatible", "arithmetic_mismatch", "uncheckable", "no_numeric_claim")
POLICY = {"version": "numeric_equalities_v1", "absolute_tolerance": "1e-8",
          "relative_tolerance": "1e-9", "decimal_precision": 40,
          "scope": "internal arithmetic only; mismatches require human review"}


def expression(text):
    """Parse a small arithmetic expression with a bounded AST whitelist, returning Decimal."""
    if len(text) > 180:
        raise ValueError("expression too long")
    text = text.strip().strip("` ").rstrip(".")
    text = text.translate(str.maketrans({"−": "-", "–": "-", "×": "*", "÷": "/"}))
    text = re.sub(r"(?<=[\d)])\s*[xX]\s*(?=[\d(])", "*", text)
    currencies = set(re.findall(r"[$£€]", text))
    if len(currencies) > 1:
        raise ValueError("mixed currency symbols")
    text = re.sub(r"[$£€]", "", text)
    for number in re.findall(r"[\d,]+", text):
        if "," in number and not re.fullmatch(r"\d{1,3}(?:,\d{3})+", number):
            raise ValueError("ambiguous comma notation")
    text = text.replace(",", "").replace("^", "**")
    number = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    text = re.sub(rf"({number})\s*%", r"(\1/100)", text)
    if not re.fullmatch(r"[\d.eE+*/()\s-]+", text):
        raise ValueError("symbols or prose require interpretation")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, RecursionError):
        raise ValueError("unsupported expression syntax") from None
    if sum(1 for _ in ast.walk(tree)) > 80:
        raise ValueError("expression too complex")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Decimal(ast.get_source_segment(text.strip(), node))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                value = left / right
            elif isinstance(node.op, ast.Pow) and right == int(right) and abs(right) <= 12:
                value = left ** int(right)
            else:
                raise ValueError("unsupported arithmetic operator or exponent")
        else:
            raise ValueError("only numeric arithmetic is supported")
        if not value.is_finite() or abs(value) > Decimal("1e30"):
            raise ValueError("numeric range exceeded")
        return value

    with localcontext() as context:
        context.prec = POLICY["decimal_precision"]
        return visit(tree.body)


def rounding_unit(text):
    """A literal result may be rounded to its last printed digit; expressions have no such rule."""
    literal = text.strip().strip("` ").rstrip(".").replace("−", "-")
    if not re.fullmatch(r"[$£€]?[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?", literal):
        return None
    percent = literal.endswith("%")
    value = Decimal(literal.lstrip("$£€").rstrip("%"))
    return Decimal(10) ** value.as_tuple().exponent / (100 if percent else 1)


def audit_text(text):
    claims = []
    # Preserve exact spans; deliberately avoid guessing where prose operands or references end.
    for match in re.finditer(r"[^;\n]+", text):
        excerpt = match.group()
        if not re.search(r"\d", excerpt):
            continue
        base = {"start": match.start(), "end": match.end(), "excerpt": excerpt,
                "relation": "", "left": "", "right": "", "expected": "", "claimed": "",
                "signed_error": "", "human_verdict": "", "human_comment": ""}
        parts = re.split(r"\s*(~=|≈|(?<![<>=!])=(?!=))\s*", excerpt.strip())
        if len(parts) < 3:
            claims.append(base | {"link": 0, "status": "uncheckable",
                                  "reason": "numeric prose, forecast or inequality; no supported equality"})
            continue
        # A named quantity before a chained equality is a label, not an operand.
        if len(parts) >= 5 and re.fullmatch(r"[A-Za-z][A-Za-z ]*", parts[0]):
            parts = parts[2:]
        parts[0] = re.sub(r"^[A-Za-z][A-Za-z ]*?(?::|\bis\b)\s*", "", parts[0])
        for i in range(0, len(parts) - 2, 2):
            left, relation, right = parts[i:i + 3]
            claim = base | {"link": i // 2, "relation": relation, "left": left, "right": right}
            try:
                if len(set(re.findall(r"[$£€]", left + right))) > 1:
                    raise ValueError("mixed currency symbols")
                expected, stated = expression(left), expression(right)
                error = stated - expected
                tolerance = max(Decimal(POLICY["absolute_tolerance"]),
                                abs(expected) * Decimal(POLICY["relative_tolerance"]))
                unit = rounding_unit(right)
                if abs(error) <= tolerance:
                    status, reason = "consistent", "numeric sides agree within fixed tolerance"
                elif unit is not None and abs(error) <= unit / 2 + tolerance:
                    status, reason = "rounding_compatible", "difference fits half the final printed unit"
                elif relation != "=":
                    status, reason = "uncheckable", "approximation exceeds rounding bound; intent needs review"
                else:
                    status, reason = "arithmetic_mismatch", "numeric sides disagree; review before attribution"
                claim |= {"expected": str(expected), "claimed": str(stated), "signed_error": str(error),
                          "status": status, "reason": reason}
            except (ValueError, DecimalException, OverflowError, RecursionError) as error:
                claim |= {"status": "uncheckable", "reason": str(error)}
            claims.append(claim)
    return claims or [{"start": 0, "end": len(text), "excerpt": text, "link": 0,
                       "status": "no_numeric_claim", "reason": "no Arabic numeral in this field"}]


def audit_journal(path):
    """Read a snapshot, including saved answers in unfinished rounds; never alter source logs."""
    raw = path.read_bytes()
    events, torn = [], False
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1 or raw.endswith(b"\n"):
                raise ValueError(f"corrupt journal: {path}, line {i + 1}") from None
            torn = True
    source = {"journal": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
              "bytes": len(raw), "torn_tail_ignored": torn}
    if not events:
        return [], source
    head = events[0]
    if head.get("event") != "header":
        raise ValueError(f"missing header: {path}")
    model = head.get("model", "unknown")
    setup = hashlib.sha256(json.dumps({k: v for k, v in head.items()
                                      if k not in ("game_seed", "event")}, sort_keys=True).encode()).hexdigest()
    committed = {(e["branch"], e["row"]["number"]) for e in events if e["event"] == "round"}
    rows, seen = [], {}
    for line, event in enumerate(events, 1):
        if event["event"] != "decision":
            continue
        key = (event["branch"], event["round"], event["seller"])
        if key in seen:
            if seen[key] != event["decision"]:
                raise ValueError(f"conflicting saved decisions: {path}, {key}")
            continue
        seen[key] = event["decision"]
        for field in FIELDS:
            text = event["decision"][field]
            identity = {"journal": str(path.resolve()), "source_sha256": source["sha256"], "line": line,
                        "run": str(path.parent.resolve()), "setup": setup, "game_seed": head.get("game_seed"),
                        "model": model if isinstance(model, str) else model["model"],
                        "branch": key[0], "round": key[1], "seller": key[2], "field": field,
                        "round_committed": key[:2] in committed,
                        "note_sha256": hashlib.sha256(text.encode()).hexdigest()}
            for claim in audit_text(text):
                claim_id = hashlib.sha256(json.dumps([str(path.resolve()), key, field, claim["start"],
                    claim["end"], claim["link"], identity["note_sha256"]]).encode()).hexdigest()[:24]
                rows.append(identity | claim | {"claim_id": claim_id})
    source["saved_decisions"] = len(seen)
    return rows, source


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export(inputs, out):
    paths = set()
    for item in map(Path, inputs):
        if not item.exists():
            raise ValueError(f"input does not exist: {item}")
        paths.update(item.rglob("game_*.jsonl") if item.is_dir() else [item])
    paths = sorted(p for p in paths if not p.name.endswith(".calls.jsonl"))
    if not paths:
        raise ValueError("no game journals found")
    rows, sources = [], []
    for path in paths:
        claims, source = audit_journal(path)
        rows.extend(claims)
        sources.append(source)
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("run", "setup", "model", "field"))
        group = groups.setdefault(key, {"counts": Counter(), "notes": set()})
        group["counts"][row["status"]] += 1
        group["notes"].add((row["journal"], row["branch"], row["round"], row["seller"]))
    summaries = [dict(zip(("run", "setup", "model", "field"), key)) |
                 {"notes": len(group["notes"])} | {s: group["counts"][s] for s in STATUSES}
                 for key, group in groups.items()]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # A new output directory protects manual annotations on reruns.
    if any(out.iterdir()):
        raise ValueError("output directory must be empty; preserve earlier human annotations")
    write_csv(out / "claims.csv", rows)
    write_csv(out / "review.csv", sorted(
        [r for r in rows if r["status"] in ("arithmetic_mismatch", "uncheckable")],
        key=lambda r: r["status"] != "arithmetic_mismatch"))
    write_csv(out / "summary.csv", summaries)
    manifest = {"created_utc": datetime.now(UTC).isoformat(), "policy": POLICY, "sources": sources,
                "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "warning": "Descriptive extraction counts only; no all-notes arithmetic error rate."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return Counter(r["status"] for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(dict(export(args.inputs, args.out)))
    print(f"Audit saved to {args.out}. Review flags; do not interpret unsupported claims as correct.")


if __name__ == "__main__":
    main()
