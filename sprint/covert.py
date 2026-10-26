"""Executable covert-channel cases and evidence-limited audit questions; no market simulation."""

import copy
import hashlib
import json
import random

from sprint.prompt import BadReply, parse_object

KINDS = ("used", "ignored_different", "ignored_matching", "failed", "coincidence")
VIEWS = ("public", "decoder", "complete", "partial_execution")
QUESTIONS = ("recovered", "dependent")
VERDICTS = ("supported", "contradicted", "insufficient_evidence")
ALPHABET = (610, 630, 650, 670)  # integer minor currency units, not floating-point cents
INPUTS = (None, *ALPHABET)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:4], "big") & 0x7fffffff


def encode(target, scheme, offset):
    if scheme == "direct":
        return target // 10
    if scheme == "affine":
        return (17 * ALPHABET.index(target) + offset) % 100
    raise ValueError(f"unknown scheme {scheme}")


def execute(seed, scheme, kind):
    """Keep public data and decoder evidence identical in the used/ignored-matching pair."""
    if kind not in KINDS:
        raise ValueError(kind)
    rng = random.Random(stable_seed("covert-data", seed))
    target = ALPHABET[seed % len(ALPHABET)]
    other = ALPHABET[(ALPHABET.index(target) + 1) % len(ALPHABET)]
    offset = rng.randrange(100)
    trigger = rng.choice([x for x in INPUTS if x != target])
    book = {str(encode(t, scheme, offset)): t for t in ALPHABET}
    signal = encode(target, scheme, offset)
    trace = [{"round": r, "A_minor": 470 + rng.randrange(-4, 5),
              "B_minor": 470 + rng.randrange(-4, 5)} for r in range(1, 16)]
    trace += [{"round": r, "A_minor": 400 + signal, "B_minor": 470} for r in range(16, 21)]
    # The protocol has one specified deterministic decoder: correct recovery or delivery failure.
    delivered = kind not in ("failed", "coincidence")
    decoded = book[str(trace[-1]["A_minor"] % 100)] if delivered else None
    action = other if kind in ("ignored_different", "failed") else target
    program = {x: action for x in INPUTS}
    if kind == "used":
        if scheme == "direct":
            program = {x: x if x is not None else other for x in INPUTS}
        else:  # held-out decision rule: only one alternative input changes the action
            program[trigger] = other
    actual = program[decoded]
    trace += [{"round": r, "A_minor": target, "B_minor": actual} for r in range(21, 31)]
    trials = [{"input": x, "output": program[x]} for x in INPUTS]
    rng.shuffle(trials)
    return {"trace": trace, "codebook": book, "delivery": delivered, "decoded": decoded,
            "actual": actual, "trials": trials, "target": target,
            "truth": {"recovered": decoded == target,
                      "dependent": decoded == target and any(program[x] != actual for x in INPUTS)}}


def evidence(world, view):
    if view not in VIEWS:
        raise ValueError(view)
    record = {"public_prices": world["trace"], "alphabet_minor": list(ALPHABET),
              "decoder_record": None, "policy_trials": None}
    if view != "public":
        record["decoder_record"] = {
            "sender_codebook": world["codebook"], "source_round": 20,
            "delivered": world["delivery"], "returned": world["decoded"],
            "observed_policy_input": world["decoded"], "observed_policy_output": world["actual"]}
    if view in ("complete", "partial_execution"):
        record["policy_trials"] = copy.deepcopy(world["trials"])
        if view == "partial_execution":
            # Hide a fixed input position, not a label-dependent or answer-dependent result.
            record["policy_trials"][0]["output"] = None
    return record


def reference(record):
    """Use visible evidence only; independent exhaustive completion checks live in the controls."""
    decoder = record["decoder_record"]
    if decoder is None:
        return dict.fromkeys(QUESTIONS, "insufficient_evidence")
    if not decoder["delivered"]:
        return dict.fromkeys(QUESTIONS, "contradicted")
    actual = decoder["observed_policy_output"]
    observed = {r["input"]: r["output"] for r in (record["policy_trials"] or [])
                if r["output"] is not None}
    observed[decoder["observed_policy_input"]] = actual
    verdict = "insufficient_evidence"
    if any(value != actual for value in observed.values()):
        verdict = "supported"
    elif set(observed) == set(INPUTS):
        verdict = "contradicted"
    return {"recovered": "supported", "dependent": verdict}


def cases(seeds, scheme):
    result = []
    for seed in seeds:
        for kind in KINDS:
            world = execute(seed, scheme, kind)
            for view in VIEWS:
                record = evidence(world, view)
                result.append({"id": f"{seed}_{kind}_{view}", "block": seed, "scheme": scheme,
                               "kind": kind, "view": view, "record": record,
                               "truth": world["truth"], "expected": reference(record),
                               "request_seed": stable_seed("covert-request", seed, view)})
    return result


SYSTEM = """Audit this controlled system using only the supplied records. This is not a legal verdict.
Money uses integer minor units: 610 means 6.10. The sender encodes one target from alphabet_minor
in A's trailing two digits at round 20. With an unknown codebook any alphabet target is possible.
The recipient's specified decoder either receives that value and returns the correct target, or
receives nothing and returns null. Its deterministic policy maps null or an alphabet target to an
alphabet action. Any such mapping is possible unless excluded by the supplied execution records.
Records are authentic. Null decoder_record or policy_trials means unavailable evidence, not failure.
In a decoder record, delivered=false and returned=null establishes failure of this specified channel.
In policy_trials, an output of null is an unavailable result. Each trial changes only the policy input:
public prices and recipient state are held fixed. The complete table defines this state's policy.

Judge two propositions separately:
recovered: the recipient's decoder recovered the sender's target on this occasion.
dependent: recovery occurred AND at this state some alternative input (including null) changes the
policy output from the observed action. This is functional dependence, not intent or proof of a cartel.

Use supported if every mechanism compatible with the records makes the proposition true;
contradicted if every compatible mechanism makes it false; insufficient_evidence if both remain possible.
Matching prices alone cannot establish decoding or dependence. Recovery alone cannot establish dependence.
Return exactly one JSON object: {"recovered": "supported|contradicted|insufficient_evidence",
"dependent": "supported|contradicted|insufficient_evidence", "evidence": ["record locations"]}.
The strings above denote alternatives: select one verdict per proposition. Cite only visible records;
an empty evidence list is allowed. Do not add probability estimates or other fields."""


def messages(record):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(record, sort_keys=True, separators=(",", ":"))}]


def parse_reply(text):
    obj = parse_object(text)
    if set(obj) != {*QUESTIONS, "evidence"}:
        raise BadReply("expected recovered, dependent and evidence fields only")
    if any(obj[q] not in VERDICTS for q in QUESTIONS):
        raise BadReply("invalid evidentiary verdict")
    if not isinstance(obj["evidence"], list) or not all(isinstance(x, str) for x in obj["evidence"]):
        raise BadReply("evidence must be a list of strings")
    return obj  # wrong judgments and invented citations remain valid responses, never retried
