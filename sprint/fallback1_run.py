"""Fallback 1 CLI, isolated from the pinned main runner. Live calls require an explicit attempt budget."""

import argparse
import asyncio
import hashlib
import json
import sys
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

from sprint.client import Budget, Fatal, Interrupted, LLMPolicy, ModelConfig, Unusable
from sprint.engine import Journal
from sprint.fallback1 import ReplacementRules, notices, replacement_game
from sprint.market import Market
from sprint.run import attempts_logged, code_sha, probe_request, provenance
from sprint.scripted import demo_policy


def fallback_sha():
    h = hashlib.sha256(code_sha().encode())
    for name in ("fallback1.py", "fallback1_run.py"):
        h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


def load_config(path):
    config = tomllib.loads(Path(path).read_text())
    if config.get("experiment") != "replacement":
        raise ValueError("expected experiment = 'replacement'")
    if not config.get("seeds") or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    if any(type(s) is not int for s in config["seeds"]):
        raise ValueError("seeds must be integers")
    if type(config.get("concurrency", 4)) is not int or config.get("concurrency", 4) <= 0:
        raise ValueError("concurrency must be a positive integer")
    config["rules"] = ReplacementRules(**config.get("rules", {}))
    config["model"] = ModelConfig(**config["model"])
    return config


def header(config, seed, stub):
    return {"experiment": "replacement", "game_seed": seed, "rules": asdict(config["rules"]),
            "market": asdict(Market(2)), "model": "stub" if stub else asdict(config["model"]),
            "code_sha": fallback_sha()}


async def run_batch(config, out, budget, stub=False, policy_factory=None):
    gate = asyncio.Semaphore(config.get("concurrency", 4))

    async def one(seed):
        async with gate:
            journal = Journal(out / f"game_{seed}.jsonl", header(config, seed, stub))
            if journal.result is not None:
                return journal.result
            if journal.failure is not None:
                return {"status": "failed", "game_seed": seed, "reason": journal.failure}
            policy = (policy_factory(seed) if policy_factory else demo_policy if stub else
                      LLMPolicy(config["model"], Market(2), budget, out / f"game_{seed}.calls.jsonl"))
            try:
                return await replacement_game(policy, seed, journal, rules=config["rules"], label=f"g{seed}:")
            except Unusable as error:
                journal.fail(str(error))
                return {"status": "failed", "game_seed": seed, "reason": str(error)}
            except Interrupted as error:
                journal.write({"event": "interrupted", "reason": str(error)})
                return {"status": "interrupted", "game_seed": seed, "reason": str(error)}
            except BaseException as error:
                journal.write({"event": "interrupted", "reason": type(error).__name__})
                raise

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(one(seed)) for seed in config["seeds"]]
    except* Fatal as errors:
        raise errors.exceptions[0] from None
    return [t.result() for t in tasks]


def full_context_request(config):
    request = probe_request(config)
    return replace(request, history=tuple(replace(o, prices=o.prices[:2]) for o in request.history),
                   notice=notices(Market(2), config["rules"])[1])


async def context_probe(config, out, budget):
    # Fixed label makes an answered probe resumable instead of buying a new one on each Run all.
    policy = LLMPolicy(config["model"], Market(2), budget, out / "probe.calls.jsonl")
    journal = Journal(out / "probe.jsonl", header(config, 0, False))
    if journal.result is not None:
        return journal.result
    request = full_context_request(config)
    decision = await policy(request, "context_probe")
    return journal.finish({"status": "complete", "price": decision.price})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--context-probe", action="store_true")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    plan = config["rules"].planned_calls()
    print(f"{config['name']}: {len(config['seeds'])} games; up to {plan['maximum']} decisions/game "
          f"({len(config['seeds']) * plan['maximum']} total), before retries", flush=True)
    if args.dry_run:
        return
    if args.context_probe and args.stub:
        ap.error("context probe is a live endpoint check; omit --stub")
    if not args.stub and args.budget <= 0:
        ap.error("set --budget for live API attempts")
    suffix = "_probe" if args.context_probe else "_stub" if args.stub else ""
    out = args.out / (config["name"] + suffix)
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.budget, attempts_logged(out))
    if args.context_probe:
        print(asyncio.run(context_probe(config, out, budget)))
        return
    results = asyncio.run(run_batch(config, out, budget, args.stub))
    record = provenance(" ".join(sys.argv)) | {"fallback_sha": fallback_sha(), "attempts_used": budget.used,
                                               "games": [{"seed": r["game_seed"], "status": r["status"]}
                                                         for r in results]}
    with (out / "runs.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    from sprint.fallback1_analyze import export_tables
    export_tables([out], out)
    for result in results:
        print(f"game {result['game_seed']}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
