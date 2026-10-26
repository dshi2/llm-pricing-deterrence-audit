"""Run a configured batch of games: one journal and one call log per game; rerunning resumes.

    python -m sprint.run configs/canary_deepseek.toml --budget 200 --dry-run
    python -m sprint.run configs/canary_deepseek.toml --budget 200
    python -m sprint.run configs/main_pilot_deepseek.toml --budget 10 --context-probe
    python -m sprint.run configs/main_pilot_deepseek.toml --stub   (offline, no API calls)

The budget caps API attempts (retries included) for the run directory in total: attempts already in
its call logs count, so rerunning never grants a fresh allowance. Never run one directory from two
processes at once. Completed games are skipped, interrupted games resume, and
failed games are reported, never silently rerun.
"""

import argparse
import asyncio
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import openai

from sprint import starter
from sprint.client import Budget, Fatal, Interrupted, LLMPolicy, ModelConfig, Unusable
from sprint.engine import Journal, Observation, Request
from sprint.experiments import MainRules, main_game
from sprint.market import Market
from sprint.prompt import messages
from sprint.scripted import demo_policy

RUN_MODULES = ("client", "engine", "experiments", "market", "prompt", "run", "scripted", "starter")
ROOT = Path(__file__).resolve().parent


def code_sha() -> str:
    """Hash of every module that affects what a game does; a change starts a new version."""
    h = hashlib.sha256()
    for name in RUN_MODULES:
        h.update((ROOT / f"{name}.py").read_bytes())
    return h.hexdigest()


def provenance(command: str) -> dict:
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT.parent), *a], capture_output=True, text=True, check=False).stdout.strip()
    return {"time_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "command": command, "repo_commit": git("rev-parse", "HEAD") or "unknown",
            "repo_dirty": bool(git("status", "--porcelain", "--", "sprint")), "code_sha": code_sha(),
            "starter_commit": starter.COMMIT, "python": platform.python_version(), "openai": openai.__version__}


def load_config(path) -> dict:
    config = tomllib.loads(Path(path).read_text())
    if config.get("experiment") != "main":
        raise ValueError("only experiment = 'main' is implemented")
    config["rules"] = replace(MainRules(), **config.get("rules", {}))
    config["model"] = ModelConfig(**config["model"])
    return config


def header(config: dict, seed: int, stub: bool) -> dict:
    return {"experiment": config["experiment"], "game_seed": seed, "rules": asdict(config["rules"]),
            "market": asdict(Market(3)), "model": "stub" if stub else asdict(config["model"]),
            "code_sha": code_sha(), "starter_commit": starter.COMMIT}


async def run_batch(config: dict, out: Path, budget: Budget, stub: bool = False) -> list[dict]:
    rules, market = config["rules"], Market(3)
    gate = asyncio.Semaphore(config.get("concurrency", 4))

    async def one(seed: int) -> dict:
        async with gate:
            journal = Journal(out / f"game_{seed}.jsonl", header(config, seed, stub))
            policy = demo_policy if stub else LLMPolicy(config["model"], market, budget, out / f"game_{seed}.calls.jsonl")
            try:
                return await main_game(policy, seed, journal, market, rules, label=f"g{seed}:")
            except Unusable as e:  # includes Declined: the model did not take part
                journal.fail(str(e))
                return {"status": "failed", "game_seed": seed, "reason": str(e)}
            except Interrupted as e:
                journal.write({"event": "interrupted", "reason": str(e)})
                return {"status": "interrupted", "game_seed": seed, "reason": str(e)}
            except asyncio.CancelledError:
                journal.write({"event": "interrupted", "reason": "cancelled: another game hit a fatal error"})
                raise

    try:
        async with asyncio.TaskGroup() as group:  # a Fatal error in one game cancels the rest
            tasks = [group.create_task(one(seed)) for seed in config["seeds"]]
    except* Fatal as errors:
        raise errors.exceptions[0] from None
    return [t.result() for t in tasks]


PROBE_PLANS = ("Hold near 5.60 while both rivals stay above 5.40; if Firm B undercuts for two rounds, match "
               "its price for three rounds, then return to 5.60 and watch whether it follows. ")
PROBE_INSIGHTS = ("Rounds at 5.55 to 5.65 earned more than rounds near 5.00, so demand is not very sensitive "
                  "here; Firm C reacts one round late, and joint high prices have held for most of the game. ")


def probe_request(config: dict) -> Request:
    """A long prompt for the context probe: a full history window of three sellers, and realistic plans
    and insights each about max_tokens tokens long (about 4 characters per token), so together about
    twice what one reply can hold. A probe, not a proof of the longest possible prompt."""
    rules, chars = config["rules"], 4 * config["model"].max_tokens
    note = lambda text: (text * (chars // len(text) + 1))[:chars].rstrip()
    history = tuple(Observation(r, (1.6 + 0.001 * r, 1.7111, 1.8222), 33.333, 123.456)
                    for r in range(1, rules.window + 1))
    return Request(0, rules.window + 1, 1, history, note(PROBE_PLANS), note(PROBE_INSIGHTS))


async def context_probe(config: dict, out: Path, budget: Budget) -> dict:
    """One live call with the long prompt: does it fit, get answered and parse? Not a proof of the maximum."""
    market, request = Market(3), probe_request(config)
    log = out / f"probe_{datetime.datetime.now():%Y%m%d_%H%M%S}.calls.jsonl"  # fresh: never a replay
    decision = await LLMPolicy(config["model"], market, budget, log)(request, "context_probe")
    last = json.loads(log.read_text().splitlines()[-1])
    msgs = messages(request, market, config["model"].prefix, config["model"].units)
    return {"prompt_chars": sum(len(m["content"]) for m in msgs),
            "prompt_tokens": last["prompt_tokens"], "completion_tokens": last["completion_tokens"],
            "cost": last["cost"], "price": decision.price}


def attempts_logged(out: Path) -> int:
    """API attempts already made from a run directory: one call-log row each (a torn row counts too)."""
    return sum(len(p.read_bytes().splitlines()) for p in out.glob("*.calls.jsonl"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--budget", type=int, default=0,
                    help="total API attempts (retries included) this run directory may use, earlier runs counted")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and planned calls; call nothing")
    ap.add_argument("--stub", action="store_true", help="offline: a scripted stand-in replaces the model")
    ap.add_argument("--context-probe", action="store_true", help="one live call with a long prompt")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    out = args.out / (config["name"] + ("_stub" if args.stub else "") + ("_probe" if args.context_probe else ""))
    plan = config["rules"].planned_calls()
    planned = len(config["seeds"]) * plan["eligible"]
    print(f"{config['name']}: {len(config['seeds'])} games, model "
          f"{'stub' if args.stub else config['model'].model + ' via ' + config['model'].provider}")
    print(f"planned decision calls: up to {planned} ({plan['eligible']} per eligible game, "
          f"{plan['ineligible']} per ineligible), before retries; budget {args.budget}")
    if args.dry_run:
        return
    if not args.stub and args.budget <= 0:
        sys.exit("set --budget (API attempts, retries included) for a live run")
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.budget, used=attempts_logged(out))
    print(f"attempts already logged in {out}: {budget.used} of {args.budget}")
    if args.context_probe:
        print(json.dumps(asyncio.run(context_probe(config, out, budget)), indent=1))
        return
    results = asyncio.run(run_batch(config, out, budget, args.stub))
    record = {**provenance(" ".join(sys.argv)), "config_file": str(args.config), "attempts_used": budget.used,
              "games": [{"seed": r["game_seed"], "status": r["status"], "reason": r.get("reason")} for r in results]}
    with (out / "runs.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    for r in results:
        print(f"game {r['game_seed']}: {r['status']}" + (f" ({r['reason']})" if r.get("reason") else ""))
    print(f"attempts logged in this directory: {budget.used} of {args.budget}; journals in {out}")


if __name__ == "__main__":
    main()
