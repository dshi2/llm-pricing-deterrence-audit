"""Independent F2 audit calls, using the reviewed transport and append-only resume journals."""

import argparse
import asyncio
import hashlib
import json
import os
import random
import tomllib
from dataclasses import asdict
from pathlib import Path

import openai

from sprint.client import Budget, Fatal, Interrupted, LLMPolicy, Unusable
from sprint.covert import cases, messages, parse_reply
from sprint.engine import Journal
from sprint.market import Market
from sprint.run import attempts_logged, code_sha, provenance
from sprint.run import load_config as load_model


def audit_sha():
    digest = hashlib.sha256(code_sha().encode())
    for name in ("covert.py", "covert_run.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def load_config(path, model_path):
    config = tomllib.loads(Path(path).read_text())
    seeds = config.get("seeds", [])
    if (config.get("experiment") != "covert_audit" or not seeds
            or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)
            or type(config.get("concurrency")) is not int or config["concurrency"] < 1
            or config.get("scheme") not in ("direct", "affine")):
        raise ValueError("invalid covert audit configuration")
    model = load_model(model_path)
    config["model"] = model["model"]
    config["name"] += "_" + model["name"].removeprefix("main_")
    return config


async def run_batch(config, out, budget, stub=None, client=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sample = cases(config["seeds"], config["scheme"])
    manifest = {"audit_sha": audit_sha(), "concurrency": config["concurrency"],
                "model": asdict(config["model"]) if stub is None
                else {"model": f"scripted:{stub}"}, "cases": sample}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("run configuration changed; use a new directory")
    else:
        temporary = out / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        temporary.replace(manifest_path)
    rng = random.Random(731)
    rng.shuffle(sample)  # order is unrelated to case truth and fixed across resumes
    semaphore = asyncio.Semaphore(config["concurrency"])
    api = client
    owned = stub is None and client is None
    if owned:
        api = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                                api_key=os.environ["OPENROUTER_API_KEY"], max_retries=0, timeout=120)
    sha = audit_sha()

    async def one(case):
        async with semaphore:
            prompts = messages(case["record"])
            head = {"experiment": "covert_audit", "audit_sha": sha,
                    "model": asdict(config["model"]) if stub is None else {"model": f"scripted:{stub}"},
                    "case": case, "messages": prompts}
            journal = Journal(out / f"case_{case['id']}.jsonl", head)
            if journal.result is not None:
                return journal.result
            if journal.failure is not None:
                return {"status": "failed"}
            try:
                if stub is not None:
                    from scripts.check_covert_controls import baseline
                    reply = baseline(stub, case["record"]) | {"evidence": []}
                else:
                    transport = LLMPolicy(config["model"], Market(2), budget,
                                          out / f"case_{case['id']}.calls.jsonl", client=api)
                    reply = await transport.complete(prompts, case["request_seed"], case["id"], parse_reply)
                return journal.finish({"status": "answered", "reply": reply})
            except Unusable as error:
                journal.fail(f"{type(error).__name__}: {error}")
                return {"status": "failed"}
            except (Interrupted, Fatal, asyncio.CancelledError) as error:
                journal.write({"event": "interrupted", "reason": f"{type(error).__name__}: {error}"})
                if isinstance(error, Interrupted):
                    return {"status": "incomplete"}
                raise

    try:
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(one(case)) for case in sample]
        except* Fatal as errors:
            raise errors.exceptions[0] from None
        return [task.result() for task in tasks]
    finally:
        if owned:
            await api.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stub", choices=("reference", "always_accuse", "always_abstain",
                                          "parallel_rise", "evidence_level_only"))
    args = parser.parse_args()
    config = load_config(args.config, args.model_config)
    sample = cases(config["seeds"], config["scheme"])
    lengths = [sum(len(m["content"]) for m in messages(c["record"])) for c in sample]
    print(f"{config['name']}: {len(sample)} independent judgments before retries; "
          f"prompt characters {min(lengths)}–{max(lengths)}. Characters are not billed tokens.")
    if args.dry_run:
        return
    if args.stub is None and args.budget <= 0:
        parser.error("set --budget to total attempts for this run directory")
    out = args.out / (config["name"] + ("_" + args.stub if args.stub else ""))
    out.mkdir(parents=True, exist_ok=True)
    # Local process lock; the notebook also forbids sharing an output directory across runtimes.
    import fcntl
    with (out / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another process owns this output directory") from None
        with (out / "provenance.jsonl").open("a") as log:
            log.write(json.dumps(provenance("covert_audit") | {"audit_sha": audit_sha()}) + "\n")
        budget = Budget(args.budget, attempts_logged(out))
        try:
            results = asyncio.run(run_batch(config, out, budget, args.stub))
        finally:
            from sprint.covert_analyze import export_tables
            export_tables([out], out)
        print(f"{sum(r['status'] == 'answered' for r in results)}/{len(results)} answered; "
              f"{budget.used} attempts used; {out}")
        if any(r["status"] != "answered" for r in results):
            raise RuntimeError("incomplete or failed judgments retained; inspect accounting before resuming")


if __name__ == "__main__":
    main()
