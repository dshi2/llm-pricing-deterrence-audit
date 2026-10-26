"""Continue finished pilots into the full run: same folder, same code and settings, more seeds.

Seeds 1-4 are the pilot; the continuation adds seeds 5..N in the same run directory, so the two pool
in one analysis. N is fixed once per model from measured pilot cost per attempt, the slowest pilot
pace scaled to a full eligible game, measured rate limiting, the key's remaining allowance and the collection cutoff, never from S, J or M.
"""

import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from itertools import pairwise
from pathlib import Path

from sprint import analyze
from sprint.run import code_sha, header, load_config, provenance
from sprint.study import ANALYSIS

FAMILIES = ("deepseek41", "qwen38", "mistral3", "gptoss120")
ATTEMPTS_PER_GAME = 875  # the pilot's allowance: 3,500 attempts for four games
CONCURRENCY = 8          # games at once per model; all continued models run together
MAX_RATE_LIMITED = 0.01  # more pilot attempts than this answered HTTP 429: keep the pilot's concurrency
MAX_SEEDS = 24
COST_RESERVE = 0.9       # plan new games against 90% of the key's remaining allowance
TIME_MARGIN = 1.5        # assume each new game takes 1.5x a full eligible game at the slowest pilot pace
RUNNING_MINUTES = 15     # pilot files touched this recently: a pilot may still be writing
PAUSE_MINUTES = 10       # longer gaps between a game's calls are pauses (e.g. resumed after credit ran out)
SETTLE_MINUTES = 10      # quiet time after the last pilot finishes, so its report cell is done


def config_text(pilot, seeds, concurrency=CONCURRENCY):
    """Materialise defaults too, so later default changes cannot alter the rules. Same name: same folder."""
    sections = {"": {"name": pilot["name"], "experiment": "main", "seeds": seeds, "concurrency": concurrency},
                "model": asdict(pilot["model"]), "rules": asdict(pilot["rules"])}
    lines = []
    for section, values in sections.items():
        if section:
            lines += ["", f"[{section}]"]
        lines += [f"{key} = {json.dumps(value)}" for key, value in values.items() if value is not None]
    return "\n".join(lines) + "\n"


def family_state(root, family, pilot):
    """'ready', 'unfinished' (resumable in the pilot notebook) or 'excluded: <reason>' (pilot-only)."""
    root = Path(root)
    canary, probe, run_dir = root / f"canary_{family}", root / f"main_{family}_probe", root / pilot["name"]
    if not canary.exists() or not probe.exists():
        return "unfinished"
    checks = analyze.launch_checks(canary, probe)
    if not checks["canary game completed"] or not checks["context probe answered and parsed"]:
        return "unfinished"
    if not all(checks.values()):
        return "excluded: launch checks failed"
    table = analyze.games([run_dir])
    games = table[table.seed.isin(pilot["seeds"])] if not table.empty else table
    if len(games) < len(pilot["seeds"]) or games.status.isin(["incomplete"]).any():
        return "unfinished"
    if not games.status.isin(["eligible", "ineligible"]).all():
        return "excluded: failed pilot game"
    return "ready"


def pilot_status(root, families, config_root=Path("configs"), now=None):
    """'running' while any pilot may still write (or has not started), else 'finished' or 'stopped'."""
    root, now = Path(root), time.time() if now is None else now
    states = {}
    for f in families:
        try:
            states[f] = family_state(root, f, load_config(Path(config_root) / f"main_pilot_{f}.toml"))
        except ValueError:  # a journal caught mid-write or mid-sync; look again next poll
            states[f] = "unfinished"
    touched = max((p.stat().st_mtime for p in root.rglob("*") if p.is_file()), default=now)
    unfinished = [f for f, s in states.items() if s == "unfinished"]
    quiet = (now - touched) / 60  # nothing written yet counts as just started
    if quiet < (RUNNING_MINUTES if unfinished else SETTLE_MINUTES):
        state = "running"
    else:
        state = "stopped" if unfinished else "finished"
    return {"state": state, "families": states, "unfinished": unfinished, "quiet_minutes": round(quiet, 1)}


def reply_rounds(rules):
    """Rounds that wait for model replies in an eligible game, the longest kind: parent, reference, the six
    branches with a live rival and the repeat. The two branches with no live rival are scripted and instant."""
    return rules.parent_rounds + 8 * rules.horizon


def measure(root, pilot):
    """Mean recorded cost per attempt, game duration and share of attempts rate-limited, from the pilot's
    own logs. Duration is the slowest pilot pace per reply round times a full eligible game, so a pilot of
    short ineligible games cannot understate it. 429 retries also use up attempts: not only slower."""
    directory = Path(root) / pilot["name"]
    for seed in pilot["seeds"]:
        saved = analyze._events(directory / f"game_{seed}.jsonl")[0]
        if any(saved.get(key) != value for key, value in header(pilot, seed, False).items()):
            raise ValueError(f"{directory}: pilot settings or collection code differ; this is not the same protocol")
    costs, hours, paces, limited, attempts = [], [], [], 0, 0
    for seed in pilot["seeds"]:
        calls = analyze._events(directory / f"game_{seed}.calls.jsonl")
        costs += [c["cost"] for c in calls if c.get("cost") is not None]
        limited += sum(c.get("http_status") == 429 for c in calls)
        attempts += len(calls)
        gaps = [(b - a).total_seconds() for a, b in pairwise(sorted(datetime.fromisoformat(c["time"]) for c in calls))]
        hours.append(sum(g for g in gaps if g <= 60 * PAUSE_MINUTES) / 3600)
        played = {(e["branch"], e["round"]) for e in analyze._events(directory / f"game_{seed}.jsonl")
                  if e["event"] == "decision"}
        paces.append(hours[-1] / len(played))
    if not costs or any(not math.isfinite(x) or x < 0 for x in costs):
        raise ValueError(f"{directory}: pilot costs must be recorded and finite")
    return {"cost_per_attempt_usd": sum(costs) / len(costs), "slowest_pilot_game_hours": max(hours),
            "game_hours": max(paces) * reply_rounds(pilot["rules"]),
            "recorded_cost_usd": sum(costs), "rate_limited_share": limited / attempts}


def allocate(game_costs, seed_caps, money, start):
    """Add one game per model in turns, cheapest first, while money and each model's time cap allow."""
    seeds, spend, grew = dict(start), 0.0, True
    while grew:
        grew = False
        for family in sorted(game_costs, key=game_costs.get):
            if seeds[family] < seed_caps[family] and spend + game_costs[family] <= money:
                seeds[family] += 1
                spend += game_costs[family]
                grew = True
    return seeds, spend


def prepare(root, families, remaining_usd, cutoff, config_root=Path("configs"), now=None):
    """Freeze N per model, configs and attempt caps once, before any continuation call; resume reuses them."""
    root, config_root = Path(root), Path(config_root)
    manifest = root / "continue_plan.json"
    pilots = {f: load_config(config_root / f"main_pilot_{f}.toml") for f in families}
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["collection_code_sha"] != code_sha():
            raise ValueError("Collection code differs from the frozen plan; resume with the original commit")
    else:
        status = pilot_status(root, families, config_root, now)
        if status["state"] != "finished":
            raise ValueError(f"Pilots not finished: {status}")
        ready = [f for f, s in status["families"].items() if s == "ready"]
        measured = {f: measure(root, pilots[f]) for f in ready}
        hours_left = (datetime.fromisoformat(cutoff).timestamp() - (time.time() if now is None else now)) / 3600
        start = {f: len(pilots[f]["seeds"]) for f in families}
        waves = {f: max(0, math.floor(hours_left / (TIME_MARGIN * m["game_hours"]))) for f, m in measured.items()}
        concurrency = {f: CONCURRENCY if m["rate_limited_share"] <= MAX_RATE_LIMITED else pilots[f]["concurrency"]
                       for f, m in measured.items()}
        caps = {f: min(MAX_SEEDS, start[f] + concurrency[f] * waves[f]) for f in ready}
        game_costs = {f: m["cost_per_attempt_usd"] * ATTEMPTS_PER_GAME for f, m in measured.items()}
        counts, spend = allocate(game_costs, caps, COST_RESERVE * remaining_usd, {f: start[f] for f in ready})
        plan = {}
        for family in families:
            n, pilot_seeds = counts.get(family, start[family]), pilots[family]["seeds"]
            seeds = pilot_seeds + list(range(max(pilot_seeds) + 1, max(pilot_seeds) + 1 + n - start[family]))
            reason = status["families"][family] if family not in ready else "pilot only: no time or money for a game"
            plan[family] = {"status": "continued" if n > start[family] else reason,
                            "seeds": seeds, "attempt_budget": ATTEMPTS_PER_GAME * n,
                            "config": config_text(pilots[family], seeds, concurrency.get(family, CONCURRENCY))}
        saved = {"families": plan, "collection_code_sha": code_sha(), "analysis": ANALYSIS,
                 "rule": {"attempts_per_game": ATTEMPTS_PER_GAME, "concurrency": CONCURRENCY,
                          "max_rate_limited_share": MAX_RATE_LIMITED, "max_seeds": MAX_SEEDS, "cost_reserve": COST_RESERVE, "time_margin": TIME_MARGIN},
                 "cutoff": cutoff, "hours_left_at_freeze": hours_left, "remaining_allowance_usd": remaining_usd,
                 "pilot_measurements": measured, "projected_new_cost_usd_not_guaranteed": spend,
                 "frozen_at": provenance("freeze main-study continuation")}
        root.mkdir(parents=True, exist_ok=True)
        temp = manifest.with_suffix(".tmp")
        temp.write_text(json.dumps(saved, indent=2) + "\n")
        temp.replace(manifest)
    config_dir = root / "configs"
    config_dir.mkdir(exist_ok=True)
    for family, entry in saved["families"].items():
        path = config_dir / f"main_continue_{family}.toml"
        if path.exists() and path.read_text() != entry["config"]:
            raise ValueError(f"Frozen config modified: {path}")
        path.write_text(entry["config"])
    return saved
