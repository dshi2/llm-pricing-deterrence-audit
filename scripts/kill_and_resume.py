"""Offline evidence that a killed run resumes exactly: SIGKILL a --stub batch several times mid-game,
rerun it to completion, and compare every game with an uninterrupted control run.

    python scripts/kill_and_resume.py configs/main_pilot_deepseek.toml /tmp/kill_check
"""

import json
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def run(config, out, kill_at_lines=None):
    """Run the stub batch; if kill_at_lines is given, SIGKILL it once game 1's journal reaches that size."""
    cmd = [sys.executable, "-m", "sprint.run", config, "--stub", "--out", str(out)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    journal = next(iter(out.glob("*/game_1.jsonl")), None) if out.exists() else None
    while kill_at_lines and proc.poll() is None:
        journal = journal or next(iter(out.glob("*/game_1.jsonl")), None)
        if journal and len(journal.read_bytes().splitlines()) >= kill_at_lines:
            proc.send_signal(signal.SIGKILL)
            break
        time.sleep(0.005)
    _, err = proc.communicate()
    if not kill_at_lines and proc.returncode:
        sys.exit(err.decode())
    return proc.returncode


def games(out):
    """Per game: its result, its committed rounds by (branch, number), and saved-decision counts."""
    found = {}
    for path in sorted(out.glob("*/game_*.jsonl")):
        if path.name.endswith(".calls.jsonl"):
            continue
        events = [json.loads(line) for line in path.read_text().splitlines()]
        found[path.name] = {
            "result": next(e["result"] for e in events if e["event"] == "complete"),
            "rounds": {(e["branch"], e["row"]["number"]): e["row"] for e in events if e["event"] == "round"},
            "decisions": Counter((e["branch"], e["round"], e["seller"]) for e in events if e["event"] == "decision"),
            "resumes": sum(e["event"] == "resumed" for e in events),
        }
    return found


def main(config, base):
    base = Path(base)
    if base.exists() and any(base.iterdir()):
        sys.exit(f"{base} is not empty")
    run(config, base / "control")
    for lines in (250, 600, 900):  # three kills: parent game, reference and branches (a game is ~1,060 lines)
        code = run(config, base / "killed", lines)
        assert code == -signal.SIGKILL, f"the kill at {lines} lines did not happen (exit {code})"
        print(f"killed at {lines} journal lines of game 1")
    run(config, base / "killed")
    control, killed = games(base / "control"), games(base / "killed")
    assert control.keys() == killed.keys() and control
    for name in control:
        c, k = control[name], killed[name]
        assert c["result"] == k["result"], f"{name}: results differ"
        assert c["rounds"] == k["rounds"], f"{name}: committed rounds differ"
        assert max(k["decisions"].values()) == 1, f"{name}: a saved decision was requested twice"
        print(f"{name}: identical result and {len(k['rounds'])} identical rounds; "
              f"resumed {k['resumes']} times; R1 = {k['result'].get('R1')}")
    print("PASS")


if __name__ == "__main__":
    main(*sys.argv[1:3])
