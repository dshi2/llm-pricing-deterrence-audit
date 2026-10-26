"""Generate a concurrent Colab launcher, or optional single-family launch copies."""

import argparse
import copy
import json
import re
from pathlib import Path

FAMILIES = ("deepseek41", "qwen38", "mistral3", "gptoss120")

OFFLINE_CHECK = '''OFFLINE_PASSED = False
sh([sys.executable, '-m', 'pytest', '-q'], env={**os.environ, 'MPLBACKEND': 'Agg'})

# Existing pilots belong to their original notebook and code pin. Do not replace them.
for family in FAMILIES:
    target = OUT / f'main_{family}'
    for journal in (OUT.parents[1]).glob(f'**/main_{family}/game_*.jsonl'):
        if journal.parent != target:
            raise RuntimeError(f'Existing {family} pilot found at {journal.parent}. '
                               'Resume its original notebook; do not launch a duplicate here.')
OFFLINE_PASSED = True
print('Offline tests passed. Do not launch another notebook for any family selected here.')
'''

LAUNCH_GUARD = '''if not globals().get('OFFLINE_PASSED') or not globals().get('BUDGET_VERIFIED'):
    raise RuntimeError('Setup, offline tests and shared key budget must pass before model calls.')

'''

CUTOFF = '2026-10-26T15:00:00+05:30'

KEY_READ = """req = urllib.request.Request('https://openrouter.ai/api/v1/key',
    headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']})
with urllib.request.urlopen(req, timeout=30) as response:
    metadata = json.load(response)['data']
print(check_key_budget(metadata))
# Key allowance is a spending cap, not proof of sufficient prepaid account credit.
del metadata, req
"""

# The continuation reads the allowance twice: now, and again when it sizes the run after the pilots finish.
KEY_FUNCTION = """def key_budget():
    req = urllib.request.Request('https://openrouter.ai/api/v1/key',
        headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']})
    with urllib.request.urlopen(req, timeout=30) as response:
        return check_key_budget(json.load(response)['data'])


print(key_budget())
# Key allowance is a spending cap, not proof of sufficient prepaid account credit.
"""

CONTINUE_INTRO = '''# Main study: continue the pilot into the full run

Same folder, same pinned collection code, same settings. Seeds 1–4 are the pilot; this adds seeds 5..N
to each model's existing run directory, so pilot and continuation pool in one analysis.

Upload `collusion-continue.bundle` to Drive's `MyDrive/collusion-sprint` folder. Use the same
`COLLUSION_OPENROUTER_API_KEY` as the pilot: one shared non-resetting $100 ceiling.

Run all. You may start it while `run_pilots.ipynb` is still running: this notebook waits until every
pilot has finished and gone quiet, then fixes N per model **once**, from measured pilot cost per
attempt, the slowest pilot pace scaled to a full eligible game, pilot rate limiting, the key's allowance
read at that moment and the collection cutoff, never from S, J or M. With N frozen, it rehearses the
full-run report on the pilot's real data, so any report problem shows hours before the final tables. A model whose pilot failed its launch checks or lost a game stays pilot-only.
The frozen plan is `continue_plan.json`; a rerun resumes it and never re-sizes.

Then all continued models run together, eight games at a time each (the pilot's four if its calls were
rate-limited), followed by the pooled analysis
(S, J, M and S−J; Bonferroni over 16 intervals) and figures. Failed and unfinished games stay in the
accounting. Never run this notebook in two runtimes at once.
'''

CONTINUE_FREEZE = '''CONTINUE_READY = False
import time
from datetime import UTC, datetime

from scripts import report_figures
from scripts.plot_paths import export as export_paths
from sprint.continuation import pilot_status, prepare
from sprint.study import export


def pooled_report(seeds, stage, analysis, figures, label):
    """The full run's tables and figures. Also rehearsed on the pilot alone, right after N is frozen."""
    run_dirs = [OUT / f'main_{family}' for family in FAMILIES]
    table, primary = export(run_dirs, analysis, stage, seeds)
    report_figures.run('main', analysis, figures, label)
    report_figures.run('diagrams', None, figures)
    export_paths('main', [d for d in run_dirs if d.exists()], figures, label)
    return table, primary


fresh = not (OUT / 'continue_plan.json').exists()
if fresh:
    while (status := pilot_status(OUT, FAMILIES))['state'] == 'running':
        print(datetime.now(UTC).strftime('%H:%M UTC'), 'waiting for pilots:', status['families'], flush=True)
        time.sleep(300)
    if status['state'] == 'stopped':
        raise RuntimeError(f"Pilots stopped unfinished: {status['unfinished']}. Resume run_pilots.ipynb, then rerun.")
# Read the allowance now, not at startup: the pilots kept spending while this notebook waited.
PLAN = prepare(OUT, FAMILIES, key_budget()['remaining_allowance_usd'], COLLECTION_CUTOFF)
for family, entry in PLAN['families'].items():
    print(f"{family}: {entry['status']}; seeds 1-{len(entry['seeds'])}; attempt cap {entry['attempt_budget']}")
print('Projected new cost, not guaranteed: $%.2f' % PLAN['projected_new_cost_usd_not_guaranteed'])
if fresh:  # N is frozen and nothing is writing yet: rehearse the full-run report on the real pilot data
    try:
        pilot_seeds = {f'main_{f}': load_config(Path('configs') / f'main_pilot_{f}.toml')['seeds'] for f in FAMILIES}
        _, rehearsal = pooled_report(pilot_seeds, 'pilot', OUT / 'analysis_pilot_rehearsal',
                                     OUT / 'figures_pilot_rehearsal', 'Main study: pilot only (report rehearsal)')
        display(rehearsal)
    except Exception as error:  # noqa: BLE001 -- figures never block collection; the report rebuilds offline
        print(f'Report rehearsal failed: {error!r}. Collection goes ahead; fix the report offline.')
CONTINUE_READY = True
'''

CONTINUE_LAUNCH = '''if not globals().get('CONTINUE_READY'):
    raise RuntimeError('Freeze the continuation plan first.')
from concurrent.futures import ThreadPoolExecutor, as_completed

from sprint import analyze

CONTINUED = [f for f, entry in PLAN['families'].items() if entry['status'] == 'continued']


# Budgets count ALL attempts already logged in the folder, pilot included, so reruns never renew them.
def run_family(family):
    config = str(OUT / 'configs' / f'main_continue_{family}.toml')
    endpoint_snapshot(load_config(config))  # same anchor as the pilot: the endpoint must not have changed
    console = OUT / f'{family}.continue.console.log'
    print(f'{family}: continuing; progress and errors -> {console}', flush=True)
    with console.open('a', buffering=1) as log:
        sh([sys.executable, '-u', '-m', 'sprint.run', config, '--out', str(OUT),
            '--budget', str(PLAN['families'][family]['attempt_budget'])], stdout=log, stderr=subprocess.STDOUT)
    endpoint_snapshot(load_config(config))
    games, seeds = analyze.games([OUT / f'main_{family}']), set(PLAN['families'][family]['seeds'])
    done = set() if games.empty else set(games.seed[games.status.isin(['eligible', 'ineligible'])])
    if done != seeds:  # interrupted games also exit normally
        counts = {} if games.empty else games.status.value_counts().to_dict()
        raise RuntimeError(f'{family}: {len(done)} of {len(seeds)} games complete ({counts}). '
                           'Rerun to resume interrupted games; failed games stay failed.')
    print(f'{family}: finished', flush=True)
    return family

failures = {}
with ThreadPoolExecutor(max_workers=max(1, len(CONTINUED))) as pool:
    jobs = {pool.submit(run_family, family): family for family in CONTINUED}
    for future in as_completed(jobs):
        family = jobs[future]
        try:
            future.result()
        except Exception as error:  # noqa: BLE001 -- report each failed family, then raise after exporting
            failures[family] = str(error)
            print(f'{family}: STOPPED: {error}', flush=True)
print('Family failures:', failures or 'none')
'''

CONTINUE_REPORT = '''from IPython.display import Image

figures = OUT / 'figures_pooled'
table, primary = pooled_report({f'main_{f}': entry['seeds'] for f, entry in PLAN['families'].items()}, 'pooled',
                               OUT / 'analysis_pooled', figures, 'Main study: pilot + continuation, one protocol')
display(table.groupby(['run', 'status']).size().rename('games'))
display(primary)
for figure in sorted(figures.glob('main_primary_*.png')):
    display(Image(filename=str(figure)))
print('Pooled tables:', OUT / 'analysis_pooled', '| figures:', figures)
if failures:
    raise RuntimeError('Some families stopped; inspect their console logs. Saved games remain in the tables.')
'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Notebook template changed; expected exactly one {old!r}")
    return text.replace(old, new)


def build(template, family, commit, stage="pilot"):
    if stage not in ("pilot", "continue") or (stage == "continue" and family != "all"):
        raise ValueError("stage must be pilot, or continue for the combined 'all' launcher")
    if family not in (*FAMILIES, "all") or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use a selected family and a full Git commit hash")
    selected = list(FAMILIES) if family == "all" else [family]
    notebook = copy.deepcopy(template)
    cells = notebook["cells"]
    intro = f"""# {family}: parallel main-study pilot

Generated from `run_models.ipynb`. Upload `collusion-parallel.bundle` to Drive's
`MyDrive/collusion-sprint` folder. Use the same OpenRouter key in every runtime,
with a **combined non-resetting limit of at most $100**. Add it to Colab Secrets
as `COLLUSION_OPENROUTER_API_KEY`. This must be your authorised personal/project key,
not an organisation's unrelated key. CPU runtime is sufficient; inference is billed by OpenRouter.

Run all: offline tests → canary → full-context probe → launch checks → competence
checks → four-market pilot → CSV tables. No expansion is launched.
Per family: at most 2,922 successful calls, with 3,670 attempts allowed across all stages.

Never launch the same family twice. The existing-run check is not a distributed lock. If you already
started this family elsewhere, keep its original notebook and journals.
Resume this run using this same notebook, commit and output folder.
"""
    if family == "all":
        intro += "\nAll four families run here at once; each runs four markets concurrently.\n"
        intro += "Do not also run the single-family notebooks. Rounds within a market remain sequential.\n"
    cells[0]["source"] = intro.splitlines(keepends=True)
    setup = "".join(cells[1]["source"])
    setup = "OFFLINE_PASSED = BUDGET_VERIFIED = False\n" + setup
    setup = replace_once(setup, "collusion-models.bundle", "collusion-parallel.bundle")
    setup = re.sub(r"(?m)^COMMIT = '[0-9a-f]+'$", f"COMMIT = '{commit}'", setup)
    setup = replace_once(setup, "Path('/content/collusion-models')",
                         f"Path('/content/collusion-parallel-{family}')")
    setup = replace_once(setup, "runs/four_models_v1", f"runs/parallel_v1/{family}")
    setup = replace_once(setup, "FAMILIES = ['deepseek31', 'qwen38', 'mistral3', 'gptoss120']",
                         f"FAMILIES = {selected!r}")
    setup = replace_once(setup, "'-e', '.',", "'-e', '.[dev]',")
    setup = replace_once(setup, "userdata.get('OPENROUTER_API_KEY')",
                         "userdata.get('COLLUSION_OPENROUTER_API_KEY')")
    cells[1]["source"] = setup.splitlines(keepends=True)
    budget = "BUDGET_VERIFIED = False\n" + "".join(cells[2]["source"])
    cells[2]["source"] = (budget + "\nBUDGET_VERIFIED = True\n").splitlines(keepends=True)
    launch = "".join(cells[4]["source"])
    launch = replace_once(launch, "        command('sprint.run', pilot, 10, '--context-probe')\n",
        "        probe_dir = OUT / (load_config(pilot)['name'] + '_probe')\n"
        "        if not analyze.launch_checks(OUT / f'canary_{family}', probe_dir)['context probe answered and parsed']:\n"
        "            command('sprint.run', pilot, 10, '--context-probe')\n")
    launch = replace_once(launch, "        command('sprint.run', pilot, 3500)\n",
        "        command('sprint.run', pilot, 3500)\n"
        "        collected = analyze.games([OUT / load_config(pilot)['name']])\n"
        "        expected_seeds = set(load_config(pilot)['seeds'])\n"
        "        if collected.empty or set(collected.seed) != expected_seeds or not collected.status.isin(['eligible', 'ineligible']).all():\n"
        "            raise RuntimeError(f'{family}: collection incomplete or failed; saved games will be reported, not replaced.')\n")
    workers = len(FAMILIES) if family == "all" else 1  # all four pilots at once: the pilot takes one game-length
    launch = replace_once(launch, "ThreadPoolExecutor(max_workers=2)", f"ThreadPoolExecutor(max_workers={workers})")
    cells[4]["source"] = (LAUNCH_GUARD + launch).splitlines(keepends=True)
    report = "".join(cells[5]["source"])
    report = report.replace("if failures:\n", "from sprint.study import export\n"
                            "table, primary = export(run_dirs, OUT / 'analysis', 'pilot', [1, 2, 3, 4])\n"
                            "display(primary)\n"
                            "from IPython.display import Image\n"
                            "for figure in sorted((OUT / 'analysis').glob('*/gains.png')):\n"
                            "    display(Image(filename=str(figure)))\n"
                            "if failures:\n")
    cells[5]["source"] = report.splitlines(keepends=True)
    cells.insert(2, {"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": OFFLINE_CHECK.splitlines(keepends=True)})
    if stage == "continue":
        cells[0]["source"] = CONTINUE_INTRO.splitlines(keepends=True)
        setup = "".join(cells[1]["source"])
        setup = replace_once(setup, "collusion-parallel.bundle", "collusion-continue.bundle")
        setup = replace_once(setup, "/content/collusion-parallel-all", "/content/collusion-continue")
        setup += f"\nCOLLECTION_CUTOFF = '{CUTOFF}'  # new games must be projected to finish by then\n"
        cells[1]["source"] = setup.splitlines(keepends=True)
        cells[2]["source"] = ("OFFLINE_PASSED = False\n"
            "sh([sys.executable, '-m', 'pytest', '-q'], env={**os.environ, 'MPLBACKEND': 'Agg'})\n"
            "OFFLINE_PASSED = True\n").splitlines(keepends=True)
        cells[3]["source"] = replace_once("".join(cells[3]["source"]), KEY_READ, KEY_FUNCTION).splitlines(keepends=True)
        cells[5:] = [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": source.splitlines(keepends=True)}
                     for source in (CONTINUE_FREEZE, LAUNCH_GUARD + CONTINUE_LAUNCH, CONTINUE_REPORT)]
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"], cell["outputs"] = None, []
            compile("".join(cell["source"]), f"run_{family}.ipynb", "exec")
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, default=Path("dist/parallel"))
    parser.add_argument("--separate", action="store_true", help="Generate one notebook per family instead")
    parser.add_argument("--stage", choices=("pilot", "continue"), default="pilot")
    args = parser.parse_args()
    template = json.loads(Path("notebooks/run_models.ipynb").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES if args.separate and args.stage == "pilot" else ("all",):
        path = args.out / (("run_pilots.ipynb" if args.stage == "pilot" else "run_continue.ipynb")
                           if family == "all" else f"run_{family}_{args.stage}.ipynb")
        path.write_text(json.dumps(build(template, family, args.commit, args.stage), indent=1) + "\n")
        print(path)
