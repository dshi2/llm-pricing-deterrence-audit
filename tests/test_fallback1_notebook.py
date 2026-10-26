"""Notebook orchestration and launch gate checks without Google credentials or paid calls."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_client import CONFIG, FakeClient, ok

from sprint.client import Budget, LLMPolicy
from sprint.engine import Journal
from sprint.fallback1 import ReplacementRules, replacement_game
from sprint.fallback1_analyze import launch_checks
from sprint.fallback1_run import full_context_request, header
from sprint.market import Market

NOTEBOOK = Path('notebooks/fallback1.ipynb')


def function(name, ns):
    for cell in json.loads(NOTEBOOK.read_text())['cells']:
        if cell['cell_type'] == 'code':
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), 'exec'), ns)  # noqa: S102
                    return ns[name]
    raise AssertionError(name)


def test_notebook_compiles_and_uses_separate_paths():
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code': compile(''.join(cell['source']),str(NOTEBOOK),'exec')
    source = ''.join(''.join(c['source']) for c in notebook['cells'])
    assert "userdata.get('COLLUSION_F1_KEY')" in source
    assert "COLLUSION_OPENROUTER_API_KEY" not in source and "userdata.get('OPENROUTER_API_KEY')" not in source
    assert "FAMILIES = ['deepseek41', 'qwen38', 'mistral3', 'gptoss120']" in source
    assert 'runs/fallback1_v2' in source and 'sprint.fallback1_run' in source
    assert 'sprint.run\'' not in source



def test_key_check_rejects_a_key_above_the_f1_allocation():
    check = function('check_key_budget', {})
    assert check({'limit': 60, 'limit_remaining': 60, 'limit_reset': None})['key_limit_usd'] == 60
    with pytest.raises(RuntimeError, match='separate F1 key'):
        check({'limit': 100, 'limit_remaining': 100, 'limit_reset': None})  # e.g. the main study's key

@pytest.mark.parametrize('offline,budget', [(False, False), (False, True), (True, False)])
def test_offline_or_budget_failure_prevents_paid_launch(offline, budget):
    cell = json.loads(NOTEBOOK.read_text())['cells'][5]
    guard = ast.parse(''.join(cell['source'])).body[0]
    with pytest.raises(RuntimeError, match='before model calls'):
        exec(compile(ast.Module(body=[guard], type_ignores=[]), str(NOTEBOOK), 'exec'),  # noqa: S102
             {'OFFLINE_PASSED': offline, 'BUDGET_VERIFIED': budget})


def test_canary_failure_blocks_pilot(tmp_path):
    sent = []
    run = function('run_family', {'OUT':tmp_path, 'load_config':lambda _: {},
        'endpoint_snapshot':lambda _:None,'sys':SimpleNamespace(executable='python'),
        'subprocess':SimpleNamespace(STDOUT=-2),'sh':lambda args,**kw:sent.append(args),
        'fallback1_analyze':SimpleNamespace(launch_checks=lambda *args: {'completed':False})})
    with pytest.raises(RuntimeError,match='launch checks failed'): run('deepseek31')
    assert len(sent)==2 and '--context-probe' in sent[1]


def test_two_seller_canary_and_probe_gate(tmp_path):
    import asyncio
    config = {'experiment':'replacement','rules':ReplacementRules(parent_rounds=6,horizon=2,restraint_window=4),
              'model':CONFIG}
    canary,probe=tmp_path/'canary',tmp_path/'probe'
    async def run():
        policy=LLMPolicy(CONFIG,Market(2),Budget(50),canary/'game_99.calls.jsonl',client=FakeClient([ok()]*50))
        await replacement_game(policy,99,Journal(canary/'game_99.jsonl',header(config,99,False)),
                               rules=config['rules'])
        pp=LLMPolicy(CONFIG,Market(2),Budget(10),probe/'probe.calls.jsonl',client=FakeClient([ok()]))
        await pp(full_context_request(config),'context_probe')
    asyncio.run(run())
    assert all(launch_checks(canary,probe).values())
    path=canary/'game_99.jsonl'
    path.write_text(''.join(l+'\n' for l in path.read_text().splitlines() if json.loads(l)['event']!='complete'))
    assert not launch_checks(canary,probe)['canary game completed']
