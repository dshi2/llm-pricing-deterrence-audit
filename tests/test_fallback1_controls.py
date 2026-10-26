import asyncio
from types import SimpleNamespace

import pytest

from scripts.check_fallback1_controls import CONTROLS, control_policy, run_controls
from sprint.fallback1 import ReplacementRules
from sprint.fallback1_run import load_config
from sprint.run import load_config as main_config


def test_controls_find_removal_and_context_effects_without_label_access(tmp_path):
    records = run_controls(tmp_path)
    prices = {(r['control'], r['contrast']): r['observed'] for r in records
              if r['outcome'] == 'mean_price'}
    assert all(value == 0 for (control, _), value in prices.items() if control == 'constant')
    assert prices['price_responsive', 'replacement_memory'] == pytest.approx(-.09)
    assert prices['price_responsive', 'removal_context_interaction'] == pytest.approx(0)
    assert prices['reset_sensitive', 'replacement_memory'] == pytest.approx(0)
    assert prices['reset_sensitive', 'removal_context_interaction'] == pytest.approx(-.09)
    assert len(records) == 18
    assert (tmp_path / 'controls.csv').exists()


@pytest.mark.parametrize('kind', CONTROLS)
def test_policy_observes_history_not_treatment_labels(kind):
    policy = control_policy(kind, ReplacementRules())
    request = SimpleNamespace(round_number=101, plans=None, history=[])
    left = asyncio.run(policy(request, 'kept_memory_neutral:101:0'))
    right = asyncio.run(policy(request, 'replaced_reset_truth:101:0'))
    assert left == right


def test_fallback_deepseek_endpoint_matches_main_without_changing_design():
    main = main_config('configs/main_pilot_deepseek41.toml')
    for stage in ('pilot', 'canary'):
        old = load_config(f'configs/fallback1_{stage}_deepseek31.toml')
        new = load_config(f'configs/fallback1_{stage}_deepseek41.toml')
        assert new['model'] == main['model'] and new['rules'] == old['rules']
    canary = [load_config(f'configs/fallback1_canary_{m}.toml') for m in ('deepseek31', 'deepseek41')]
    assert canary[0]['seeds'] == canary[1]['seeds'] and canary[0]['concurrency'] == canary[1]['concurrency']


def test_selected_f1_pilots_share_one_design():
    pilots = [load_config(f'configs/fallback1_pilot_{m}.toml') for m in ('deepseek41', 'qwen38', 'mistral3', 'gptoss120')]
    for config in pilots:
        assert config['seeds'] == list(range(1, 21)) and config['concurrency'] == 10
        assert config['rules'] == pilots[0]['rules']
