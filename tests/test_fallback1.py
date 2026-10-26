"""Scientific and recovery checks for controlled replacement, without API calls."""

import asyncio
import itertools
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sprint.client import Budget, Declined, Fatal, Interrupted
from sprint.engine import Decision, Journal
from sprint.fallback1 import (
    CELLS,
    ReplacementRules,
    dynamic_benchmark,
    next_state,
    notices,
    replacement_game,
    state_after,
)
from sprint.fallback1_analyze import summary, tables
from sprint.fallback1_run import full_context_request, load_config, run_batch
from sprint.market import Market
from sprint.scripted import as_policy, fixed, punisher

RULES = ReplacementRules(parent_rounds=6, horizon=3, restraint_window=4)


def run(tmp_path, policy=None, rules=RULES, seed=7):
    return asyncio.run(replacement_game(policy or as_policy(fixed(1.75)), seed,
                                       Journal(tmp_path, {"test": "replacement"}), rules=rules))


def independent_profit(a, b):
    ea, eb, outside = math.exp((2-a)/.25), math.exp((2-b)/.25), 1
    return 100 * ea / (outside + ea + eb) * (a-1)


def test_benchmark_matches_independent_grid_dynamic_program():
    market = Market(2)
    rules = ReplacementRules(horizon=20)
    benchmark = dynamic_benchmark(market, rules)
    # Independent dense-grid Bellman recursion and independent demand expression.
    prices = np.unique(np.r_[np.linspace(1, 4, 6001), rules.trigger,
                             market.best_response(0, (1, rules.calm)),
                             market.best_response(0, (1, market.nash()))])
    values = np.zeros(6)
    for _ in range(20):
        new = []
        for state in range(6):
            rival = market.nash() if state else rules.calm
            rewards = np.array([independent_profit(p, rival) for p in prices])
            next_values = values[state-1] if state else np.where(prices < 1.7, values[5], values[0])
            new.append(float(np.max(rewards + .95 * next_values)))
        values = np.array(new)
    assert benchmark['value'] == pytest.approx(values[0], abs=1e-7)
    assert benchmark['fixed_best_response'] == pytest.approx(1.5928, abs=1e-4)
    # Finite horizon: the last calm round has no future punishment cost.
    assert benchmark['policies_by_remaining_horizon'][0][0] < rules.trigger
    assert benchmark['value'] >= sum(.95**t * independent_profit(1.7, 1.75) for t in range(20))


def test_short_horizon_benchmark_matches_exhaustive_paths():
    rules, market = replace(RULES, horizon=4), Market(2)
    bench = dynamic_benchmark(market, rules)
    actions = (bench['fixed_best_response'], rules.trigger, market.best_response(0, (1, market.nash())))
    best = -math.inf
    for seq in itertools.product(actions, repeat=4):
        state, value = 0, 0
        for t, a in enumerate(seq):
            b = market.nash() if state else rules.calm
            value += rules.discount**t * independent_profit(a, b)
            state = state-1 if state else (5 if a < 1.7 else 0)
        best = max(best, value)
    assert bench['value'] == pytest.approx(best)


def test_punishment_timing_and_no_retrigger():
    rules = ReplacementRules()
    strategy = punisher(1.75, 1.7, 1.4729, 5)
    history = []
    states = []
    for _ in range(8):
        states.append(strategy(history, 1))
        history.append((1.6, states[-1]))
    assert states == [1.75] + [1.4729]*5 + [1.75, 1.4729]
    assert next_state(0, 1.7, rules) == 0
    assert state_after([SimpleNamespace(prices=p) for p in history], rules) == 4


def test_full_block_calls_and_independent_scores(tmp_path):
    calls = []
    async def policy(request, label):
        calls.append((request, label))
        return Decision(1.75, 'saved plan', 'saved insight')
    rules = ReplacementRules()
    result = run(tmp_path/'g.jsonl', policy, rules)
    assert len(calls) == rules.planned_calls()['maximum'] == 260
    assert result['checkpoint']['restrained']
    assert set(result['paths']) == set(CELLS) | {'fresh_fixed', 'kept_memory_neutral_repeat'}
    expected = sum(.95**t * independent_profit(1.75, 1.75) for t in range(20))
    assert all(p['discounted_profit'] == pytest.approx(expected) for p in result['paths'].values())
    assert result['paths']['kept_memory_neutral'] == result['paths']['kept_memory_neutral_repeat']


def test_reset_disclosure_isolation_and_price_continuity(tmp_path):
    seen = []
    async def policy(request, label):
        seen.append((request, label))
        return Decision(1.75, label, label)
    result = run(tmp_path/'g.jsonl', policy)
    first = {label.split(':')[0]: req for req, label in seen if req.round_number == 7}
    for cell, (_, reset, truthful) in CELLS.items():
        req = first[cell]
        assert [h.number for h in req.history] == ([6] if reset else list(range(1,7)))
        assert req.plans == (None if reset else 'parent:6:0')
        assert ('regardless' in req.notice) == truthful
    assert len({first[c].seed for c in CELLS}) == 1
    for req, label in seen:
        if req.round_number == 8:
            assert req.plans.startswith(label.split(':')[0] + ':7:')
    events = [json.loads(x) for x in (tmp_path/'g.jsonl').read_text().splitlines()]
    rows = [e['row'] for e in events if e['event']=='round' and e['row']['number']==7]
    assert all(r['prices'][1] == 1.75 for r in rows)
    fresh = next(r for r,l in seen if l=='fresh_fixed:1:0')
    assert not fresh.history and fresh.plans is None
    assert result['checkpoint']['calm']
    neutral, truth = notices(Market(2), RULES)
    assert len(neutral.split()) == len(truth.split())


def test_active_checkpoint_keeps_fresh_baseline_without_replacement(tmp_path):
    async def policy(request, label):
        return Decision(1.6 if label=='parent:6:0' else 1.75)
    result = run(tmp_path/'g.jsonl', policy)
    assert not result['branches_run'] and not result['checkpoint']['restrained']
    assert set(result['paths']) == {'fresh_fixed'}


def test_calm_unrestrained_parent_still_runs_adjustment_branches(tmp_path):
    # Cut in first calm round, finish five punishment rounds; no restraint but next state is calm.
    result = run(tmp_path/'g.jsonl', as_policy(fixed(1.6)))
    assert result['branches_run'] and not result['checkpoint']['restrained']
    assert result['checkpoint']['next_punishment_remaining'] == 0


@pytest.mark.parametrize('stop', [2, 7, 12, 25])
def test_resume_no_repeated_successful_decisions(tmp_path, stop):
    control = run(tmp_path/'control.jsonl')
    calls = []
    async def interrupted(request, label):
        calls.append(label)
        if len(calls)==stop:
            raise Interrupted('offline disconnect')
        return Decision(1.75)
    with pytest.raises(Interrupted):
        run(tmp_path/'g.jsonl', interrupted)
    resumed = []
    async def policy(request, label):
        resumed.append(label)
        return Decision(1.75)
    assert run(tmp_path/'g.jsonl', policy) == control
    assert not set(calls[:-1]) & set(resumed)
    assert len(calls)-1+len(resumed) == RULES.planned_calls()['maximum']


def test_runner_analysis_failures_and_pairing(tmp_path):
    config = load_config('configs/fallback1_pilot_deepseek31.toml')
    config |= {'rules': RULES, 'seeds': [1,2,3], 'concurrency': 1}
    def factory(seed):
        async def policy(req, label):
            if seed==2: raise Declined('declined')
            if seed==3: raise Interrupted('budget exhausted')
            return Decision(1.75)
        return policy
    asyncio.run(run_batch(config,tmp_path,Budget(0),True,factory))
    games, paths, contrasts = tables([tmp_path])
    assert games.status.tolist() == ['complete','failed','incomplete']
    assert len(contrasts) == 20 and contrasts.seed.eq(1).all()
    assert contrasts.difference.eq(0).all()
    assert summary(contrasts).low.isna().all()  # n=1 is not certainty
    assert len(paths) == 8
    assert full_context_request(config).history[0].prices == pytest.approx((1.601,1.7111))


def test_fatal_logged_and_not_silently_counted(tmp_path):
    config = load_config('configs/fallback1_pilot_deepseek31.toml')
    config |= {'rules': RULES, 'seeds': [1]}
    async def fatal(req, label): raise Fatal('wrong provider')
    with pytest.raises(Fatal):
        asyncio.run(run_batch(config,tmp_path,Budget(0),True,lambda _:fatal))
    assert tables([tmp_path])[0].status.tolist() == ['incomplete']


def test_invalid_rules():
    for kw in ({'restraint_window':101},{'trigger':float('nan')},{'discount':0},{'horizon':0}):
        with pytest.raises(ValueError): ReplacementRules(**kw)


def test_nonzero_replacement_effect_matches_independent_payoff_paths(tmp_path):
    async def cutter(request, label):
        return Decision(1.75 if request.round_number <= RULES.parent_rounds else 1.6)
    result = run(tmp_path/'g.jsonl', cutter)
    nash = Market(2).nash()
    kept = independent_profit(1.6, 1.75) + sum(.95**t * independent_profit(1.6, nash) for t in (1,2))
    removed = sum(.95**t * independent_profit(1.6, 1.75) for t in range(3))
    assert removed > kept
    assert result['paths']['kept_memory_neutral']['discounted_profit'] == pytest.approx(kept)
    assert result['paths']['replaced_memory_neutral']['discounted_profit'] == pytest.approx(removed)
    events = [json.loads(x) for x in (tmp_path/'g.jsonl').read_text().splitlines()]
    for branch, expected in [('kept_memory_neutral',[1.75,nash,nash]),
                             ('replaced_memory_neutral',[1.75]*3)]:
        rival = [e['row']['prices'][1] for e in events if e['event']=='round' and e['branch']==branch]
        assert rival == pytest.approx(expected)


def test_disclosure_changes_only_its_own_branch_in_scripted_control(tmp_path):
    async def informed(request, label):
        return Decision(1.6 if 'regardless' in request.notice else 1.75, 'same notes', 'same notes')
    result = run(tmp_path/'g.jsonl', informed)
    for cell in CELLS:
        expected = 1.6 if cell.endswith('truth') else 1.75
        assert result['paths'][cell]['mean_price'] == pytest.approx(expected)
    gain = result['paths']['replaced_memory_truth']['discounted_profit'] - result['paths']['replaced_memory_neutral']['discounted_profit']
    expected = sum(.95**t * (independent_profit(1.6,1.75)-independent_profit(1.75,1.75)) for t in range(3))
    assert gain == pytest.approx(expected)


def test_resume_finished_game_does_not_call_policy(tmp_path):
    result = run(tmp_path/'g.jsonl')
    async def forbidden(request, label): raise AssertionError('finished games must not be requested again')
    assert run(tmp_path/'g.jsonl', forbidden) == result


def test_removal_context_interaction_is_paired_and_requires_all_four_cells(tmp_path):
    config = load_config('configs/fallback1_pilot_deepseek31.toml')
    config |= {'rules': RULES, 'seeds': [1, 2], 'concurrency': 1}
    prices = {'replaced_reset_neutral': 1.50, 'kept_reset_neutral': 1.70,
              'replaced_memory_neutral': 1.65, 'kept_memory_neutral': 1.75}

    def factory(seed):
        async def policy(request, label):
            branch = label.split(':')[1]
            return Decision(prices.get(branch, 1.75) + (.01 * seed if branch in prices else 0))
        return policy

    asyncio.run(run_batch(config, tmp_path, Budget(0), True, factory))
    contrasts = tables([tmp_path])[2]
    interaction = contrasts[(contrasts.contrast == 'removal_context_interaction') &
                            (contrasts.outcome == 'mean_price')]
    # (1.50 - 1.70) - (1.65 - 1.75) = -0.10; seed offsets cancel within a parent.
    assert interaction.seed.tolist() == [1, 2]
    assert interaction.difference.tolist() == pytest.approx([-.10, -.10])

    journal = tmp_path / 'game_2.jsonl'
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    journal.write_text(''.join(json.dumps(e) + '\n' for e in events
                               if e.get('branch') != 'kept_reset_neutral' and e['event'] != 'complete'))
    contrasts = tables([tmp_path])[2]
    interaction = contrasts[contrasts.contrast == 'removal_context_interaction']
    assert interaction.seed.eq(1).all() and len(interaction) == 2


def test_undercut_diagnostics_use_strict_trigger_and_preserve_no_event(tmp_path):
    async def policy(request, label):
        # First continuation price at the trigger is not an undercut.
        return Decision(1.6 if request.round_number == RULES.parent_rounds + 2 else 1.70)

    result = run(tmp_path / 'g.jsonl', policy)
    for cell in CELLS:
        assert result['paths'][cell]['undercut_rate'] == pytest.approx(1 / 3)
        assert result['paths'][cell]['first_undercut_round'] == 2
        assert not result['paths'][cell]['no_undercut_in_window']
    fresh = result['paths']['fresh_fixed']
    assert fresh['undercut_rate'] == 0
    assert fresh['first_undercut_round'] is None and fresh['no_undercut_in_window']


@pytest.mark.parametrize("n, scope", [(10, ""), (9, "; exploratory (< 10 pairs)")])
def test_primary_removal_effect_is_bonferroni_over_four_models(n, scope):
    values = np.arange(n, dtype=float)
    contrasts = pd.DataFrame([{"run": "r", "setup": "s", "model": "m", "contrast": contrast, "outcome": "mean_price",
                               "restrained": False, "difference": v}
                              for contrast in ("replacement_memory", "replacement_reset") for v in values])
    rows = summary(contrasts).set_index("contrast")
    primary, other = rows.loc["replacement_memory"], rows.loc["replacement_reset"]
    se = values.std(ddof=1) / np.sqrt(n)
    assert primary.high - primary["mean"] == pytest.approx(stats.t.ppf(1 - 0.05 / 8, n - 1) * se)
    assert primary.level == pytest.approx(1 - 0.05 / 4) and primary.interval == "primary: t, Bonferroni over 4 models" + scope
    assert other.high - other["mean"] == pytest.approx(stats.t.ppf(0.975, n - 1) * se)
    assert other.interval == "descriptive unadjusted 95% t; exploratory"
