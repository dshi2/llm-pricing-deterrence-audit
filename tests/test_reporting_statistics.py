"""Inference contracts: sampling unit, conditional denominators and paired baselines."""
import numpy as np
import pandas as pd
import pytest

from sprint import covert_analyze, study
from sprint.analyze import bootstrap_ci


def frame(records):
    base = {'run': 'snapshot', 'setup': 'fixed', 'model': 'test', 'scheme': 'affine',
                'view': 'decoder', 'question': 'dependent', 'answered': True}
    return pd.DataFrame([base | r for r in records])


def test_f2_ambiguous_only_and_equal_block_weight(monkeypatch):
    records = [{'block': 1, 'expected': 'insufficient_evidence', 'predicted': 'supported', 'correct': False}]
    records += [{'block': 2, 'expected': 'insufficient_evidence', 'predicted': 'insufficient_evidence', 'correct': True}] * 3
    records += [{'block': 2, 'expected': 'contradicted', 'predicted': 'supported', 'correct': False},
                {'block': 2, 'expected': 'insufficient_evidence', 'predicted': None, 'correct': False, 'answered': False}]
    captured = []
    def ci(values):
        captured.append(list(values))
        return (0., 1.)
    monkeypatch.setattr(covert_analyze, 'bootstrap_ci', ci)
    result = covert_analyze.summary(frame(records)).set_index('metric')
    r = result.loc['unsupported_positive']
    assert r.numerator == 1 and r.denominator == 4
    assert r.blocks == 2 and r.mean_block_rate == pytest.approx(.5)  # not pooled 1/4
    assert captured[2] == [1., 0.]  # one resampling value per block
    assert result.loc['answer_coverage', 'denominator'] == 6
    assert result.loc['answer_coverage', 'numerator'] == 5
    assert result.loc['resolved_accuracy', 'denominator'] == 1
    assert result.loc['false_positive', 'numerator'] == 1


def test_f2_paired_baseline_uses_same_cases_and_blocks(monkeypatch):
    rows = []
    for view in ('public', 'decoder', 'complete'):
        for block, values in [(1, [(True, False)]), (2, [(False, True)] * 3)]:
            for kind, (correct, base_correct) in enumerate(values):
                rows.append(dict(view=view, block=block, kind=str(kind), correct=correct,
                                 answered=correct, expected='supported', predicted='supported' if correct else None,
                                 **{n+'_correct': base_correct for n in
                                    ('evidence_level_only', 'parallel_rise', 'always_abstain', 'always_accuse')}))
    captured = []
    def ci(values):
        captured.append(list(values))
        return (-1., 1.)
    monkeypatch.setattr(covert_analyze, 'bootstrap_ci', ci)
    result = covert_analyze.paired_comparisons(frame(rows)).set_index('contrast')
    r = result.loc['decoder_minus_evidence_level_only']
    assert r.blocks == 2
    assert r.mean_accuracy_difference == pytest.approx(0.)  # mean(+1,-1), not case-weighted -.5
    assert [1., -1.] in captured  # unanswered judgments remain incorrect, not silently dropped


def test_bootstrap_reproducible_percentiles_of_block_means():
    block_rates = np.array([0., .25, 1.])
    seed, draws = 37, 4000
    expected = np.percentile(np.random.default_rng(seed).choice(
        block_rates, size=(draws, 3)).mean(axis=1), [2.5, 97.5])
    assert bootstrap_ci(block_rates, n_boot=draws, seed=seed) == pytest.approx(expected)


@pytest.mark.parametrize('n', [1, 4])
def test_main_one_game_or_zero_variance_does_not_claim_certainty(n):
    table = pd.DataFrame([{'run': 'r', 'setup': 's', 'model': 'm', 'status': 'eligible',
                               'G10': 4., 'G01': 2., 'G11': 1.} for _ in range(n)])
    result = study.primary(table, 'pooled').set_index('outcome')
    assert (result.eligible == n).all()
    assert result.loc['S_minus_J', 'mean'] == 2.
    assert result.lo.isna().all() and result.hi.isna().all()
    assert (result.interpretation == 'uncertainty unavailable').all()
    assert (result.level == 1-.05/16).all()
