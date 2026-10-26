"""Inference contracts: sampling unit, conditional denominators and paired baselines."""
import numpy as np
import pandas as pd
import pytest

from sprint import study
from sprint.analyze import bootstrap_ci


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
