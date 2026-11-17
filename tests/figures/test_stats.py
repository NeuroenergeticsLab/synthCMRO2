"""Unit tests for the statistics, using simulated data with known answers."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from synthcmro2.figures.stats import (
    agreement,
    fdr_bh,
    format_p,
    games_howell,
    paired_difference_summary,
    pairwise_wilcoxon,
    significance_stars,
)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def test_agreement_recovers_known_slope_and_mae(rng):
    x = rng.uniform(50, 300, 2000)
    y = 10 + 0.8 * x + rng.normal(0, 5, x.size)
    res = agreement(x, y, n_bootstrap=500, rng=1)
    assert res.slope == pytest.approx(0.8, abs=0.01)
    assert res.intercept == pytest.approx(10, abs=1.5)
    assert res.mae == pytest.approx(np.mean(np.abs(x - y)))
    assert res.r_ci_low < res.r < res.r_ci_high


@pytest.mark.filterwarnings("ignore::RuntimeWarning", "ignore::scipy.stats.DegenerateDataWarning")
def test_agreement_perfect_identity():
    x = np.linspace(1, 100, 200)
    res = agreement(x, x.copy(), n_bootstrap=200, rng=1)
    assert res.r == pytest.approx(1.0)
    assert res.slope == pytest.approx(1.0)
    assert res.mae == pytest.approx(0.0)


def test_agreement_ignores_non_finite_values(rng):
    x = rng.normal(100, 20, 300)
    y = x + rng.normal(0, 5, 300)
    x_nan, y_nan = x.copy(), y.copy()
    x_nan[:5] = np.nan
    res = agreement(x_nan, y_nan, n_bootstrap=200, rng=1)
    assert res.n == 295
    assert res.r == pytest.approx(stats.pearsonr(x[5:], y[5:])[0])


def test_agreement_bootstrap_is_reproducible(rng):
    x = rng.normal(100, 20, 500)
    y = x + rng.normal(0, 10, 500)
    a = agreement(x, y, n_bootstrap=300, rng=7)
    b = agreement(x, y, n_bootstrap=300, rng=7)
    assert (a.r_ci_low, a.r_ci_high) == (b.r_ci_low, b.r_ci_high)


def test_agreement_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        agreement(np.ones(5), np.ones(6))


def test_fdr_bh_matches_known_values():
    # Values from the original notebook: pg.multicomp([0.02, 0.002, 0.001], method="fdr_bh")
    np.testing.assert_allclose(fdr_bh(np.array([0.02, 0.002, 0.001])), [0.02, 0.003, 0.003])


def test_fdr_bh_keeps_nan():
    out = fdr_bh(np.array([0.01, np.nan, 0.04]))
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], [0.02, 0.04])


def test_wilcoxon_all_negative_differences_gives_exact_minimum_p():
    a = np.arange(10, dtype=float)
    res = paired_difference_summary(a, a + np.arange(1, 11))
    assert res["W"] == 0
    assert res["p"] == pytest.approx(2 / 2**10)  # exact two-sided p for n = 10
    assert res["mean_diff"] == pytest.approx(-5.5)


def test_pairwise_wilcoxon_covers_all_pairs(rng):
    wide = pd.DataFrame(rng.normal(size=(12, 4)), columns=list("ABCD"))
    out = pairwise_wilcoxon(wide)
    assert len(out) == 6
    assert (out["p_fdr"] >= out["p"]).all()


def test_games_howell_two_groups_equals_welch_t_test(rng):
    a, b = rng.normal(0, 1, 30), rng.normal(0.5, 2, 15)
    values = np.concatenate([a, b])
    groups = ["a"] * a.size + ["b"] * b.size
    gh = games_howell(pd.Series(values), pd.Series(groups)).iloc[0]
    welch = stats.ttest_ind(a, b, equal_var=False)
    assert gh["T"] == pytest.approx(welch.statistic)
    assert gh["p"] == pytest.approx(welch.pvalue, rel=1e-4)


def test_games_howell_three_groups_returns_three_pairs(rng):
    values = rng.normal(size=60)
    groups = np.repeat(["x", "y", "z"], 20)
    assert len(games_howell(pd.Series(values), pd.Series(groups))) == 3


@pytest.mark.parametrize(("p", "stars"), [(0.2, ""), (0.04, "*"), (0.009, "**"), (0.0005, "***")])
def test_significance_stars(p, stars):
    assert significance_stars(p) == stars


def test_format_p():
    assert format_p(1e-6, italic=False) == "P < .001"
    assert format_p(0.0042, italic=False) == "P = .004"
    assert format_p(0.0172, italic=False) == "P = .02"
    assert format_p(0.999, italic=False) == "P > .99"
    assert format_p(0.5) == r"$\it{P}$ = .50"


def test_format_num_minus():
    from synthcmro2.figures.stats import format_num

    assert format_num(-0.42) == "\u22120.42"
    assert format_num(26.75, 1) == "26.8"
