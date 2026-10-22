"""
Tests for statistical algorithms, permutation tests, Welch t-test, and multiple-comparison control.
"""

import math
import pytest
from stackgate.stats import (
    benjamini_hochberg,
    benjamini_yekutieli,
    betainc,
    bonferroni,
    bootstrap_confidence_interval,
    cv,
    mann_whitney_u_test,
    mean,
    permutation_test,
    std_dev,
    student_t_cdf,
    variance,
    welch_ttest,
)

try:
    import scipy.special
    import scipy.stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def test_basic_stats():
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert mean(data) == 5.0
    assert abs(variance(data) - 4.571428) < 1e-4
    assert abs(std_dev(data) - 2.138089) < 1e-4
    assert abs(cv(data) - (2.138089 / 5.0 * 100)) < 1e-3

    # Empty / single element
    assert mean([]) == 0.0
    assert variance([10.0]) == 0.0
    assert cv([0.0]) == 0.0


def test_betainc_matches_scipy():
    # Test boundary and typical values
    assert betainc(1.0, 1.0, 0.5) == 0.5
    assert betainc(2.0, 3.0, 0.0) == 0.0
    assert betainc(2.0, 3.0, 1.0) == 1.0

    if HAS_SCIPY:
        for a, b, x in [
            (0.5, 0.5, 0.2),
            (2.0, 5.0, 0.3),
            (10.0, 10.0, 0.5),
            (1.5, 2.5, 0.7),
            (4.0, 1.0, 0.9),
        ]:
            expected = scipy.special.betainc(a, b, x)
            actual = betainc(a, b, x)
            assert abs(actual - expected) < 1e-6, f"Mismatch at a={a}, b={b}, x={x}"


def test_student_t_cdf():
    # Symmetry at 0
    assert student_t_cdf(0.0, 5.0) == 0.5

    # Known values & comparison with scipy
    if HAS_SCIPY:
        for t_val in [-3.0, -1.96, -0.5, 0.0, 0.5, 1.96, 3.0]:
            for df in [2.0, 5.0, 10.0, 30.0]:
                expected = scipy.stats.t.cdf(t_val, df=df)
                actual = student_t_cdf(t_val, df)
                assert abs(actual - expected) < 1e-5, f"Mismatch at t={t_val}, df={df}"


def test_welch_ttest():
    x = [10.0, 10.2, 9.8, 10.1, 9.9]
    y = [12.0, 12.1, 11.9, 12.2, 11.8]

    # Candidate y is strictly slower (greater)
    p_reg, t_stat, df = welch_ttest(x, y, alternative="greater")
    assert p_reg < 0.001
    assert t_stat > 0

    if HAS_SCIPY:
        scipy_res = scipy.stats.ttest_ind(y, x, equal_var=False, alternative="greater")
        assert abs(p_reg - scipy_res.pvalue) < 1e-4

    # Identical groups
    p_ident, _, _ = welch_ttest(x, x, alternative="greater")
    assert abs(p_ident - 0.5) < 1e-4


def test_permutation_test_exact():
    # Small sample sizes (K=4, M=4 -> comb(8,4)=70)
    x = [1.0, 2.0, 3.0, 4.0]
    y = [5.0, 6.0, 7.0, 8.0]

    # y is completely larger than x
    p_reg, delta = permutation_test(x, y, alternative="greater")
    assert delta == 4.0
    # There is only 1 combination out of 70 where sum(y) >= sum(y_obs)
    assert abs(p_reg - (1.0 / 70.0)) < 1e-6

    # Two-sided
    p_two, _ = permutation_test(x, y, alternative="two-sided")
    assert abs(p_two - (2.0 / 70.0)) < 1e-6


def test_permutation_test_monte_carlo():
    x = [10.0 + (i * 0.1) for i in range(10)]
    y = [12.0 + (i * 0.1) for i in range(10)]

    p_val, delta = permutation_test(x, y, alternative="greater", num_permutations=2000, seed=42)
    assert delta > 1.9
    assert p_val < 0.01

    # Determinism
    p_val2, _ = permutation_test(x, y, alternative="greater", num_permutations=2000, seed=42)
    assert p_val == p_val2


def test_mann_whitney_u():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [6.0, 7.0, 8.0, 9.0, 10.0]

    p_val, u_stat = mann_whitney_u_test(x, y, alternative="greater")
    assert p_val < 0.05
    assert u_stat == 25.0  # All y > x (5 * 5 = 25)

    if HAS_SCIPY:
        scipy_res = scipy.stats.mannwhitneyu(y, x, alternative="greater")
        assert abs(p_val - scipy_res.pvalue) < 0.02


def test_bootstrap_confidence_interval():
    x = [10.0, 10.2, 9.8, 10.1, 9.9]
    y = [11.0, 11.2, 10.8, 11.1, 10.9]

    d_low, d_high, pct_low, pct_high = bootstrap_confidence_interval(
        x, y, ci_level=0.95, num_bootstrap=2000, seed=42
    )

    assert d_low < d_high
    assert pct_low < pct_high
    # Mean delta is 1.0, mean pct change is 10%
    assert 0.7 < d_low < 1.0
    assert 1.0 < d_high < 1.3
    assert 7.0 < pct_low < 10.0
    assert 10.0 < pct_high < 13.0


def test_benjamini_hochberg_hand_calculated():
    # m = 4, p-values: [0.01, 0.04, 0.03, 0.20]
    # Sorted: (0.01, rank 1), (0.03, rank 2), (0.04, rank 3), (0.20, rank 4)
    # Raw adjustments:
    # rank 4: 0.20 * 4/4 = 0.20
    # rank 3: 0.04 * 4/3 = 0.05333...
    # rank 2: 0.03 * 4/2 = 0.06
    # rank 1: 0.01 * 4/1 = 0.04
    # Step-up min:
    # q[3] = 0.20
    # q[2] = 0.05333...
    # q[1] = min(0.05333..., 0.06) = 0.05333...
    # q[0] = min(0.05333..., 0.04) = 0.04
    p_vals = [0.01, 0.04, 0.03, 0.20]
    q_vals = benjamini_hochberg(p_vals)

    assert abs(q_vals[0] - 0.04) < 1e-4
    assert abs(q_vals[1] - 0.053333) < 1e-4
    assert abs(q_vals[2] - 0.053333) < 1e-4
    assert abs(q_vals[3] - 0.20) < 1e-4


def test_benjamini_hochberg_edge_cases():
    assert benjamini_hochberg([]) == []
    assert benjamini_hochberg([0.05]) == [0.05]

    # All identical
    q_same = benjamini_hochberg([0.02, 0.02, 0.02])
    assert all(abs(q - 0.02) < 1e-6 for q in q_same)


def test_benjamini_yekutieli_and_bonferroni():
    p_vals = [0.01, 0.02, 0.05]
    bonf = bonferroni(p_vals)
    assert bonf == pytest.approx([0.03, 0.06, 0.15])

    by = benjamini_yekutieli(p_vals)
    # BY must be >= BH
    bh = benjamini_hochberg(p_vals)
    for q_by, q_bh in zip(by, bh):
        assert q_by >= q_bh
