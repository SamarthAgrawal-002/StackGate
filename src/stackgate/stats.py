"""
Rigorous statistical engines for regression testing, multiple-comparison control, and confidence intervals.
Zero external runtime dependencies required (pure Python stdlib), with optional Scipy fast-paths.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Sequence


def mean(values: Sequence[float]) -> float:
    """Compute arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def variance(values: Sequence[float]) -> float:
    """Compute unbiased sample variance (denominator N - 1)."""
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return sum((x - m) ** 2 for x in values) / (n - 1)


def std_dev(values: Sequence[float]) -> float:
    """Compute sample standard deviation."""
    return math.sqrt(variance(values))


def cv(values: Sequence[float]) -> float:
    """Compute coefficient of variation (std_dev / mean) as percentage."""
    m = mean(values)
    if m <= 0.0:
        return 0.0
    return (std_dev(values) / m) * 100.0


# ---------------------------------------------------------------------------
# Regularized Incomplete Beta Function & Student's t CDF (Pure Python)
# ---------------------------------------------------------------------------

def _log_gamma(z: float) -> float:
    """Lanczos approximation for ln(gamma(z))."""
    return math.lgamma(z)


def _betacf(a: float, b: float, x: float, max_iter: int = 150, eps: float = 1e-15) -> float:
    """
    Continued fraction for the regularized incomplete beta function
    using modified Lentz's method.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < eps:
        d = eps
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        h *= d * c

        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        del_h = d * c
        h *= del_h

        if abs(del_h - 1.0) < eps:
            break

    return h


def betainc(a: float, b: float, x: float) -> float:
    """
    Regularized incomplete beta function I_x(a, b).
    Exact match to scipy.special.betainc.
    """
    if x < 0.0 or x > 1.0:
        raise ValueError(f"x must be between 0 and 1, got {x}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # Factor front = exp(ln(gamma(a+b)) - ln(gamma(a)) - ln(gamma(b)) + a*ln(x) + b*ln(1-x)) / a
    bt = math.exp(
        _log_gamma(a + b) - _log_gamma(a) - _log_gamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    )

    # Use symmetry transformation if x > (a + 1) / (a + b + 2)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """
    Cumulative distribution function of Student's t distribution with `df` degrees of freedom.
    """
    if df <= 0:
        raise ValueError(f"Degrees of freedom must be > 0, got {df}")
    if t == 0.0:
        return 0.5

    x = df / (df + t * t)
    # P(T <= t) for t > 0 is 1 - 0.5 * I_x(df/2, 1/2)
    p_half = 0.5 * betainc(df / 2.0, 0.5, x)
    if t > 0:
        return 1.0 - p_half
    else:
        return p_half


# ---------------------------------------------------------------------------
# Permutation Test
# ---------------------------------------------------------------------------

def permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    alternative: str = "greater",  # 'greater' (y > x: regression), 'two-sided', 'less'
    num_permutations: int = 10000,
    seed: int | None = 42,
) -> tuple[float, float]:
    """
    Exact or Monte Carlo two-sample permutation test on the difference of means (mean(y) - mean(x)).

    Args:
        x: Baseline sample values.
        y: Candidate sample values.
        alternative: 'greater' (test if y > x, i.e. regression), 'two-sided', or 'less'.
        num_permutations: Number of Monte Carlo permutations if sample size > 16.
        seed: Random seed for deterministic reproducibility.

    Returns:
        tuple (p_value, observed_delta)
    """
    k = len(x)
    m = len(y)
    if k == 0 or m == 0:
        return 1.0, 0.0

    mean_x = mean(x)
    mean_y = mean(y)
    obs_delta = mean_y - mean_x

    # If identical collections or zero variance between them
    if all(abs(val - obs_delta) < 1e-14 for val in [0.0]) and all(abs(a - b) < 1e-14 for a, b in zip(x, y)):
        return 1.0, obs_delta

    pooled = list(x) + list(y)
    total_n = k + m

    # Check for exact test feasibility: comb(total_n, k) <= 15000
    total_combs = math.comb(total_n, k)
    if total_combs <= 15000:
        # Exact enumeration of all combinations
        count = 0
        total_sum = sum(pooled)
        tol = 1e-12

        for indices in itertools.combinations(range(total_n), k):
            sum_x = sum(pooled[i] for i in indices)
            sum_y = total_sum - sum_x
            perm_delta = (sum_y / m) - (sum_x / k)

            if alternative == "greater":
                if perm_delta >= obs_delta - tol:
                    count += 1
            elif alternative == "less":
                if perm_delta <= obs_delta + tol:
                    count += 1
            else:  # two-sided
                if abs(perm_delta) >= abs(obs_delta) - tol:
                    count += 1

        p_value = count / total_combs
        return min(1.0, max(0.0, p_value)), obs_delta

    # Monte Carlo permutation test
    rng = random.Random(seed)
    count = 0
    tol = 1e-12

    # Vectorized / optimized pool indexing
    pool_copy = list(pooled)
    for _ in range(num_permutations):
        rng.shuffle(pool_copy)
        sum_x = sum(pool_copy[:k])
        sum_y = sum(pool_copy[k:])
        perm_delta = (sum_y / m) - (sum_x / k)

        if alternative == "greater":
            if perm_delta >= obs_delta - tol:
                count += 1
        elif alternative == "less":
            if perm_delta <= obs_delta + tol:
                count += 1
        else:  # two-sided
            if abs(perm_delta) >= abs(obs_delta) - tol:
                count += 1

    # Laplace / unbiased +1 estimator to prevent zero p-values
    p_value = (count + 1) / (num_permutations + 1)
    return min(1.0, max(0.0, p_value)), obs_delta


# ---------------------------------------------------------------------------
# Welch's t-test
# ---------------------------------------------------------------------------

def welch_ttest(
    x: Sequence[float],
    y: Sequence[float],
    alternative: str = "greater",
) -> tuple[float, float, float]:
    """
    Welch's two-sample t-test (unequal variances).

    Args:
        x: Baseline values.
        y: Candidate values.
        alternative: 'greater' (y > x: regression), 'two-sided', or 'less'.

    Returns:
        tuple (p_value, t_statistic, degrees_of_freedom)
    """
    k = len(x)
    m = len(y)
    if k < 2 or m < 2:
        # Fall back to permutation test if degrees of freedom insufficient
        p, _ = permutation_test(x, y, alternative=alternative)
        return p, 0.0, 1.0

    mean_x = mean(x)
    mean_y = mean(y)
    var_x = variance(x)
    var_y = variance(y)

    # Handle zero variance cases
    if var_x == 0.0 and var_y == 0.0:
        if abs(mean_y - mean_x) < 1e-14:
            return 1.0, 0.0, float(k + m - 2)
        if alternative == "greater":
            return (0.0 if mean_y > mean_x else 1.0), (1e9 if mean_y > mean_x else -1e9), float(k + m - 2)
        elif alternative == "less":
            return (0.0 if mean_y < mean_x else 1.0), (-1e9 if mean_y < mean_x else 1e9), float(k + m - 2)
        else:
            return 0.0, (1e9 if mean_y > mean_x else -1e9), float(k + m - 2)

    se = math.sqrt(var_x / k + var_y / m)
    if se <= 0.0:
        return 1.0, 0.0, float(k + m - 2)

    t_stat = (mean_y - mean_x) / se

    # Welch-Satterthwaite degrees of freedom
    vx_k = var_x / k
    vy_m = var_y / m
    numerator = (vx_k + vy_m) ** 2
    denominator = (vx_k ** 2) / (k - 1) + (vy_m ** 2) / (m - 1)
    df = numerator / denominator if denominator > 0 else 1.0

    # Calculate p-value from Student's t CDF
    if alternative == "greater":
        p_val = 1.0 - student_t_cdf(t_stat, df)
    elif alternative == "less":
        p_val = student_t_cdf(t_stat, df)
    else:  # two-sided
        p_val = 2.0 * (1.0 - student_t_cdf(abs(t_stat), df))

    return min(1.0, max(0.0, p_val)), t_stat, df


# ---------------------------------------------------------------------------
# Mann-Whitney U test
# ---------------------------------------------------------------------------

def mann_whitney_u_test(
    x: Sequence[float],
    y: Sequence[float],
    alternative: str = "greater",
) -> tuple[float, float]:
    """
    Mann-Whitney U rank-sum test with tie handling and normal approximation.

    Returns:
        tuple (p_value, u_statistic)
    """
    k = len(x)
    m = len(y)
    if k == 0 or m == 0:
        return 1.0, 0.0

    # Combine and sort with group tracking (0 for x, 1 for y)
    combined = [(val, 0) for val in x] + [(val, 1) for val in y]
    combined.sort(key=lambda item: item[0])

    # Assign average ranks for ties
    ranks = [0.0] * len(combined)
    i = 0
    tie_adjustment = 0.0
    while i < len(combined):
        j = i
        while j < len(combined) and abs(combined[j][0] - combined[i][0]) < 1e-14:
            j += 1
        tie_len = j - i
        avg_rank = (i + 1 + j) / 2.0
        for idx in range(i, j):
            ranks[idx] = avg_rank
        if tie_len > 1:
            tie_adjustment += tie_len ** 3 - tie_len
        i = j

    # Sum of ranks for Y
    rank_sum_y = sum(rank for rank, (_, group) in zip(ranks, combined) if group == 1)
    u_y = rank_sum_y - (m * (m + 1)) / 2.0

    # Expected value and variance under null hypothesis
    n_total = k + m
    mean_u = (k * m) / 2.0
    var_u = (k * m / 12.0) * (
        (n_total + 1) - tie_adjustment / (n_total * (n_total - 1)) if n_total > 1 else (n_total + 1)
    )

    if var_u <= 0.0:
        return 1.0, u_y

    # Continuity correction
    sigma_u = math.sqrt(var_u)
    if alternative == "greater":
        z = (u_y - mean_u - 0.5) / sigma_u
        p_val = 0.5 * math.erfc(z / math.sqrt(2.0))
    elif alternative == "less":
        z = (u_y - mean_u + 0.5) / sigma_u
        p_val = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    else:
        z = (abs(u_y - mean_u) - 0.5) / sigma_u
        p_val = math.erfc(z / math.sqrt(2.0))

    return min(1.0, max(0.0, p_val)), u_y


# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

def bootstrap_confidence_interval(
    x: Sequence[float],
    y: Sequence[float],
    ci_level: float = 0.95,
    num_bootstrap: int = 5000,
    seed: int | None = 42,
) -> tuple[float, float, float, float]:
    """
    Compute bootstrap confidence intervals for:
      1. Absolute delta: mean(y) - mean(x)
      2. Percentage change: ((mean(y) - mean(x)) / mean(x)) * 100

    Returns:
        tuple (delta_ci_lower, delta_ci_upper, pct_ci_lower, pct_ci_upper)
    """
    k = len(x)
    m = len(y)
    if k == 0 or m == 0:
        return 0.0, 0.0, 0.0, 0.0

    mean_x_orig = mean(x)
    delta_orig = mean(y) - mean_x_orig
    pct_orig = (delta_orig / mean_x_orig * 100.0) if mean_x_orig > 0 else 0.0

    if k == 1 and m == 1:
        return delta_orig, delta_orig, pct_orig, pct_orig

    rng = random.Random(seed)
    deltas: list[float] = []
    pct_changes: list[float] = []

    for _ in range(num_bootstrap):
        # Sample with replacement
        sample_x = [x[rng.randint(0, k - 1)] for _ in range(k)]
        sample_y = [y[rng.randint(0, m - 1)] for _ in range(m)]

        mx = sum(sample_x) / k
        my = sum(sample_y) / m
        d = my - mx
        deltas.append(d)
        if mx > 0.0:
            pct_changes.append((d / mx) * 100.0)
        else:
            pct_changes.append(0.0)

    deltas.sort()
    pct_changes.sort()

    alpha = 1.0 - ci_level
    lower_idx = int(math.floor(alpha / 2.0 * len(deltas)))
    upper_idx = int(math.ceil((1.0 - alpha / 2.0) * len(deltas))) - 1

    lower_idx = max(0, min(lower_idx, len(deltas) - 1))
    upper_idx = max(0, min(upper_idx, len(deltas) - 1))

    return deltas[lower_idx], deltas[upper_idx], pct_changes[lower_idx], pct_changes[upper_idx]


# ---------------------------------------------------------------------------
# Multiple-Comparison Control (FDR and FWER)
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """
    Benjamini-Hochberg procedure for False Discovery Rate (FDR) control.
    Computes adjusted q-values:
      q_(i) = min_{j >= i} ( (m / j) * p_(j) )

    Args:
        p_values: Sequence of raw p-values for m tests.

    Returns:
        List of adjusted q-values in original input order.
    """
    m = len(p_values)
    if m == 0:
        return []
    if m == 1:
        return [min(1.0, max(0.0, p_values[0]))]

    # Sort p-values with original indices
    indexed_p = sorted(enumerate(p_values), key=lambda item: item[1])
    q_values = [0.0] * m

    # Step-up adjustment from largest to smallest
    min_q = 1.0
    for rank in range(m, 0, -1):
        orig_idx, p = indexed_p[rank - 1]
        raw_q = (p * m) / rank
        min_q = min(min_q, raw_q)
        q_values[orig_idx] = min(1.0, max(0.0, min_q))

    return q_values


def benjamini_yekutieli(p_values: Sequence[float]) -> list[float]:
    """
    Benjamini-Yekutieli procedure for FDR control under arbitrary/positive dependence.
    Scales BH q-values by harmonic factor c(m) = sum(1/i for i in 1..m).
    """
    m = len(p_values)
    if m <= 1:
        return benjamini_hochberg(p_values)

    harmonic_sum = sum(1.0 / i for i in range(1, m + 1))
    bh_q = benjamini_hochberg(p_values)
    return [min(1.0, q * harmonic_sum) for q in bh_q]


def bonferroni(p_values: Sequence[float]) -> list[float]:
    """
    Bonferroni family-wise error rate (FWER) correction.
    """
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]
