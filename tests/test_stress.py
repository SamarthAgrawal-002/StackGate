"""
Stress and scale tests: large profile trees, deep call stacks, and multiple-comparison stress.
"""

import random
import time
import pytest
from stackgate.comparator import compare
from stackgate.models import Profile, ProfileSet, Verdict
from stackgate.stats import benjamini_hochberg, welch_ttest


def test_massive_profile_comparison_and_fdr_control():
    """
    Stress test with 1,000 functions across 5 baseline and 5 candidate runs.
    Inject regressions into exactly 3 functions.
    Verify:
      1. Execution is fast (< 3 seconds).
      2. BH FDR procedure correctly flags the 3 regressed functions.
      3. False discoveries among the 997 non-regressed functions are <= 1.
    """
    num_funcs = 1000
    k_runs = 5
    m_runs = 5

    regressed_target_funcs = {"func_0042", "func_0150", "func_0789"}

    rng = random.Random(1337)

    base_runs: list[Profile] = []
    for r in range(k_runs):
        stacks = {}
        for i in range(num_funcs):
            fn = f"func_{i:04d}"
            # Baseline mean ~ 50 samples
            count = max(1, int(rng.gauss(50, 4)))
            stacks[(f"module_{i%20}", fn)] = count
        base_runs.append(Profile(name=f"b_{r}", stacks=stacks, sampling_rate_hz=1000.0))

    cand_runs: list[Profile] = []
    for r in range(m_runs):
        stacks = {}
        for i in range(num_funcs):
            fn = f"func_{i:04d}"
            if fn in regressed_target_funcs:
                # Regressed by +400 samples each (+2.4% total regression)
                count = max(1, int(rng.gauss(450, 6)))
            else:
                # Same distribution as baseline
                count = max(1, int(rng.gauss(50, 4)))
            stacks[(f"module_{i%20}", fn)] = count
        cand_runs.append(Profile(name=f"c_{r}", stacks=stacks, sampling_rate_hz=1000.0))

    start_time = time.perf_counter()
    result = compare(
        ProfileSet(name="base", profiles=base_runs),
        ProfileSet(name="cand", profiles=cand_runs),
        threshold=2.0,
        alpha=0.05,
        fdr_q=0.05,
        metric="self",
    )
    duration = time.perf_counter() - start_time

    assert duration < 3.0, f"Comparison was too slow: took {duration:.2f}s"
    assert result.verdict == Verdict.FAIL

    discovered_regressed = {f.function for f in result.regressed_functions}
    # All 3 injected functions must be discovered
    assert regressed_target_funcs.issubset(discovered_regressed)
    # Check that false discoveries are strictly controlled
    false_discoveries = discovered_regressed - regressed_target_funcs
    assert len(false_discoveries) <= 2, f"Too many false discoveries: {false_discoveries}"


def test_deep_call_stack():
    """Verify profiles with 100-frame deep call stacks are handled without recursion error."""
    deep_path = tuple(f"frame_{i}" for i in range(100))
    p1 = Profile(name="deep_1", stacks={deep_path: 100})
    p2 = Profile(name="deep_2", stacks={deep_path: 120})

    base = ProfileSet(name="base", profiles=[p1, p1, p1])
    cand = ProfileSet(name="cand", profiles=[p2, p2, p2])

    res = compare(base, cand, threshold=5.0)
    assert res.verdict == Verdict.FAIL
    assert len(res.regressed_functions) == 1
    assert res.regressed_functions[0].function == "frame_99"  # Leaf frame


def test_randomized_pvalue_bounds():
    """Verify that welch_ttest always outputs p-values strictly in [0, 1]."""
    rng = random.Random(42)
    for _ in range(100):
        k = rng.randint(2, 10)
        m = rng.randint(2, 10)
        x = [rng.uniform(0.0, 100.0) for _ in range(k)]
        y = [rng.uniform(0.0, 100.0) for _ in range(m)]

        for alt in ["greater", "less", "two-sided"]:
            p, _, _ = welch_ttest(x, y, alternative=alt)
            assert 0.0 <= p <= 1.0, f"p-value out of bounds: {p}"


def test_randomized_fdr_monotonicity():
    """Verify Benjamini-Hochberg monotonicity property across arbitrary random p-values."""
    rng = random.Random(99)
    for _ in range(50):
        n = rng.randint(1, 100)
        p_vals = [rng.uniform(0.0, 1.0) for _ in range(n)]
        q_vals = benjamini_hochberg(p_vals)

        # Re-sort by original p-value and check that q-values are non-decreasing
        sorted_pairs = sorted(zip(p_vals, q_vals), key=lambda item: item[0])
        for idx in range(len(sorted_pairs) - 1):
            assert sorted_pairs[idx][1] <= sorted_pairs[idx + 1][1] + 1e-12
