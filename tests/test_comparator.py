"""
Tests for profile comparison engine, regression detection, function attribution, and verdicts.
"""

import pytest
from stackgate.comparator import ComparisonConfig, compare
from stackgate.exceptions import RegressionDetectedError
from stackgate.models import Profile, ProfileSet, Verdict


def _make_mock_set(name: str, runs: list[dict[tuple[str, ...], int]]) -> ProfileSet:
    profiles = [
        Profile(name=f"{name}_{i+1}", stacks=run, sampling_rate_hz=1000.0)
        for i, run in enumerate(runs)
    ]
    return ProfileSet(name=name, profiles=profiles)


def test_compare_clean_pass():
    # 5 runs each with identical behavior and slight noise
    base_runs = [
        {("main", "compute"): 100 + i, ("main", "io"): 50}
        for i in range(-2, 3)
    ]
    cand_runs = [
        {("main", "compute"): 100 + i, ("main", "io"): 50}
        for i in range(-2, 3)
    ]
    base = _make_mock_set("base", base_runs)
    cand = _make_mock_set("cand", cand_runs)

    res = compare(base, cand, threshold=2.0)
    assert res.verdict == Verdict.PASS
    assert abs(res.total_pct_change) < 1.0
    assert len(res.regressed_functions) == 0
    assert not res.has_regression


def test_compare_injected_regression_with_attribution():
    # Baseline: compute takes ~100ms, io takes ~100ms. Total ~200ms
    base_runs = [
        {("main", "compute"): 100 + i, ("main", "io"): 100}
        for i in range(-2, 3)
    ]
    # Candidate: compute regresses to ~160ms (+60ms), io unchanged (~100ms). Total ~260ms (+30%)
    cand_runs = [
        {("main", "compute"): 160 + i, ("main", "io"): 100}
        for i in range(-2, 3)
    ]
    base = _make_mock_set("base", base_runs)
    cand = _make_mock_set("cand", cand_runs)

    res = compare(base, cand, threshold=5.0, alpha=0.05, fdr_q=0.05)
    assert res.verdict == Verdict.FAIL
    assert res.has_regression
    assert res.total_pct_change > 25.0
    assert res.total_p_value < 0.05

    # Check function attribution
    assert len(res.regressed_functions) == 1
    reg_fn = res.regressed_functions[0]
    assert reg_fn.function == "compute"
    assert reg_fn.attribution_pct > 95.0  # Explains nearly 100% of the regression
    assert reg_fn.adjusted_p_value < 0.05

    # Check that raise_for_regression raises
    with pytest.raises(RegressionDetectedError):
        res.raise_for_regression()


def test_compare_improvement():
    # Candidate is 40% faster
    base_runs = [{("main", "task"): 200 + i} for i in range(-2, 3)]
    cand_runs = [{("main", "task"): 120 + i} for i in range(-2, 3)]

    base = _make_mock_set("base", base_runs)
    cand = _make_mock_set("cand", cand_runs)

    res = compare(base, cand, threshold=5.0)
    assert res.verdict == Verdict.IMPROVEMENT


def test_compare_new_and_removed_paths():
    base_runs = [
        {("main", "old_auth"): 50, ("main", "worker"): 100}
        for _ in range(5)
    ]
    cand_runs = [
        {("main", "new_auth"): 50, ("main", "worker"): 100}
        for _ in range(5)
    ]
    base = _make_mock_set("base", base_runs)
    cand = _make_mock_set("cand", cand_runs)

    res = compare(base, cand, threshold=5.0)
    # Check new paths
    assert any(p.path_str == "main;new_auth" for p in res.new_paths)
    # Check removed paths
    assert any(p.path_str == "main;old_auth" for p in res.removed_paths)


def test_compare_inconclusive_wide_ci():
    # High variance, small sample size K=2, M=2
    base_runs = [
        {("main", "task"): 50},
        {("main", "task"): 150},
    ]
    cand_runs = [
        {("main", "task"): 60},
        {("main", "task"): 140},
    ]
    base = _make_mock_set("base", base_runs)
    cand = _make_mock_set("cand", cand_runs)

    res = compare(base, cand, threshold=2.0)
    assert res.verdict == Verdict.INCONCLUSIVE
    assert res.power_warning is not None


def test_compare_statistical_methods():
    base_runs = [{("main", "task"): 100 + i} for i in range(5)]
    cand_runs = [{("main", "task"): 120 + i} for i in range(5)]
    base = _make_mock_set("base", base_runs)
    cand = _make_mock_set("cand", cand_runs)

    for method in ["permutation", "welch", "mann-whitney"]:
        cfg = ComparisonConfig(method=method, threshold_pct=5.0)
        res = compare(base, cand, config=cfg)
        assert res.verdict == Verdict.FAIL
        assert res.method == method


def test_compare_metric_modes():
    # Test 'inclusive' vs 'self'
    base_runs = [
        {("root", "middle", "leaf"): 100}
        for _ in range(5)
    ]
    cand_runs = [
        {("root", "middle", "leaf"): 150}
        for _ in range(5)
    ]
    base = _make_mock_set("base", base_runs)
    cand = _make_mock_set("cand", cand_runs)

    res_self = compare(base, cand, metric="self", threshold=5.0)
    assert res_self.metric == "self"
    assert res_self.regressed_functions[0].function == "leaf"

    res_inc = compare(base, cand, metric="inclusive", threshold=5.0)
    assert res_inc.metric == "inclusive"
    inc_funcs = {f.function for f in res_inc.regressed_functions}
    assert "root" in inc_funcs
    assert "middle" in inc_funcs
    assert "leaf" in inc_funcs
