"""
Tests for StackGate data models.
"""

import json
import pytest
from stackgate.exceptions import RegressionDetectedError
from stackgate.models import (
    ComparisonResult,
    FunctionDiff,
    PathDiff,
    Profile,
    ProfileSet,
    StackFrame,
    Verdict,
)


def test_stack_frame_parsing():
    f1 = StackFrame.from_string("my_func")
    assert f1.name == "my_func"
    assert f1.file is None
    assert f1.line is None
    assert str(f1) == "my_func"

    f2 = StackFrame.from_string("my_func (app/main.py:42)")
    assert f2.name == "my_func"
    assert f2.file == "app/main.py"
    assert f2.line == 42
    assert str(f2) == "my_func (app/main.py:42)"

    f3 = StackFrame.from_string("app/main.py:my_func:100")
    assert f3.name == "my_func"
    assert f3.file == "app/main.py"
    assert f3.line == 100

    f_empty = StackFrame.from_string("")
    assert f_empty.name == "<unknown>"


def test_profile_aggregations():
    stacks = {
        ("root", "worker", "parse"): 100,
        ("root", "worker", "compute"): 50,
        ("root", "io"): 50,
    }
    p = Profile(name="run1", stacks=stacks, sampling_rate_hz=1000.0)

    assert p.total_samples == 200
    assert p.total_duration == 0.2  # 200 / 1000

    # Self times
    self_samples = p.self_time_samples
    assert self_samples["parse"] == 100
    assert self_samples["compute"] == 50
    assert self_samples["io"] == 50
    assert "root" not in self_samples

    # Inclusive times
    inc_samples = p.inclusive_time_samples
    assert inc_samples["root"] == 200
    assert inc_samples["worker"] == 150
    assert inc_samples["parse"] == 100
    assert inc_samples["compute"] == 50
    assert inc_samples["io"] == 50

    # Durations
    assert p.self_time_duration["parse"] == 0.1
    assert p.inclusive_time_duration["worker"] == 0.15


def test_profile_recursion_handling():
    # Recursive stack: a -> b -> a -> b
    stacks = {
        ("a", "b", "a", "b"): 40,
    }
    p = Profile(name="rec", stacks=stacks, sampling_rate_hz=1000.0)
    assert p.self_time_samples["b"] == 40
    # In inclusive time, each unique function in a stack is counted once
    assert p.inclusive_time_samples["a"] == 40
    assert p.inclusive_time_samples["b"] == 40


def test_profile_serialization():
    stacks = {("main", "process"): 25}
    p = Profile(name="test_p", stacks=stacks, sampling_rate_hz=500.0, metadata={"env": "ci"})
    d = p.to_dict()
    assert d["name"] == "test_p"
    assert d["sampling_rate_hz"] == 500.0
    assert d["stacks"]["main;process"] == 25

    p2 = Profile.from_dict(d)
    assert p2.name == p.name
    assert p2.sampling_rate_hz == p.sampling_rate_hz
    assert p2.stacks == p.stacks


def test_profile_set_metrics():
    p1 = Profile(name="r1", stacks={("a",): 100}, sampling_rate_hz=1000.0)
    p2 = Profile(name="r2", stacks={("a",): 110}, sampling_rate_hz=1000.0)
    p3 = Profile(name="r3", stacks={("a",): 90}, sampling_rate_hz=1000.0)

    ps = ProfileSet(name="base", profiles=[p1, p2, p3])
    assert ps.num_runs == 3
    assert ps.total_samples == [100, 110, 90]
    assert abs(ps.mean_duration - 0.1) < 1e-6
    assert ps.std_duration > 0
    assert ps.cv_duration > 0
    assert ps.all_functions() == {"a"}
    assert ps.all_paths() == {("a",)}

    assert ps.get_function_durations("a", metric="self") == [0.1, 0.11, 0.09]
    assert ps.get_function_durations("nonexistent", metric="self") == [0.0, 0.0, 0.0]


def test_comparison_result_raise_for_regression():
    res_pass = ComparisonResult(
        verdict=Verdict.PASS,
        baseline_runs=5,
        candidate_runs=5,
        baseline_mean_duration=1.0,
        candidate_mean_duration=1.0,
        baseline_std_duration=0.01,
        candidate_std_duration=0.01,
        baseline_cv_pct=1.0,
        candidate_cv_pct=1.0,
        total_delta=0.0,
        total_pct_change=0.0,
        total_p_value=0.5,
        ci_lower=-1.0,
        ci_upper=1.0,
        ci_level=0.95,
        threshold_pct=2.0,
        alpha=0.05,
        fdr_q=0.05,
        method="permutation",
        metric="self",
        noise_floor_pct=1.5,
    )
    # Does not raise
    res_pass.raise_for_regression()

    res_fail = ComparisonResult(
        verdict=Verdict.FAIL,
        baseline_runs=5,
        candidate_runs=5,
        baseline_mean_duration=1.0,
        candidate_mean_duration=1.1,
        baseline_std_duration=0.01,
        candidate_std_duration=0.01,
        baseline_cv_pct=1.0,
        candidate_cv_pct=1.0,
        total_delta=0.1,
        total_pct_change=10.0,
        total_p_value=0.001,
        ci_lower=8.0,
        ci_upper=12.0,
        ci_level=0.95,
        threshold_pct=2.0,
        alpha=0.05,
        fdr_q=0.05,
        method="permutation",
        metric="self",
        noise_floor_pct=1.5,
    )
    with pytest.raises(RegressionDetectedError) as exc_info:
        res_fail.raise_for_regression()
    assert "Performance regression detected" in str(exc_info.value)
    assert exc_info.value.result is res_fail

    # INCONCLUSIVE with strict
    res_inconclusive = ComparisonResult(
        verdict=Verdict.INCONCLUSIVE,
        baseline_runs=2,
        candidate_runs=2,
        baseline_mean_duration=1.0,
        candidate_mean_duration=1.05,
        baseline_std_duration=0.2,
        candidate_std_duration=0.2,
        baseline_cv_pct=20.0,
        candidate_cv_pct=20.0,
        total_delta=0.05,
        total_pct_change=5.0,
        total_p_value=0.2,
        ci_lower=-10.0,
        ci_upper=20.0,
        ci_level=0.95,
        threshold_pct=2.0,
        alpha=0.05,
        fdr_q=0.05,
        method="permutation",
        metric="self",
        noise_floor_pct=1.5,
        power_warning="Sample count too low",
    )
    # strict=False does not raise
    res_inconclusive.raise_for_regression(strict=False)
    # strict=True raises
    with pytest.raises(RegressionDetectedError):
        res_inconclusive.raise_for_regression(strict=True)


def test_comparison_result_to_json():
    res = ComparisonResult(
        verdict=Verdict.PASS,
        baseline_runs=3,
        candidate_runs=3,
        baseline_mean_duration=1.0,
        candidate_mean_duration=1.0,
        baseline_std_duration=0.0,
        candidate_std_duration=0.0,
        baseline_cv_pct=0.0,
        candidate_cv_pct=0.0,
        total_delta=0.0,
        total_pct_change=0.0,
        total_p_value=1.0,
        ci_lower=0.0,
        ci_upper=0.0,
        ci_level=0.95,
        threshold_pct=2.0,
        alpha=0.05,
        fdr_q=0.05,
        method="permutation",
        metric="self",
        noise_floor_pct=1.0,
    )
    json_str = res.to_json()
    data = json.loads(json_str)
    assert data["verdict"] == "PASS"
    assert data["summary"]["baseline_runs"] == 3
