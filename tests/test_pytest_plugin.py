"""
Tests for pytest integration plugin and assertion helpers.
"""

import pytest
from stackgate.exceptions import RegressionDetectedError
from stackgate.models import Profile, ProfileSet
from stackgate.pytest_plugin import assert_no_regression, stackgate_benchmark


def test_assert_no_regression_passes():
    p_base = [Profile(name=f"b_{i}", stacks={("task",): 100}) for i in range(5)]
    p_cand = [Profile(name=f"c_{i}", stacks={("task",): 100}) for i in range(5)]

    base = ProfileSet(name="base", profiles=p_base)
    cand = ProfileSet(name="cand", profiles=p_cand)

    res = assert_no_regression(base, cand, threshold=5.0)
    assert res.verdict == "PASS"


def test_assert_no_regression_fails():
    p_base = [Profile(name=f"b_{i}", stacks={("task",): 100}) for i in range(5)]
    p_cand = [Profile(name=f"c_{i}", stacks={("task",): 140}) for i in range(5)]

    base = ProfileSet(name="base", profiles=p_base)
    cand = ProfileSet(name="cand", profiles=p_cand)

    with pytest.raises(RegressionDetectedError) as exc_info:
        assert_no_regression(base, cand, threshold=5.0)
    assert "Performance regression detected" in str(exc_info.value)


def test_stackgate_benchmark_callable(stackgate_benchmark):
    def compute_work():
        return sum(range(10000))

    profiles = stackgate_benchmark(compute_work, runs=3, frequency_hz=200.0, warmup=1)

    assert isinstance(profiles, ProfileSet)
    assert profiles.num_runs == 3
