"""
Pytest integration plugin for StackGate.
Provides fixtures, markers, and assertion helpers for performance regression testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence, Union

from stackgate.comparator import ComparisonConfig, compare
from stackgate.exceptions import RegressionDetectedError
from stackgate.models import ComparisonResult, Profile, ProfileSet
from stackgate.recorder import record_callable


def assert_no_regression(
    baseline: Union[str, Path, Sequence[str | Path | Profile], ProfileSet, Profile],
    candidate: Union[str, Path, Sequence[str | Path | Profile], ProfileSet, Profile],
    threshold: float = 2.0,
    alpha: float = 0.05,
    strict: bool = False,
    config: ComparisonConfig | None = None,
    **kwargs: object,
) -> ComparisonResult:
    """
    Assert that the candidate profiles do not exhibit a statistically significant performance regression.

    Args:
        baseline: Baseline profile(s) or directory.
        candidate: Candidate profile(s) or directory.
        threshold: Allowed percentage regression threshold (e.g. 2.0 for 2%).
        alpha: Statistical significance level (default: 0.05).
        strict: If True, inconclusive comparisons fail the assertion.
        config: Optional pre-configured ComparisonConfig.
        **kwargs: Overrides for comparison settings.

    Returns:
        ComparisonResult if no regression is detected.

    Raises:
        RegressionDetectedError: If a statistically significant regression is found.
    """
    res = compare(
        baseline=baseline,
        candidate=candidate,
        threshold=threshold,
        alpha=alpha,
        strict=strict,
        config=config,
        **kwargs,
    )
    res.raise_for_regression(strict=strict)
    return res


# Pytest hooks (optional when pytest is imported)
try:
    import pytest

    def pytest_configure(config: Any) -> None:
        """Register the stackgate marker."""
        config.addinivalue_line(
            "markers",
            "stackgate(threshold=2.0, runs=5): Mark test for sampling profile regression gate.",
        )

    @pytest.fixture
    def stackgate_benchmark() -> Callable[..., ProfileSet]:
        """
        Pytest fixture to benchmark a callable under the sampling profiler.

        Usage:
            def test_feature(stackgate_benchmark):
                profiles = stackgate_benchmark(run_workload, runs=5)
                assert_no_regression("baselines/workload.folded", profiles)
        """
        def _benchmark(
            target: Callable[[], Any],
            runs: int = 5,
            frequency_hz: float = 200.0,
            warmup: int = 1,
            out_dir: str | Path | None = None,
        ) -> ProfileSet:
            return record_callable(
                target=target,
                runs=runs,
                frequency_hz=frequency_hz,
                warmup=warmup,
                out_dir=out_dir,
            )

        return _benchmark

except ImportError:
    pass
