"""
Data models for StackGate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from stackgate.exceptions import RegressionDetectedError


class Verdict(str, Enum):
    """Overall outcome of a performance comparison."""

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    IMPROVEMENT = "IMPROVEMENT"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class StackFrame:
    """Represents a single execution stack frame."""

    name: str
    file: str | None = None
    line: int | None = None

    @classmethod
    def from_string(cls, raw: str) -> StackFrame:
        """
        Parse a frame string into a StackFrame.
        Handles formats like:
          'my_func'
          'my_func (module.py:42)'
          'module.py:my_func:42'
        """
        raw = raw.strip()
        if not raw:
            return cls(name="<unknown>")

        # Check for 'func (file:line)' or 'func (file)'
        if "(" in raw and raw.endswith(")"):
            name_part, loc_part = raw[:-1].split("(", 1)
            name = name_part.strip()
            loc_part = loc_part.strip()
            if ":" in loc_part:
                f, l = loc_part.rsplit(":", 1)
                try:
                    return cls(name=name, file=f, line=int(l))
                except ValueError:
                    return cls(name=name, file=loc_part)
            return cls(name=name, file=loc_part)

        # Handle Windows drive letters (e.g. C:\...) before splitting on colon
        colon_parts: list[str] = []
        work = raw
        if len(work) >= 3 and work[1] == ":" and (work[2] == "\\" or work[2] == "/"):
            drive = work[:2]
            rest = work[2:]
            sub_parts = rest.split(":")
            colon_parts = [drive + sub_parts[0]] + sub_parts[1:]
        else:
            colon_parts = work.split(":")

        if len(colon_parts) == 3:
            f, p1, p2 = colon_parts
            # Check if p2 is line number: file:func:line
            try:
                line_num = int(p2)
                return cls(name=p1, file=f, line=line_num)
            except ValueError:
                pass
            # Check if p1 is line number: file:line:func
            try:
                line_num = int(p1)
                return cls(name=p2, file=f, line=line_num)
            except ValueError:
                pass
            return cls(name=p1, file=f)

        if len(colon_parts) == 2:
            p0, p1 = colon_parts
            try:
                line_num = int(p1)
                return cls(name=p0, line=line_num)
            except ValueError:
                return cls(name=p1, file=p0)

        return cls(name=raw)

    def __str__(self) -> str:
        if self.file and self.line is not None:
            return f"{self.name} ({self.file}:{self.line})"
        if self.file:
            return f"{self.name} ({self.file})"
        return self.name


@dataclass
class Profile:
    """
    Represents a single profile run consisting of call stacks and sample counts.
    """

    name: str
    stacks: dict[tuple[str, ...], int]
    sampling_rate_hz: float = 1000.0  # Default 1000Hz (1 sample = 1ms)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Cached aggregations
    _total_samples: int | None = field(default=None, init=False, repr=False)
    _self_samples: dict[str, int] | None = field(default=None, init=False, repr=False)
    _inclusive_samples: dict[str, int] | None = field(default=None, init=False, repr=False)

    @property
    def total_samples(self) -> int:
        """Total number of samples collected in this profile."""
        if self._total_samples is None:
            self._total_samples = sum(self.stacks.values())
        return self._total_samples

    @property
    def total_duration(self) -> float:
        """Estimated total execution duration in seconds."""
        if self.sampling_rate_hz <= 0:
            return float(self.total_samples)
        return self.total_samples / self.sampling_rate_hz

    @property
    def self_time_samples(self) -> dict[str, int]:
        """Exclusive sample counts per function (leaf frame in call stack)."""
        if self._self_samples is None:
            res: dict[str, int] = {}
            for path, count in self.stacks.items():
                if path:
                    leaf = path[-1]
                    res[leaf] = res.get(leaf, 0) + count
            self._self_samples = res
        return self._self_samples

    @property
    def inclusive_time_samples(self) -> dict[str, int]:
        """Cumulative sample counts per function (appearing anywhere in call stack)."""
        if self._inclusive_samples is None:
            res: dict[str, int] = {}
            for path, count in self.stacks.items():
                # Count each unique function once per stack occurrence to handle recursion
                for fn in set(path):
                    res[fn] = res.get(fn, 0) + count
            self._inclusive_samples = res
        return self._inclusive_samples

    @property
    def self_time_duration(self) -> dict[str, float]:
        """Exclusive time in seconds per function."""
        rate = self.sampling_rate_hz if self.sampling_rate_hz > 0 else 1.0
        return {fn: count / rate for fn, count in self.self_time_samples.items()}

    @property
    def inclusive_time_duration(self) -> dict[str, float]:
        """Inclusive time in seconds per function."""
        rate = self.sampling_rate_hz if self.sampling_rate_hz > 0 else 1.0
        return {fn: count / rate for fn, count in self.inclusive_time_samples.items()}

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile to dictionary."""
        return {
            "name": self.name,
            "sampling_rate_hz": self.sampling_rate_hz,
            "metadata": self.metadata,
            "total_samples": self.total_samples,
            "total_duration": self.total_duration,
            "stacks": {";".join(path): count for path, count in self.stacks.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Profile:
        """Deserialize profile from dictionary."""
        raw_stacks = data.get("stacks", {})
        stacks = {
            tuple(p.split(";")) if p else (): int(count)
            for p, count in raw_stacks.items()
        }
        return cls(
            name=data.get("name", "profile"),
            stacks=stacks,
            sampling_rate_hz=float(data.get("sampling_rate_hz", 1000.0)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProfileSet:
    """
    A collection of independent profile runs for one candidate or baseline.
    """

    name: str
    profiles: list[Profile]

    def __post_init__(self) -> None:
        if not self.profiles:
            raise ValueError(f"ProfileSet '{self.name}' must contain at least one profile.")

    @property
    def num_runs(self) -> int:
        return len(self.profiles)

    @property
    def total_durations(self) -> list[float]:
        return [p.total_duration for p in self.profiles]

    @property
    def total_samples(self) -> list[int]:
        return [p.total_samples for p in self.profiles]

    @property
    def mean_duration(self) -> float:
        durs = self.total_durations
        return sum(durs) / len(durs)

    @property
    def std_duration(self) -> float:
        durs = self.total_durations
        n = len(durs)
        if n < 2:
            return 0.0
        mean = sum(durs) / n
        var = sum((x - mean) ** 2 for x in durs) / (n - 1)
        return var ** 0.5

    @property
    def cv_duration(self) -> float:
        """Coefficient of variation (std / mean) as a percentage."""
        mean = self.mean_duration
        if mean <= 0:
            return 0.0
        return (self.std_duration / mean) * 100.0

    def all_functions(self) -> set[str]:
        """Set of all unique function names appearing across all profiles."""
        funcs: set[str] = set()
        for p in self.profiles:
            funcs.update(p.inclusive_time_samples.keys())
        return funcs

    def all_paths(self) -> set[tuple[str, ...]]:
        """Set of all unique call paths appearing across all profiles."""
        paths: set[tuple[str, ...]] = set()
        for p in self.profiles:
            paths.update(p.stacks.keys())
        return paths

    def get_function_durations(self, function: str, metric: str = "self") -> list[float]:
        """
        Return the duration array for a function across all runs.
        Returns 0.0 for runs where the function was not sampled.
        """
        vals: list[float] = []
        for p in self.profiles:
            dur_map = p.self_time_duration if metric == "self" else p.inclusive_time_duration
            vals.append(dur_map.get(function, 0.0))
        return vals

    def get_path_durations(self, path: tuple[str, ...]) -> list[float]:
        """
        Return the duration array for an exact call path across all runs.
        """
        vals: list[float] = []
        for p in self.profiles:
            count = p.stacks.get(path, 0)
            rate = p.sampling_rate_hz if p.sampling_rate_hz > 0 else 1.0
            vals.append(count / rate)
        return vals


@dataclass
class FunctionDiff:
    """Statistical difference and attribution for a single function."""

    function: str
    baseline_mean: float
    candidate_mean: float
    baseline_std: float
    candidate_std: float
    delta: float
    pct_change: float
    p_value: float
    adjusted_p_value: float  # FDR q-value
    ci_lower: float
    ci_upper: float
    attribution_pct: float
    is_significant: bool
    metric: str = "self"
    is_new: bool = False
    is_removed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "metric": self.metric,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "baseline_std": self.baseline_std,
            "candidate_std": self.candidate_std,
            "delta": self.delta,
            "pct_change": self.pct_change,
            "p_value": self.p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "attribution_pct": self.attribution_pct,
            "is_significant": self.is_significant,
            "is_new": self.is_new,
            "is_removed": self.is_removed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FunctionDiff:
        return cls(
            function=str(data.get("function", "")),
            baseline_mean=float(data.get("baseline_mean", 0.0)),
            candidate_mean=float(data.get("candidate_mean", 0.0)),
            baseline_std=float(data.get("baseline_std", 0.0)),
            candidate_std=float(data.get("candidate_std", 0.0)),
            delta=float(data.get("delta", 0.0)),
            pct_change=float(data.get("pct_change", 0.0)),
            p_value=float(data.get("p_value", 1.0)),
            adjusted_p_value=float(data.get("adjusted_p_value", 1.0)),
            ci_lower=float(data.get("ci_lower", 0.0)),
            ci_upper=float(data.get("ci_upper", 0.0)),
            attribution_pct=float(data.get("attribution_pct", 0.0)),
            is_significant=bool(data.get("is_significant", False)),
            metric=str(data.get("metric", "self")),
            is_new=bool(data.get("is_new", False)),
            is_removed=bool(data.get("is_removed", False)),
        )


@dataclass
class PathDiff:
    """Statistical difference for an exact call path."""

    path: tuple[str, ...]
    path_str: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    pct_change: float
    p_value: float
    adjusted_p_value: float
    is_new: bool
    is_removed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": list(self.path),
            "path_str": self.path_str,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "delta": self.delta,
            "pct_change": self.pct_change,
            "p_value": self.p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "is_new": self.is_new,
            "is_removed": self.is_removed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PathDiff:
        raw_path = data.get("path", [])
        path_tuple = tuple(raw_path) if isinstance(raw_path, (list, tuple)) else ()
        path_str = str(data.get("path_str", ";".join(path_tuple)))
        return cls(
            path=path_tuple,
            path_str=path_str,
            baseline_mean=float(data.get("baseline_mean", 0.0)),
            candidate_mean=float(data.get("candidate_mean", 0.0)),
            delta=float(data.get("delta", 0.0)),
            pct_change=float(data.get("pct_change", 0.0)),
            p_value=float(data.get("p_value", 0.0)),
            adjusted_p_value=float(data.get("adjusted_p_value", 0.0)),
            is_new=bool(data.get("is_new", False)),
            is_removed=bool(data.get("is_removed", False)),
        )


@dataclass
class ComparisonResult:
    """Full outcome of comparing baseline profiles against candidate profiles."""

    verdict: Verdict
    baseline_runs: int
    candidate_runs: int
    baseline_mean_duration: float
    candidate_mean_duration: float
    baseline_std_duration: float
    candidate_std_duration: float
    baseline_cv_pct: float
    candidate_cv_pct: float
    total_delta: float
    total_pct_change: float
    total_p_value: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    threshold_pct: float
    alpha: float
    fdr_q: float
    method: str
    metric: str
    noise_floor_pct: float
    regressed_functions: list[FunctionDiff] = field(default_factory=list)
    improved_functions: list[FunctionDiff] = field(default_factory=list)
    neutral_functions_count: int = 0
    new_paths: list[PathDiff] = field(default_factory=list)
    removed_paths: list[PathDiff] = field(default_factory=list)
    power_warning: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def has_regression(self) -> bool:
        """Whether a regression was flagged."""
        return self.verdict == Verdict.FAIL

    def raise_for_regression(self, strict: bool = False) -> None:
        """
        Raise RegressionDetectedError if verdict is FAIL, or if strict and INCONCLUSIVE.
        """
        if self.verdict == Verdict.FAIL:
            msg = (
                f"Performance regression detected: total duration changed by "
                f"{self.total_pct_change:+.2f}% (p={self.total_p_value:.4f}, "
                f"threshold={self.threshold_pct:.2f}%)."
            )
            raise RegressionDetectedError(msg, result=self)
        if strict and self.verdict == Verdict.INCONCLUSIVE:
            msg = (
                f"Comparison is INCONCLUSIVE: {self.power_warning or 'High variance or insufficient runs'}."
            )
            raise RegressionDetectedError(msg, result=self)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "verdict": self.verdict.value,
            "summary": {
                "baseline_runs": self.baseline_runs,
                "candidate_runs": self.candidate_runs,
                "baseline_mean_duration": self.baseline_mean_duration,
                "candidate_mean_duration": self.candidate_mean_duration,
                "baseline_std_duration": self.baseline_std_duration,
                "candidate_std_duration": self.candidate_std_duration,
                "baseline_cv_pct": self.baseline_cv_pct,
                "candidate_cv_pct": self.candidate_cv_pct,
                "total_delta": self.total_delta,
                "total_pct_change": self.total_pct_change,
                "total_p_value": self.total_p_value,
                "ci_lower": self.ci_lower,
                "ci_upper": self.ci_upper,
                "ci_level": self.ci_level,
                "threshold_pct": self.threshold_pct,
                "alpha": self.alpha,
                "fdr_q": self.fdr_q,
                "method": self.method,
                "metric": self.metric,
                "noise_floor_pct": self.noise_floor_pct,
                "neutral_functions_count": self.neutral_functions_count,
            },
            "power_warning": self.power_warning,
            "warnings": self.warnings,
            "regressed_functions": [f.to_dict() for f in self.regressed_functions],
            "improved_functions": [f.to_dict() for f in self.improved_functions],
            "new_paths": [p.to_dict() for p in self.new_paths],
            "removed_paths": [p.to_dict() for p in self.removed_paths],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Render markdown report suitable for GitHub PR comments."""
        from stackgate.formatter import format_markdown
        return format_markdown(self)

    def to_terminal(self, use_color: bool = True) -> str:
        """Render terminal report with ANSI formatting."""
        from stackgate.formatter import format_terminal
        return format_terminal(self, use_color=use_color)

    def to_html(self) -> str:
        """Render stand-alone HTML report."""
        from stackgate.formatter import format_html
        return format_html(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComparisonResult:
        summary = data.get("summary", {})
        verdict_raw = data.get("verdict", "PASS")
        try:
            verdict = Verdict(verdict_raw)
        except ValueError:
            verdict = Verdict.PASS

        regressed = [FunctionDiff.from_dict(f) for f in data.get("regressed_functions", [])]
        improved = [FunctionDiff.from_dict(f) for f in data.get("improved_functions", [])]
        new_paths = [PathDiff.from_dict(p) for p in data.get("new_paths", [])]
        removed_paths = [PathDiff.from_dict(p) for p in data.get("removed_paths", [])]

        return cls(
            verdict=verdict,
            baseline_runs=int(summary.get("baseline_runs", 1)),
            candidate_runs=int(summary.get("candidate_runs", 1)),
            baseline_mean_duration=float(summary.get("baseline_mean_duration", 0.0)),
            candidate_mean_duration=float(summary.get("candidate_mean_duration", 0.0)),
            baseline_std_duration=float(summary.get("baseline_std_duration", 0.0)),
            candidate_std_duration=float(summary.get("candidate_std_duration", 0.0)),
            baseline_cv_pct=float(summary.get("baseline_cv_pct", 0.0)),
            candidate_cv_pct=float(summary.get("candidate_cv_pct", 0.0)),
            total_delta=float(summary.get("total_delta", 0.0)),
            total_pct_change=float(summary.get("total_pct_change", 0.0)),
            total_p_value=float(summary.get("total_p_value", 1.0)),
            ci_lower=float(summary.get("ci_lower", 0.0)),
            ci_upper=float(summary.get("ci_upper", 0.0)),
            ci_level=float(summary.get("ci_level", 0.95)),
            threshold_pct=float(summary.get("threshold_pct", 2.0)),
            alpha=float(summary.get("alpha", 0.05)),
            fdr_q=float(summary.get("fdr_q", 0.05)),
            method=str(summary.get("method", "permutation")),
            metric=str(summary.get("metric", "self")),
            noise_floor_pct=float(summary.get("noise_floor_pct", 1.5)),
            regressed_functions=regressed,
            improved_functions=improved,
            neutral_functions_count=int(summary.get("neutral_functions_count", 0)),
            new_paths=new_paths,
            removed_paths=removed_paths,
            power_warning=data.get("power_warning"),
            warnings=list(data.get("warnings", [])),
        )


@dataclass
class CalibrationResult:
    """Outcome of A/A calibration and sample size analysis."""

    runs_evaluated: int
    splits_performed: int
    empirical_fpr: float
    noise_floor_pct: float
    mean_duration: float
    cv_pct: float
    recommended_k: dict[str, int]
    alpha: float
    permutations_tested: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs_evaluated": self.runs_evaluated,
            "splits_performed": self.splits_performed,
            "empirical_fpr": self.empirical_fpr,
            "noise_floor_pct": self.noise_floor_pct,
            "mean_duration": self.mean_duration,
            "cv_pct": self.cv_pct,
            "recommended_k": self.recommended_k,
            "alpha": self.alpha,
            "permutations_tested": self.permutations_tested,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationResult:
        return cls(
            runs_evaluated=int(data.get("runs_evaluated", 0)),
            splits_performed=int(data.get("splits_performed", 0)),
            empirical_fpr=float(data.get("empirical_fpr", 0.0)),
            noise_floor_pct=float(data.get("noise_floor_pct", 0.0)),
            mean_duration=float(data.get("mean_duration", 0.0)),
            cv_pct=float(data.get("cv_pct", 0.0)),
            recommended_k=dict(data.get("recommended_k", {})),
            alpha=float(data.get("alpha", 0.05)),
            permutations_tested=int(data.get("permutations_tested", 0)),
        )

    def summary(self) -> str:
        k_recs = ", ".join(f"{mde}: K={k}" for mde, k in self.recommended_k.items())
        return (
            f"A/A Calibration on {self.runs_evaluated} runs ({self.splits_performed} splits):\n"
            f"  Empirical False Positive Rate: {self.empirical_fpr * 100:.1f}% (target: {self.alpha * 100:.1f}%)\n"
            f"  Run-to-run noise floor (95th pct): {self.noise_floor_pct:.2f}%\n"
            f"  Duration CV: {self.cv_pct:.2f}%\n"
            f"  Recommended sample sizes for 80% power: [{k_recs}]"
        )
