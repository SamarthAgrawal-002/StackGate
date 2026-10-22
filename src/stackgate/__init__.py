"""
StackGate: Statistically rigorous, function-level performance-regression detection from sampling profiles.

Author: Samarth Agrawal <agrawalsamarth0206@gmail.com>
"""

from __future__ import annotations

from stackgate.calibrator import calibrate
from stackgate.comparator import ComparisonConfig, compare
from stackgate.exceptions import (
    ConfigurationError,
    InsufficientRunsError,
    InvalidDataError,
    ProfileParseError,
    RegressionDetectedError,
    SamplingError,
    StackGateError,
)
from stackgate.models import (
    CalibrationResult,
    ComparisonResult,
    FunctionDiff,
    PathDiff,
    Profile,
    ProfileSet,
    StackFrame,
    Verdict,
)
from stackgate.parser import load_runs, parse_collapsed, parse_directory
from stackgate.pytest_plugin import assert_no_regression
from stackgate.recorder import record_callable, record_command, write_folded
from stackgate.sampler import Sampler, sample_callable
from stackgate.stats import normal_ppf

__version__ = "0.1.0"
__author__ = "Samarth Agrawal"
__email__ = "agrawalsamarth0206@gmail.com"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # Core API
    "compare",
    "calibrate",
    "assert_no_regression",
    "parse_collapsed",
    "parse_directory",
    "load_runs",
    "record_callable",
    "record_command",
    "write_folded",
    "Sampler",
    "sample_callable",
    "normal_ppf",
    # Data models
    "Profile",
    "ProfileSet",
    "StackFrame",
    "FunctionDiff",
    "PathDiff",
    "ComparisonResult",
    "CalibrationResult",
    "Verdict",
    "ComparisonConfig",
    # Exceptions
    "StackGateError",
    "ProfileParseError",
    "InsufficientRunsError",
    "InvalidDataError",
    "RegressionDetectedError",
    "SamplingError",
    "ConfigurationError",
]
