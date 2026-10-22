"""
Comprehensive coverage boost tests covering edge cases, branches, and error handling.
"""

import io
import json
import os
import sys
from pathlib import Path
import pytest
from stackgate.cli import main
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
from stackgate.formatter import format_html, format_markdown, format_terminal
from stackgate.models import (
    ComparisonResult,
    FunctionDiff,
    PathDiff,
    Profile,
    ProfileSet,
    StackFrame,
    Verdict,
)
from stackgate.parser import load_runs, parse_collapsed, parse_directory, parse_line
from stackgate.recorder import record_command, write_folded
from stackgate.sampler import _extract_stack
from stackgate.stats import (
    betainc,
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


def test_exception_formatting():
    err1 = ProfileParseError("Bad syntax", line_number=12, line_content="func;bad_count abc", file_path="prof.txt")
    s = str(err1)
    assert "Bad syntax in file 'prof.txt' at line 12 -> 'func;bad_count abc'" in s

    # Long line snippet truncation
    long_line = "a;" * 50 + " 100"
    err2 = ProfileParseError("Too long", line_number=1, line_content=long_line)
    assert "..." in str(err2)


def test_stats_edge_cases():
    # betainc validations
    with pytest.raises(ValueError):
        betainc(1.0, 1.0, -0.1)
    with pytest.raises(ValueError):
        betainc(1.0, 1.0, 1.1)

    # student_t_cdf validations
    with pytest.raises(ValueError):
        student_t_cdf(1.0, df=-1.0)
    assert student_t_cdf(-2.0, df=5.0) < 0.5

    # Welch t-test edge cases
    # Zero variances with different means
    p_gt, t_gt, _ = welch_ttest([1.0, 1.0], [2.0, 2.0], alternative="greater")
    assert p_gt == 0.0
    p_lt, t_lt, _ = welch_ttest([1.0, 1.0], [2.0, 2.0], alternative="less")
    assert p_lt == 1.0
    p_two, _, _ = welch_ttest([1.0, 1.0], [2.0, 2.0], alternative="two-sided")
    assert p_two == 0.0

    # Welch t-test identical with zero variance
    p_eq, _, _ = welch_ttest([3.0, 3.0], [3.0, 3.0], alternative="two-sided")
    assert p_eq == 1.0

    # Permutation test alternative='less'
    p_perm_lt, _ = permutation_test([5.0, 6.0, 7.0], [1.0, 2.0, 3.0], alternative="less")
    assert p_perm_lt < 0.1

    # Mann-Whitney alternative='less' and 'two-sided'
    p_mw_lt, _ = mann_whitney_u_test([5.0, 6.0, 7.0], [1.0, 2.0, 3.0], alternative="less")
    assert p_mw_lt < 0.05
    p_mw_two, _ = mann_whitney_u_test([1.0, 2.0, 3.0], [5.0, 6.0, 7.0], alternative="two-sided")
    assert p_mw_two < 0.1

    # Bootstrap with 1 sample
    d_l, d_h, p_l, p_h = bootstrap_confidence_interval([10.0], [12.0])
    assert d_l == 2.0 and d_h == 2.0


def test_formatter_all_verdicts_and_branches():
    # Test INCONCLUSIVE terminal
    res_inconcl = ComparisonResult(
        verdict=Verdict.INCONCLUSIVE,
        baseline_runs=2,
        candidate_runs=2,
        baseline_mean_duration=1.0,
        candidate_mean_duration=1.05,
        baseline_std_duration=0.1,
        candidate_std_duration=0.1,
        baseline_cv_pct=10.0,
        candidate_cv_pct=10.0,
        total_delta=0.05,
        total_pct_change=5.0,
        total_p_value=0.2,
        ci_lower=-5.0,
        ci_upper=15.0,
        ci_level=0.95,
        threshold_pct=2.0,
        alpha=0.05,
        fdr_q=0.05,
        method="permutation",
        metric="self",
        noise_floor_pct=2.0,
        warnings=["CV is high", "Sample size low"],
    )
    term_inc = format_terminal(res_inconcl, use_color=True)
    assert "INCONCLUSIVE" in term_inc
    assert "Warning: CV is high" in term_inc

    md_inc = format_markdown(res_inconcl)
    assert "INCONCLUSIVE" in md_inc
    assert "> [!WARNING]" in md_inc

    # Test IMPROVEMENT terminal
    res_imp = ComparisonResult(
        verdict=Verdict.IMPROVEMENT,
        baseline_runs=5,
        candidate_runs=5,
        baseline_mean_duration=2.0,
        candidate_mean_duration=1.0,
        baseline_std_duration=0.05,
        candidate_std_duration=0.05,
        baseline_cv_pct=2.5,
        candidate_cv_pct=5.0,
        total_delta=-1.0,
        total_pct_change=-50.0,
        total_p_value=0.0001,
        ci_lower=-55.0,
        ci_upper=-45.0,
        ci_level=0.95,
        threshold_pct=2.0,
        alpha=0.05,
        fdr_q=0.05,
        method="permutation",
        metric="self",
        noise_floor_pct=1.0,
        improved_functions=[
            FunctionDiff(
                function="fast_algo",
                baseline_mean=1.5,
                candidate_mean=0.5,
                baseline_std=0.05,
                candidate_std=0.05,
                delta=-1.0,
                pct_change=-66.7,
                p_value=0.0001,
                adjusted_p_value=0.0001,
                ci_lower=-1.1,
                ci_upper=-0.9,
                attribution_pct=0.0,
                is_significant=True,
            )
        ],
        removed_paths=[
            PathDiff(
                path=("main", "slow_path"),
                path_str="main;slow_path",
                baseline_mean=0.5,
                candidate_mean=0.0,
                delta=-0.5,
                pct_change=-100.0,
                p_value=0.0,
                adjusted_p_value=0.0,
                is_new=False,
                is_removed=True,
            )
        ],
    )
    term_imp = format_terminal(res_imp, use_color=False)
    assert "Speedup" in term_imp
    assert "fast_algo" in term_imp
    assert "Removed call paths" in term_imp

    md_imp = format_markdown(res_imp)
    assert "⚡ Improved Functions" in md_imp


def test_cli_formats_and_record(tmp_path: Path):
    base_dir = tmp_path / "base"
    cand_dir = tmp_path / "cand"
    base_dir.mkdir()
    cand_dir.mkdir()
    (base_dir / "r1.folded").write_text("a;b 100\n", encoding="utf-8")
    (base_dir / "r2.folded").write_text("a;b 102\n", encoding="utf-8")
    (cand_dir / "r1.folded").write_text("a;b 100\n", encoding="utf-8")
    (cand_dir / "r2.folded").write_text("a;b 101\n", encoding="utf-8")

    # CLI compare markdown format
    md_out = tmp_path / "report.md"
    ret = main(["compare", str(base_dir), str(cand_dir), "--format", "markdown", "-o", str(md_out)])
    assert ret == 0
    assert md_out.exists()
    assert "StackGate Performance Gate" in md_out.read_text(encoding="utf-8")

    # CLI compare HTML format
    html_out = tmp_path / "report.html"
    ret = main(["compare", str(base_dir), str(cand_dir), "--format", "html", "-o", str(html_out)])
    assert ret == 0
    assert html_out.exists()
    assert "<!DOCTYPE html>" in html_out.read_text(encoding="utf-8")

    # CLI record with builtin engine
    rec_out = tmp_path / "rec_profiles"
    ret = main([
        "record",
        "--runs", "2",
        "--out", str(rec_out),
        "--engine", "builtin",
        "--",
        sys.executable, "-c", "import time; time.sleep(0.01)"
    ])
    assert ret == 0
    assert (rec_out / "manifest.json").exists()
    assert (rec_out / "run_001.folded").exists()


def test_parser_strict_errors(tmp_path: Path):
    bad_file = tmp_path / "bad.folded"
    bad_file.write_text("func_without_count\n", encoding="utf-8")

    with pytest.raises(ProfileParseError):
        parse_collapsed(bad_file, strict=True)


def test_models_additional_coverage():
    f = StackFrame.from_string("main.py:func (custom)")
    assert f.name == "main.py:func"
    assert f.file == "custom"

    # Profile from_dict with empty stacks
    p = Profile.from_dict({"name": "empty", "sampling_rate_hz": 100.0, "stacks": {}})
    assert p.total_samples == 0
    assert p.total_duration == 0.0

    # ProfileSet cv_duration when mean duration is 0
    p0 = Profile(name="zero", stacks={})
    ps0 = ProfileSet(name="z", profiles=[p0])
    assert ps0.cv_duration == 0.0
    assert ps0.std_duration == 0.0
