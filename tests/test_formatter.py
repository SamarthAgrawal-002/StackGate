"""
Tests for report formatting (Terminal, Markdown, HTML).
"""

from stackgate.formatter import format_html, format_markdown, format_terminal
from stackgate.models import (
    ComparisonResult,
    FunctionDiff,
    PathDiff,
    Verdict,
)


def _make_sample_result() -> ComparisonResult:
    func_diff = FunctionDiff(
        function="parse_json",
        baseline_mean=0.100,
        candidate_mean=0.180,
        baseline_std=0.005,
        candidate_std=0.005,
        delta=0.080,
        pct_change=80.0,
        p_value=0.0001,
        adjusted_p_value=0.0004,
        ci_lower=0.070,
        ci_upper=0.090,
        attribution_pct=88.9,
        is_significant=True,
        metric="self",
    )
    new_path = PathDiff(
        path=("main", "handle", "new_opt"),
        path_str="main;handle;new_opt",
        baseline_mean=0.0,
        candidate_mean=0.010,
        delta=0.010,
        pct_change=100.0,
        p_value=0.0,
        adjusted_p_value=0.0,
        is_new=True,
        is_removed=False,
    )
    return ComparisonResult(
        verdict=Verdict.FAIL,
        baseline_runs=5,
        candidate_runs=5,
        baseline_mean_duration=0.200,
        candidate_mean_duration=0.290,
        baseline_std_duration=0.008,
        candidate_std_duration=0.010,
        baseline_cv_pct=4.0,
        candidate_cv_pct=3.4,
        total_delta=0.090,
        total_pct_change=45.0,
        total_p_value=0.0002,
        ci_lower=38.0,
        ci_upper=52.0,
        ci_level=0.95,
        threshold_pct=2.0,
        alpha=0.05,
        fdr_q=0.05,
        method="permutation",
        metric="self",
        noise_floor_pct=1.8,
        regressed_functions=[func_diff],
        improved_functions=[],
        neutral_functions_count=42,
        new_paths=[new_path],
        removed_paths=[],
    )


def test_format_terminal():
    res = _make_sample_result()
    text = format_terminal(res, use_color=False)
    assert "Verdict: FAIL (Performance Regression Detected)" in text
    assert "parse_json" in text
    assert "+0.080s" in text
    assert "(88.9% of regression)" in text
    assert "main;handle;new_opt" in text
    assert "42 functions" in text


def test_format_markdown():
    res = _make_sample_result()
    md = format_markdown(res)
    assert "## 🚦 StackGate Performance Gate: ❌ **FAIL**" in md
    assert "| `parse_json` |" in md
    assert "88.9%" in md
    assert "<details><summary><b>New Call Paths (1)</b></summary>" in md
    assert "Runner Noise Floor" in md
    assert "StackGate" in md


def test_format_html():
    res = _make_sample_result()
    html_text = format_html(res)
    assert "<!DOCTYPE html>" in html_text
    assert "StackGate Performance Report" in html_text
    assert "parse_json" in html_text
    assert "FAIL" in html_text
