"""
Robustness and regression tests verifying all bug fixes, boundary conditions, and edge cases.
"""

import json
import math
import sys
import tempfile
from pathlib import Path
import pytest

from stackgate.calibrator import calibrate
from stackgate.cli import main
from stackgate.comparator import ComparisonConfig, compare
from stackgate.exceptions import (
    ConfigurationError,
    InsufficientRunsError,
    InvalidDataError,
    ProfileParseError,
)
from stackgate.formatter import format_html, format_markdown, format_terminal
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
from stackgate.parser import load_runs, parse_collapsed, parse_line
from stackgate.recorder import record_command
from stackgate.sampler import Sampler
from stackgate.stats import (
    benjamini_hochberg,
    bonferroni,
    bootstrap_confidence_interval,
    normal_ppf,
    permutation_test,
)


def _make_mock_set(name: str, runs: list[dict[tuple[str, ...], int]]) -> ProfileSet:
    profiles = [
        Profile(name=f"{name}_{i+1}", stacks=run, sampling_rate_hz=1000.0)
        for i, run in enumerate(runs)
    ]
    return ProfileSet(name=name, profiles=profiles)


def test_permutation_test_order_invariance():
    """Verify permutation test is strictly order-invariant across sample permutations."""
    x = [1.0, 2.0]
    y1 = [1.0, 2.0]
    y2 = [2.0, 1.0]

    p1, _ = permutation_test(x, y1, alternative="greater")
    p2, _ = permutation_test(x, y2, alternative="greater")
    assert abs(p1 - p2) < 1e-12

    # Multiset equality with larger sets
    a = [10.0, 20.0, 30.0, 40.0]
    b1 = [10.0, 20.0, 30.0, 40.0]
    b2 = [40.0, 10.0, 30.0, 20.0]
    pa, _ = permutation_test(a, b1, alternative="greater")
    pb, _ = permutation_test(a, b2, alternative="greater")
    assert abs(pa - pb) < 1e-12

    # Zero variance across pooled values
    p_zero, _ = permutation_test([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    assert p_zero == 1.0


def test_compare_improvement_function_attribution():
    """Verify functions that improve are correctly detected, attributed, and stored in improved_functions."""
    base_runs = [
        {("main", "parse_json"): 500 + i, ("main", "compute"): 200}
        for i in range(-2, 3)
    ]
    # parse_json speeds up by 90% (from ~500ms to ~50ms)
    cand_runs = [
        {("main", "parse_json"): 50 + i, ("main", "compute"): 200}
        for i in range(-2, 3)
    ]
    base = _make_mock_set("base", base_runs)
    cand = _make_mock_set("cand", cand_runs)

    res = compare(base, cand, threshold=5.0, alpha=0.05, fdr_q=0.05)
    assert res.verdict == Verdict.IMPROVEMENT
    assert len(res.improved_functions) == 1
    imp_fn = res.improved_functions[0]
    assert imp_fn.function == "parse_json"
    assert imp_fn.delta < -400.0 / 1000.0  # ~ -0.45s
    assert imp_fn.attribution_pct > 90.0  # Explains nearly all of the speedup
    assert imp_fn.adjusted_p_value < 0.05


def test_comparison_result_and_calibration_result_roundtrip():
    """Verify full JSON serialization and deserialization roundtrip for ComparisonResult and CalibrationResult."""
    fn_reg = FunctionDiff(
        function="slow_func",
        baseline_mean=0.1,
        candidate_mean=0.2,
        baseline_std=0.01,
        candidate_std=0.01,
        delta=0.1,
        pct_change=100.0,
        p_value=0.001,
        adjusted_p_value=0.002,
        ci_lower=0.08,
        ci_upper=0.12,
        attribution_pct=100.0,
        is_significant=True,
    )
    fn_imp = FunctionDiff(
        function="fast_func",
        baseline_mean=0.3,
        candidate_mean=0.1,
        baseline_std=0.01,
        candidate_std=0.01,
        delta=-0.2,
        pct_change=-66.7,
        p_value=0.0001,
        adjusted_p_value=0.0002,
        ci_lower=-0.22,
        ci_upper=-0.18,
        attribution_pct=0.0,
        is_significant=True,
    )
    p_new = PathDiff(
        path=("main", "new_step"),
        path_str="main;new_step",
        baseline_mean=0.0,
        candidate_mean=0.05,
        delta=0.05,
        pct_change=100.0,
        p_value=0.0,
        adjusted_p_value=0.0,
        is_new=True,
        is_removed=False,
    )
    p_rem = PathDiff(
        path=("main", "old_step"),
        path_str="main;old_step",
        baseline_mean=0.05,
        candidate_mean=0.0,
        delta=-0.05,
        pct_change=-100.0,
        p_value=0.0,
        adjusted_p_value=0.0,
        is_new=False,
        is_removed=True,
    )

    res = ComparisonResult(
        verdict=Verdict.FAIL,
        baseline_runs=5,
        candidate_runs=5,
        baseline_mean_duration=1.0,
        candidate_mean_duration=1.2,
        baseline_std_duration=0.02,
        candidate_std_duration=0.02,
        baseline_cv_pct=2.0,
        candidate_cv_pct=1.7,
        total_delta=0.2,
        total_pct_change=20.0,
        total_p_value=0.001,
        ci_lower=15.0,
        ci_upper=25.0,
        ci_level=0.95,
        threshold_pct=2.0,
        alpha=0.05,
        fdr_q=0.05,
        method="permutation",
        metric="self",
        noise_floor_pct=1.5,
        regressed_functions=[fn_reg],
        improved_functions=[fn_imp],
        neutral_functions_count=10,
        new_paths=[p_new],
        removed_paths=[p_rem],
        warnings=["Low runner stability"],
    )

    # JSON roundtrip
    json_str = res.to_json()
    loaded_dict = json.loads(json_str)
    reconstructed = ComparisonResult.from_dict(loaded_dict)

    assert reconstructed.verdict == res.verdict
    assert reconstructed.total_delta == res.total_delta
    assert len(reconstructed.regressed_functions) == 1
    assert reconstructed.regressed_functions[0].function == "slow_func"
    assert len(reconstructed.improved_functions) == 1
    assert reconstructed.improved_functions[0].function == "fast_func"
    assert len(reconstructed.new_paths) == 1
    assert reconstructed.new_paths[0].path == ("main", "new_step")
    assert len(reconstructed.removed_paths) == 1
    assert reconstructed.warnings == ["Low runner stability"]

    # CalibrationResult roundtrip
    cal = CalibrationResult(
        runs_evaluated=10,
        splits_performed=50,
        empirical_fpr=0.04,
        noise_floor_pct=1.2,
        mean_duration=2.5,
        cv_pct=1.8,
        recommended_k={"2.0%": 5, "5.0%": 3},
        alpha=0.05,
        permutations_tested=100000,
    )
    cal_dict = cal.to_dict()
    cal_reconstructed = CalibrationResult.from_dict(cal_dict)
    assert cal_reconstructed.runs_evaluated == 10
    assert cal_reconstructed.noise_floor_pct == 1.2
    assert cal_reconstructed.recommended_k == {"2.0%": 5, "5.0%": 3}


def test_cli_report_full_rendering(tmp_path: Path):
    """Verify stackgate report CLI command re-renders JSON into Markdown and HTML."""
    json_path = tmp_path / "result.json"
    res = ComparisonResult(
        verdict=Verdict.FAIL,
        baseline_runs=5,
        candidate_runs=5,
        baseline_mean_duration=1.0,
        candidate_mean_duration=1.2,
        baseline_std_duration=0.05,
        candidate_std_duration=0.05,
        baseline_cv_pct=5.0,
        candidate_cv_pct=4.2,
        total_delta=0.2,
        total_pct_change=20.0,
        total_p_value=0.001,
        ci_lower=15.0,
        ci_upper=25.0,
        ci_level=0.95,
        threshold_pct=2.0,
        alpha=0.05,
        fdr_q=0.05,
        method="permutation",
        metric="self",
        noise_floor_pct=1.5,
        regressed_functions=[
            FunctionDiff(
                function="heavy_calc",
                baseline_mean=0.5,
                candidate_mean=0.7,
                baseline_std=0.02,
                candidate_std=0.02,
                delta=0.2,
                pct_change=40.0,
                p_value=0.0001,
                adjusted_p_value=0.0001,
                ci_lower=0.18,
                ci_upper=0.22,
                attribution_pct=100.0,
                is_significant=True,
            )
        ],
    )
    json_path.write_text(res.to_json(), encoding="utf-8")

    # Render HTML
    html_out = tmp_path / "report.html"
    ret = main(["report", str(json_path), "--format", "html", "-o", str(html_out)])
    assert ret == 0
    assert html_out.exists()
    content = html_out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "heavy_calc" in content
    assert "FAIL" in content

    # Render Markdown
    md_out = tmp_path / "report.md"
    ret_md = main(["report", str(json_path), "--format", "markdown", "-o", str(md_out)])
    assert ret_md == 0
    assert md_out.exists()
    md_content = md_out.read_text(encoding="utf-8")
    assert "## StackGate Performance Gate: **FAIL**" in md_content
    assert "| `heavy_calc` |" in md_content

    # Error on non-existent file
    ret_missing = main(["report", "non_existent_file.json"])
    assert ret_missing == 3


def test_load_runs_with_tuple_and_sequences(tmp_path: Path):
    """Verify load_runs supports tuples, generic sequences, and directory paths inside lists."""
    p1 = Profile("p1", {("a",): 10})
    p2 = Profile("p2", {("a",): 20})

    # From tuple of Profiles
    ps_tuple = load_runs((p1, p2))
    assert ps_tuple.num_runs == 2

    # From tuple of folded strings
    ps_str_tuple = load_runs(("func1 10", "func2 20"))
    assert ps_str_tuple.num_runs == 2

    # From list containing directory path
    sub_dir = tmp_path / "sub_profiles"
    sub_dir.mkdir()
    (sub_dir / "run1.folded").write_text("main;work 100\n", encoding="utf-8")
    (sub_dir / "run2.folded").write_text("main;work 105\n", encoding="utf-8")

    ps_dir_in_list = load_runs([sub_dir])
    assert ps_dir_in_list.num_runs == 2

    # Empty sequence raises InvalidDataError
    with pytest.raises(InvalidDataError):
        load_runs([])
    with pytest.raises(InvalidDataError):
        load_runs(())


def test_stack_frame_windows_paths_and_variants():
    """Verify StackFrame.from_string handles Windows drive letters, file:line:func, and file:func:line."""
    # Windows path file:func:line
    f1 = StackFrame.from_string(r"C:\Users\sam\app.py:main:42")
    assert f1.name == "main"
    assert f1.file == r"C:\Users\sam\app.py"
    assert f1.line == 42

    # Windows path file:line:func
    f2 = StackFrame.from_string(r"C:\Users\sam\app.py:42:main")
    assert f2.name == "main"
    assert f2.file == r"C:\Users\sam\app.py"
    assert f2.line == 42

    # Standard Linux path file:func:line
    f3 = StackFrame.from_string("src/main.py:compute:123")
    assert f3.name == "compute"
    assert f3.file == "src/main.py"
    assert f3.line == 123

    # Parentheses with Windows drive
    f4 = StackFrame.from_string(r"main (C:\app.py:50)")
    assert f4.name == "main"
    assert f4.file == r"C:\app.py"
    assert f4.line == 50

    # Parentheses without line
    f5 = StackFrame.from_string("main (app.py)")
    assert f5.name == "main"
    assert f5.file == "app.py"
    assert f5.line is None

    # Plain name
    f6 = StackFrame.from_string("my_simple_func")
    assert f6.name == "my_simple_func"
    assert f6.file is None


def test_parser_negative_count_and_utf8_bom(tmp_path: Path):
    """Verify negative sample count in strict mode and UTF-8 BOM stripping."""
    # Negative count in strict mode
    with pytest.raises(ProfileParseError) as exc_info:
        parse_line("main;worker -10", strict=True)
    assert "Negative sample count -10 is invalid" in str(exc_info.value)

    # Negative count in non-strict mode returns None
    assert parse_line("main;worker -10", strict=False) is None

    # UTF-8 BOM file
    bom_file = tmp_path / "bom.folded"
    bom_content = "\ufeffmain;handler 100\nmain;worker 200\n"
    bom_file.write_text(bom_content, encoding="utf-8")

    prof = parse_collapsed(bom_file, strict=True)
    assert ("main", "handler") in prof.stacks
    # Ensure BOM character was stripped from first frame
    assert not any("\ufeff" in frame for stack in prof.stacks for frame in stack)


def test_configuration_validation_errors():
    """Verify ConfigurationError is raised on invalid configurations."""
    p = Profile("p", {("a",): 100})
    base = ProfileSet("base", [p, p, p])
    cand = ProfileSet("cand", [p, p, p])

    with pytest.raises(ConfigurationError):
        compare(base, cand, method="nonexistent_method")

    with pytest.raises(ConfigurationError):
        compare(base, cand, metric="nonexistent_metric")

    with pytest.raises(ConfigurationError):
        compare(base, cand, alpha=1.5)

    with pytest.raises(ConfigurationError):
        compare(base, cand, alpha=-0.1)

    with pytest.raises(ConfigurationError):
        compare(base, cand, fdr_q=0.0)

    with pytest.raises(ConfigurationError):
        compare(base, cand, ci_level=1.0)

    with pytest.raises(ConfigurationError):
        compare(base, cand, permutations=0)

    with pytest.raises(ConfigurationError):
        compare(base, cand, sampling_rate_hz=-1.0)

    # Calibrator validations
    with pytest.raises(ConfigurationError):
        calibrate(base, alpha=-0.05)

    with pytest.raises(ConfigurationError):
        calibrate(base, num_splits=0)


def test_normal_ppf_and_dynamic_z_factor():
    """Verify normal_ppf implementation accuracy and calibrator dynamic z_factor calculation."""
    # Symmetry around 0.5
    assert abs(normal_ppf(0.5)) < 1e-12
    # Known quantiles
    assert abs(normal_ppf(0.975) - 1.9599639845) < 1e-6
    assert abs(normal_ppf(0.025) - (-1.9599639845)) < 1e-6
    assert abs(normal_ppf(0.80) - 0.8416212335) < 1e-6

    # Out of bounds
    with pytest.raises(ValueError):
        normal_ppf(0.0)
    with pytest.raises(ValueError):
        normal_ppf(1.0)
    with pytest.raises(ValueError):
        normal_ppf(-0.5)

    # Verify calibration with different alphas
    p = Profile("p", {("a",): 100})
    runs = ProfileSet("runs", [p for _ in range(6)])

    cal_05 = calibrate(runs, alpha=0.05, num_splits=10)
    cal_01 = calibrate(runs, alpha=0.01, num_splits=10)
    assert cal_05.alpha == 0.05
    assert cal_01.alpha == 0.01


def test_markdown_escaping_pipe_characters():
    """Verify markdown table escaping when function or path contains pipe characters."""
    func_with_pipe = FunctionDiff(
        function="operator|",
        baseline_mean=0.1,
        candidate_mean=0.2,
        baseline_std=0.01,
        candidate_std=0.01,
        delta=0.1,
        pct_change=100.0,
        p_value=0.001,
        adjusted_p_value=0.001,
        ci_lower=0.08,
        ci_upper=0.12,
        attribution_pct=100.0,
        is_significant=True,
    )
    path_with_pipe = PathDiff(
        path=("std", "operator|"),
        path_str="std;operator|",
        baseline_mean=0.0,
        candidate_mean=0.1,
        delta=0.1,
        pct_change=100.0,
        p_value=0.0,
        adjusted_p_value=0.0,
        is_new=True,
        is_removed=False,
    )
    res = ComparisonResult(
        verdict=Verdict.FAIL,
        baseline_runs=3,
        candidate_runs=3,
        baseline_mean_duration=0.5,
        candidate_mean_duration=0.6,
        baseline_std_duration=0.01,
        candidate_std_duration=0.01,
        baseline_cv_pct=2.0,
        candidate_cv_pct=1.7,
        total_delta=0.1,
        total_pct_change=20.0,
        total_p_value=0.001,
        ci_lower=15.0,
        ci_upper=25.0,
        ci_level=0.95,
        threshold_pct=2.0,
        alpha=0.05,
        fdr_q=0.05,
        method="permutation",
        metric="self",
        noise_floor_pct=1.5,
        regressed_functions=[func_with_pipe],
        new_paths=[path_with_pipe],
    )
    md = format_markdown(res)
    # Check that pipe is escaped inside table cells
    assert r"operator\|" in md


def test_sampler_lifecycle_robustness():
    """Verify Sampler handles multiple start calls, clean stopping, and wall clock timing."""
    sampler = Sampler(frequency_hz=100.0)

    # to_profile before starting has zero wall duration
    p0 = sampler.to_profile()
    assert p0.metadata["wall_duration_s"] == 0.0

    # Stop before start is harmless
    sampler.stop()

    # Start and start again (idempotent, no second worker thread)
    sampler.start()
    w1 = sampler._worker
    sampler.start()
    w2 = sampler._worker
    assert w1 is w2

    sampler.stop()
    p1 = sampler.to_profile()
    assert p1.metadata["wall_duration_s"] > 0.0


def test_recorder_builtin_python_script_execution(tmp_path: Path):
    """Verify record_command with builtin engine properly profiles Python script functions."""
    script_file = tmp_path / "workload.py"
    script_file.write_text(
        "import time\n"
        "def custom_worker():\n"
        "    time.sleep(0.05)\n"
        "custom_worker()\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "prof_out"
    ps = record_command(
        [sys.executable, str(script_file)],
        runs=2,
        out_dir=out_dir,
        frequency_hz=100.0,
        warmup=0,
        engine="builtin",
    )
    assert ps.num_runs == 2
    # Ensure custom_worker was sampled
    all_funcs = ps.all_functions()
    assert any("custom_worker" in fn for fn in all_funcs)
