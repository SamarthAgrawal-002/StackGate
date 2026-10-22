"""
Tests for Command-Line Interface (CLI).
"""

import json
from pathlib import Path
import pytest
from stackgate.cli import main


def _create_profiles(dir_path: Path, counts: list[int]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for i, count in enumerate(counts, start=1):
        file = dir_path / f"run_{i:02d}.folded"
        file.write_text(f"main;handler {count}\n", encoding="utf-8")


def test_cli_version(capsys):
    ret = main(["--version"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "stackgate 0.1.1" in captured.out


def test_cli_help(capsys):
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Available subcommands" in captured.out


def test_cli_compare_pass(tmp_path: Path):
    base_dir = tmp_path / "base"
    cand_dir = tmp_path / "cand"
    _create_profiles(base_dir, [100, 102, 98, 101, 99])
    _create_profiles(cand_dir, [100, 101, 99, 102, 98])

    ret = main(["compare", str(base_dir), str(cand_dir), "--threshold", "5.0"])
    assert ret == 0


def test_cli_compare_fail(tmp_path: Path):
    base_dir = tmp_path / "base"
    cand_dir = tmp_path / "cand"
    _create_profiles(base_dir, [100, 102, 98, 101, 99])
    _create_profiles(cand_dir, [140, 142, 138, 141, 139])

    ret = main(["compare", str(base_dir), str(cand_dir), "--threshold", "5.0"])
    assert ret == 1


def test_cli_compare_no_fail(tmp_path: Path):
    base_dir = tmp_path / "base"
    cand_dir = tmp_path / "cand"
    _create_profiles(base_dir, [100, 102, 98, 101, 99])
    _create_profiles(cand_dir, [140, 142, 138, 141, 139])

    # --no-fail should return 0 even when regression detected
    ret = main(["compare", str(base_dir), str(cand_dir), "--threshold", "5.0", "--no-fail"])
    assert ret == 0


def test_cli_compare_output_file_json(tmp_path: Path):
    base_dir = tmp_path / "base"
    cand_dir = tmp_path / "cand"
    _create_profiles(base_dir, [100, 102, 98, 101, 99])
    _create_profiles(cand_dir, [100, 102, 98, 101, 99])

    out_file = tmp_path / "result.json"
    ret = main(["compare", str(base_dir), str(cand_dir), "--format", "json", "-o", str(out_file)])
    assert ret == 0
    assert out_file.exists()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["verdict"] == "PASS"


def test_cli_calibrate(tmp_path: Path, capsys):
    runs_dir = tmp_path / "calib_runs"
    _create_profiles(runs_dir, [100, 102, 99, 101, 98, 100, 101, 99])

    ret = main(["calibrate", str(runs_dir), "--splits", "20", "--format", "json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["runs_evaluated"] == 8
    assert "noise_floor_pct" in data


def test_cli_report(tmp_path: Path, capsys):
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps({"verdict": "PASS", "summary": {"total_pct_change": 0.5}}), encoding="utf-8")

    ret = main(["report", str(json_path), "--format", "markdown"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "StackGate Performance Report" in captured.out


def test_cli_error_invalid_paths():
    ret = main(["compare", "non_existent_base_dir_xyz", "non_existent_cand_dir_xyz"])
    assert ret == 3
