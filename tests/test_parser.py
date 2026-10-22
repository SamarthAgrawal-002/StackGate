"""
Tests for profile and collapsed stack parsing.
"""

import io
from pathlib import Path
import pytest
from stackgate.exceptions import InvalidDataError, ProfileParseError
from stackgate.models import Profile, ProfileSet
from stackgate.parser import load_runs, parse_collapsed, parse_directory, parse_line


def test_parse_line_valid():
    # Standard format
    parsed = parse_line("a;b;c 100")
    assert parsed == (("a", "b", "c"), 100)

    # Multiple spaces & tabs
    parsed = parse_line("main;\tworker;parse \t  250")
    assert parsed == (("main", "worker", "parse"), 250)

    # Floating point count rounded
    parsed = parse_line("a;b 42.7")
    assert parsed == (("a", "b"), 43)

    # With units
    parsed = parse_line("a;b 120 samples")
    assert parsed == (("a", "b"), 120)
    parsed = parse_line("a;b 15 ms")
    assert parsed == (("a", "b"), 15)


def test_parse_line_comments_and_blank():
    assert parse_line("") is None
    assert parse_line("   \n") is None
    assert parse_line("# This is a comment") is None
    assert parse_line("// This is a C-style comment") is None


def test_parse_line_malformed_strict():
    # Missing count
    with pytest.raises(ProfileParseError) as exc_info:
        parse_line("a;b;c", strict=True)
    assert "missing valid sample count" in str(exc_info.value).lower()

    # Negative count
    with pytest.raises(ProfileParseError):
        parse_line("a;b -5", strict=True)

    # No stack path
    with pytest.raises(ProfileParseError):
        parse_line("   100", strict=True)

    # Empty frames
    with pytest.raises(ProfileParseError):
        parse_line(";;; 100", strict=True)


def test_parse_line_malformed_non_strict():
    # In non-strict mode, returns None without raising
    assert parse_line("malformed_line_without_count", strict=False) is None
    assert parse_line("a;b -10", strict=False) is None


def test_parse_collapsed_string():
    content = """
    # Profile run 1
    main;handle_req;parse_json 100
    main;handle_req;validate 50
    main;worker;db_query 200
    """
    prof = parse_collapsed(content, name="test_run", sampling_rate_hz=500.0)
    assert prof.name == "test_run"
    assert prof.total_samples == 350
    assert prof.total_duration == 0.7  # 350 / 500
    assert prof.self_time_samples["db_query"] == 200
    assert prof.inclusive_time_samples["main"] == 350


def test_parse_collapsed_stream():
    stream = io.StringIO("alpha;beta 10\nalpha;gamma 20\n")
    prof = parse_collapsed(stream, name="stream_prof")
    assert prof.name == "stream_prof"
    assert prof.total_samples == 30


def test_parse_collapsed_file(tmp_path: Path):
    prof_file = tmp_path / "sample.folded"
    prof_file.write_text("fn1;fn2 40\nfn1;fn3 60\n", encoding="utf-8")

    prof = parse_collapsed(prof_file)
    assert prof.name == "sample"
    assert prof.total_samples == 100


def test_parse_collapsed_json(tmp_path: Path):
    json_file = tmp_path / "profile.json"
    data = {
        "name": "json_prof",
        "sampling_rate_hz": 1000.0,
        "stacks": {
            "a;b": 30,
            "a;c": 70,
        },
    }
    import json
    json_file.write_text(json.dumps(data), encoding="utf-8")

    prof = parse_collapsed(json_file)
    assert prof.name == "json_prof"
    assert prof.total_samples == 100
    assert prof.self_time_samples["b"] == 30


def test_parse_directory(tmp_path: Path):
    dir_path = tmp_path / "runs"
    dir_path.mkdir()

    (dir_path / "run_01.folded").write_text("main;task 100\n", encoding="utf-8")
    (dir_path / "run_02.folded").write_text("main;task 105\n", encoding="utf-8")
    (dir_path / "run_03.folded").write_text("main;task 95\n", encoding="utf-8")

    ps = parse_directory(dir_path)
    assert ps.num_runs == 3
    assert len(ps.profiles) == 3
    assert ps.profiles[0].name == "run_01"
    assert ps.profiles[1].name == "run_02"
    assert ps.profiles[2].name == "run_03"


def test_parse_directory_errors(tmp_path: Path):
    # Non-existent
    with pytest.raises(InvalidDataError):
        parse_directory(tmp_path / "does_not_exist")

    # Directory with no profile files
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(InvalidDataError):
        parse_directory(empty_dir)


def test_load_runs_variants(tmp_path: Path):
    # 1. From existing ProfileSet
    p1 = Profile(name="p1", stacks={("a",): 10})
    ps1 = ProfileSet(name="ps", profiles=[p1])
    assert load_runs(ps1) is ps1

    # 2. From single Profile
    ps2 = load_runs(p1, name="single")
    assert ps2.num_runs == 1

    # 3. From list of profiles
    p2 = Profile(name="p2", stacks={("a",): 20})
    ps3 = load_runs([p1, p2])
    assert ps3.num_runs == 2

    # 4. From multiline string
    ps4 = load_runs("func1;func2 50\nfunc1;func3 50\n")
    assert ps4.num_runs == 1
    assert ps4.profiles[0].total_samples == 100

    # 5. From list of file paths
    f1 = tmp_path / "f1.folded"
    f2 = tmp_path / "f2.folded"
    f1.write_text("root;leaf 10\n", encoding="utf-8")
    f2.write_text("root;leaf 20\n", encoding="utf-8")
    ps5 = load_runs([f1, f2])
    assert ps5.num_runs == 2

    # 6. Invalid type
    with pytest.raises(InvalidDataError):
        load_runs(12345)  # type: ignore


def test_parse_collapsed_additional_branches(tmp_path: Path):
    # Non-existent file path string
    with pytest.raises(ProfileParseError):
        parse_collapsed("non_existent_folder/file.folded")

    # Unsupported source type
    with pytest.raises(ProfileParseError):
        parse_collapsed(42)  # type: ignore

    # Empty content in strict mode
    with pytest.raises(ProfileParseError):
        parse_collapsed("", strict=True)

    # Empty content non-strict returns profile with 0 samples
    prof = parse_collapsed("", strict=False)
    assert prof.total_samples == 0

    # parse_directory on a file instead of directory raises InvalidDataError
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("hello", encoding="utf-8")
    with pytest.raises(InvalidDataError):
        parse_directory(dummy_file)
