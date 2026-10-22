"""
Tests for Sampler and Recorder components.
"""

import time
from pathlib import Path
import pytest
from stackgate.models import Profile
from stackgate.parser import parse_collapsed
from stackgate.recorder import record_callable, write_folded
from stackgate.sampler import Sampler, sample_callable


def _workload():
    """Simple CPU workload to generate frames."""
    total = 0
    for i in range(100000):
        total += (i % 7)
    return total


def test_sampler_basic():
    sampler = Sampler(frequency_hz=200.0, name="test_sampler")
    with sampler:
        _workload()
        time.sleep(0.05)

    prof = sampler.to_profile()
    assert prof.name == "test_sampler"
    assert prof.total_samples > 0
    assert len(prof.stacks) > 0


def test_sample_callable():
    result, prof = sample_callable(_workload, frequency_hz=200.0, name="call_run")
    assert result > 0
    assert prof.name == "call_run"


def test_write_folded_and_readback(tmp_path: Path):
    stacks = {
        ("app", "handler", "parse"): 150,
        ("app", "handler", "validate"): 75,
    }
    p = Profile(name="roundtrip", stacks=stacks, sampling_rate_hz=1000.0)

    dest = tmp_path / "test.folded"
    write_folded(p, dest)
    assert dest.exists()

    # Read back
    reloaded = parse_collapsed(dest)
    assert reloaded.total_samples == 225
    assert reloaded.self_time_samples["parse"] == 150
    assert reloaded.self_time_samples["validate"] == 75


def test_record_callable(tmp_path: Path):
    out_dir = tmp_path / "recorded_profiles"

    ps = record_callable(
        _workload,
        runs=3,
        out_dir=out_dir,
        frequency_hz=200.0,
        warmup=1,
    )

    assert ps.num_runs == 3
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "run_001.folded").exists()
    assert (out_dir / "run_002.folded").exists()
    assert (out_dir / "run_003.folded").exists()
