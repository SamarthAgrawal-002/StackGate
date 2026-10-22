"""
Tests for A/A calibration and empirical noise floor analysis.
"""

import pytest
from stackgate.calibrator import calibrate
from stackgate.exceptions import InsufficientRunsError
from stackgate.models import Profile, ProfileSet


def test_calibrator_runs():
    # 10 runs with small random variations
    profiles = [
        Profile(name=f"run_{i}", stacks={("main", "compute"): 1000 + (i * 5)}, sampling_rate_hz=1000.0)
        for i in range(10)
    ]
    ps = ProfileSet(name="base_runs", profiles=profiles)

    calib = calibrate(ps, alpha=0.05, num_splits=50, seed=42)
    assert calib.runs_evaluated == 10
    assert calib.splits_performed == 50
    assert 0.0 <= calib.empirical_fpr <= 0.20
    assert calib.noise_floor_pct >= 0.0
    assert calib.cv_pct > 0.0
    assert "5.0%" in calib.recommended_k
    assert calib.recommended_k["5.0%"] >= 3

    # Check summary and dict
    summary_text = calib.summary()
    assert "A/A Calibration" in summary_text
    assert "noise floor" in summary_text

    d = calib.to_dict()
    assert d["runs_evaluated"] == 10
    assert d["splits_performed"] == 50


def test_calibrator_insufficient_runs():
    # Fewer than 4 runs must raise InsufficientRunsError
    profiles = [
        Profile(name=f"run_{i}", stacks={("main", "task"): 100})
        for i in range(3)
    ]
    ps = ProfileSet(name="too_few", profiles=profiles)

    with pytest.raises(InsufficientRunsError):
        calibrate(ps)
