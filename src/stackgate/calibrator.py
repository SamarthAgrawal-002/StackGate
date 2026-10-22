"""
A/A Calibration and Noise Floor Modeling.
Measures empirical false-positive rate (FPR) and calculates recommended sample size K.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Sequence, Union

from stackgate.exceptions import InsufficientRunsError, InvalidDataError
from stackgate.models import CalibrationResult, Profile, ProfileSet
from stackgate.parser import load_runs
from stackgate.stats import cv, mean, permutation_test, std_dev


def calibrate(
    runs: Union[str, Path, list[str | Path | Profile], ProfileSet, Profile],
    alpha: float = 0.05,
    num_splits: int = 100,
    seed: int | None = 42,
    target_mde_list: Sequence[float] = (2.0, 3.0, 5.0, 10.0),
    sampling_rate_hz: float = 1000.0,
) -> CalibrationResult:
    """
    Run empirical A/A calibration on a collection of baseline profiles.

    Args:
        runs: ProfileSet, directory path, or list of profile runs to calibrate.
        alpha: Nominal significance level (default: 0.05).
        num_splits: Number of random partitions to evaluate (default: 100).
        seed: Random seed for partition reproducibility.
        target_mde_list: List of Minimum Detectable Effect percentages for sample size calculation.
        sampling_rate_hz: Sampling frequency in Hertz.

    Returns:
        CalibrationResult containing empirical FPR, noise floor, and recommended K.
    """
    profile_set = load_runs(runs, name="calibration", sampling_rate_hz=sampling_rate_hz)
    n = profile_set.num_runs

    if n < 4:
        raise InsufficientRunsError(
            f"A/A calibration requires at least 4 runs to split into two groups, but got {n}."
        )

    durs = profile_set.total_durations
    mean_dur = mean(durs)
    std_dur = std_dev(durs)
    cv_pct = cv(durs)

    k_a = n // 2
    k_b = n - k_a

    rng = random.Random(seed)
    false_positives = 0
    pct_diffs: list[float] = []

    indices = list(range(n))

    for _ in range(num_splits):
        rng.shuffle(indices)
        idx_a = indices[:k_a]
        idx_b = indices[k_a:]

        group_a = [durs[i] for i in idx_a]
        group_b = [durs[i] for i in idx_b]

        ma = mean(group_a)
        mb = mean(group_b)

        # Relative delta
        pct_diff = abs(mb - ma) / ma * 100.0 if ma > 0 else 0.0
        pct_diffs.append(pct_diff)

        # Statistical test on pseudo A/B groups
        p_val, _ = permutation_test(
            group_a, group_b, alternative="two-sided", num_permutations=2000, seed=rng.randint(0, 100000)
        )
        if p_val <= alpha:
            false_positives += 1

    empirical_fpr = false_positives / num_splits

    # Noise floor: 95th percentile of observed A/A percentage differences
    pct_diffs.sort()
    pct_95_idx = int(math.floor(0.95 * len(pct_diffs)))
    noise_floor_pct = pct_diffs[min(pct_95_idx, len(pct_diffs) - 1)]

    # Sample size estimation for 80% power at alpha=alpha
    # Standard formula: K = 2 * ((z_alpha/2 + z_beta) * CV / MDE)^2
    # For alpha=0.05 (z=1.96) and beta=0.20 (z=0.8416), factor ~ 2.8016
    z_factor = 2.8016
    recommended_k: dict[str, int] = {}

    for mde in target_mde_list:
        if mde <= 0:
            continue
        # If CV is very small, minimal K is 3
        if cv_pct <= 0:
            recommended_k[f"{mde:.1f}%"] = 3
        else:
            k_calc = 2.0 * ((z_factor * cv_pct) / mde) ** 2
            # Ceil and clamp between 3 and 100
            k_val = max(3, min(100, int(math.ceil(k_calc))))
            recommended_k[f"{mde:.1f}%"] = k_val

    return CalibrationResult(
        runs_evaluated=n,
        splits_performed=num_splits,
        empirical_fpr=empirical_fpr,
        noise_floor_pct=noise_floor_pct,
        mean_duration=mean_dur,
        cv_pct=cv_pct,
        recommended_k=recommended_k,
        alpha=alpha,
        permutations_tested=num_splits * 2000,
    )
