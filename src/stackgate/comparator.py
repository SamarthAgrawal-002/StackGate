"""
Comparator engine: multi-run profile comparison, noise modeling, attribution, and verdicts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

from stackgate.exceptions import InvalidDataError
from stackgate.models import (
    ComparisonResult,
    FunctionDiff,
    PathDiff,
    Profile,
    ProfileSet,
    Verdict,
)
from stackgate.parser import load_runs
from stackgate.stats import (
    benjamini_hochberg,
    bootstrap_confidence_interval,
    cv,
    mann_whitney_u_test,
    mean,
    permutation_test,
    std_dev,
    welch_ttest,
)

logger = logging.getLogger(__name__)


@dataclass
class ComparisonConfig:
    """Configuration parameters for comparing profile sets."""

    threshold_pct: float = 2.0  # Relative percentage regression threshold
    alpha: float = 0.05  # Significance level for total regression
    fdr_q: float = 0.05  # False discovery rate q-value threshold
    min_effect_ms: float = 5.0  # Minimum function effect in milliseconds
    min_effect_pct: float = 1.0  # Minimum function effect percentage
    noise_floor_pct: float | None = None  # Noise floor override (defaults to 1.5%)
    method: str = "permutation"  # 'permutation', 'welch', or 'mann-whitney'
    metric: str = "self"  # 'self' (exclusive) or 'inclusive' (cumulative)
    ci_level: float = 0.95  # Confidence interval level (e.g. 0.95 for 95% CI)
    permutations: int = 10000  # Number of Monte Carlo permutations
    seed: int | None = 42  # Deterministic seed for permutations & bootstrap
    strict: bool = False  # If True, inconclusive results raise errors
    sampling_rate_hz: float = 1000.0  # Sampling frequency


def _run_stat_test(
    x: Sequence[float],
    y: Sequence[float],
    method: str,
    alternative: str = "greater",
    permutations: int = 10000,
    seed: int | None = 42,
) -> float:
    """Dispatch to selected statistical hypothesis test."""
    method_lower = method.lower()
    if method_lower == "welch":
        p_val, _, _ = welch_ttest(x, y, alternative=alternative)
        return p_val
    elif method_lower == "mann-whitney":
        p_val, _ = mann_whitney_u_test(x, y, alternative=alternative)
        return p_val
    else:  # default permutation
        p_val, _ = permutation_test(
            x, y, alternative=alternative, num_permutations=permutations, seed=seed
        )
        return p_val


def compare(
    baseline: Union[str, Path, list[str | Path | Profile], ProfileSet, Profile],
    candidate: Union[str, Path, list[str | Path | Profile], ProfileSet, Profile],
    threshold: float | None = None,
    alpha: float | None = None,
    fdr_q: float | None = None,
    config: ComparisonConfig | None = None,
    **kwargs: object,
) -> ComparisonResult:
    """
    Compare baseline profiles against candidate profiles with statistical rigor.

    Args:
        baseline: Directory path, list of filepaths, or ProfileSet for baseline runs.
        candidate: Directory path, list of filepaths, or ProfileSet for candidate runs.
        threshold: Regression threshold percentage (e.g. 2.0 for 2%).
        alpha: Statistical significance level (default: 0.05).
        fdr_q: Multiple-comparison FDR q-value threshold (default: 0.05).
        config: Optional pre-built ComparisonConfig.
        **kwargs: Overrides for ComparisonConfig fields.

    Returns:
        ComparisonResult with verdict, attribution, and confidence intervals.
    """
    cfg = config or ComparisonConfig()
    if threshold is not None:
        cfg.threshold_pct = float(threshold)
    if alpha is not None:
        cfg.alpha = float(alpha)
    if fdr_q is not None:
        cfg.fdr_q = float(fdr_q)
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # Ingest baseline and candidate
    base_set = load_runs(
        baseline,
        name="baseline",
        sampling_rate_hz=cfg.sampling_rate_hz,
        strict=cfg.strict,
    )
    cand_set = load_runs(
        candidate,
        name="candidate",
        sampling_rate_hz=cfg.sampling_rate_hz,
        strict=cfg.strict,
    )

    k_runs = base_set.num_runs
    m_runs = cand_set.num_runs

    warnings: list[str] = []
    power_warning: str | None = None

    if k_runs < 3 or m_runs < 3:
        power_warning = (
            f"Low sample count (K={k_runs} baseline, M={m_runs} candidate). "
            f"Recommended minimum is K=5 runs per side to reliably filter noise."
        )
        warnings.append(power_warning)

    base_cv = base_set.cv_duration
    cand_cv = cand_set.cv_duration
    if base_cv > 15.0:
        cv_msg = f"Baseline duration has high run-to-run noise (CV={base_cv:.1f}%)."
        warnings.append(cv_msg)

    # 1. Test Total Run Duration / Samples
    base_durs = base_set.total_durations
    cand_durs = cand_set.total_durations

    base_mean = mean(base_durs)
    cand_mean = mean(cand_durs)
    base_std = std_dev(base_durs)
    cand_std = std_dev(cand_durs)

    total_delta = cand_mean - base_mean
    total_pct_change = (total_delta / base_mean * 100.0) if base_mean > 0 else 0.0

    # Total p-value testing alternative='greater' (regression)
    p_reg = _run_stat_test(
        base_durs,
        cand_durs,
        method=cfg.method,
        alternative="greater",
        permutations=cfg.permutations,
        seed=cfg.seed,
    )

    # Speedup p-value testing alternative='less'
    p_speedup = _run_stat_test(
        base_durs,
        cand_durs,
        method=cfg.method,
        alternative="less",
        permutations=cfg.permutations,
        seed=cfg.seed,
    )

    # Bootstrap Confidence Interval for total delta and percentage change
    delta_ci_low, delta_ci_high, pct_ci_low, pct_ci_high = bootstrap_confidence_interval(
        base_durs,
        cand_durs,
        ci_level=cfg.ci_level,
        seed=cfg.seed,
    )

    # Noise floor determination
    if cfg.noise_floor_pct is not None:
        noise_floor = cfg.noise_floor_pct
    else:
        # Default empirical noise floor based on baseline CV (e.g. 1.96 * CV / sqrt(K))
        noise_floor = max(1.0, (1.96 * base_cv / (k_runs ** 0.5))) if k_runs > 1 else 2.5

    # 2. Function-Level Attribution and Multiple-Comparison Control
    all_funcs = sorted(list(base_set.all_functions() | cand_set.all_functions()))
    raw_p_values: list[float] = []
    func_data: list[dict[str, object]] = []

    min_effect_sec = cfg.min_effect_ms / 1000.0
    neutral_count = 0

    for fn in all_funcs:
        bx = base_set.get_function_durations(fn, metric=cfg.metric)
        cy = cand_set.get_function_durations(fn, metric=cfg.metric)

        # Check if function was sampled under chosen metric
        has_base = any(v > 0 for v in bx)
        has_cand = any(v > 0 for v in cy)
        if not has_base and not has_cand:
            neutral_count += 1
            continue

        bm = mean(bx)
        cm = mean(cy)
        bs = std_dev(bx)
        cs = std_dev(cy)
        d = cm - bm
        pct = (d / bm * 100.0) if bm > 0 else (100.0 if cm > 0 else 0.0)

        # Single function p-value for regression
        fn_p = _run_stat_test(
            bx,
            cy,
            method="welch" if len(bx) >= 4 and len(cy) >= 4 else "permutation",
            alternative="greater",
            permutations=min(cfg.permutations, 2000),
            seed=cfg.seed,
        )

        # Fast analytical CI via standard error (or bootstrap if needed)
        se = ( (bs ** 2 / k_runs + cs ** 2 / m_runs) ** 0.5 ) if (k_runs > 1 and m_runs > 1) else 0.0
        f_d_low = d - 1.96 * se
        f_d_high = d + 1.96 * se

        is_new = (not has_base and has_cand)
        is_removed = (has_base and not has_cand)

        raw_p_values.append(fn_p)
        func_data.append({
            "function": fn,
            "baseline_mean": bm,
            "candidate_mean": cm,
            "baseline_std": bs,
            "candidate_std": cs,
            "delta": d,
            "pct_change": pct,
            "p_value": fn_p,
            "ci_lower": f_d_low,
            "ci_upper": f_d_high,
            "is_new": is_new,
            "is_removed": is_removed,
        })

    # Benjamini-Hochberg FDR Adjustment
    adj_q_values = benjamini_hochberg(raw_p_values)

    regressed_funcs: list[FunctionDiff] = []
    improved_funcs: list[FunctionDiff] = []

    for data, q_val in zip(func_data, adj_q_values):
        fn = str(data["function"])
        d = float(data["delta"])
        pct = float(data["pct_change"])
        bm = float(data["baseline_mean"])
        cm = float(data["candidate_mean"])
        p_val = float(data["p_value"])

        # Attribution: share of total delta
        if total_delta > 0 and d > 0:
            attrib = (d / total_delta) * 100.0
        else:
            attrib = 0.0

        is_sig = (
            q_val <= cfg.fdr_q
            and abs(d) >= min_effect_sec
            and abs(pct) >= cfg.min_effect_pct
        )

        diff = FunctionDiff(
            function=fn,
            baseline_mean=bm,
            candidate_mean=cm,
            baseline_std=float(data["baseline_std"]),
            candidate_std=float(data["candidate_std"]),
            delta=d,
            pct_change=pct,
            p_value=p_val,
            adjusted_p_value=q_val,
            ci_lower=float(data["ci_lower"]),
            ci_upper=float(data["ci_upper"]),
            attribution_pct=attrib,
            is_significant=is_sig,
            metric=cfg.metric,
            is_new=bool(data["is_new"]),
            is_removed=bool(data["is_removed"]),
        )

        if is_sig and d > 0:
            regressed_funcs.append(diff)
        elif is_sig and d < 0:
            improved_funcs.append(diff)
        else:
            neutral_count += 1

    # Sort regressed functions by absolute delta descending
    regressed_funcs.sort(key=lambda item: item.delta, reverse=True)
    improved_funcs.sort(key=lambda item: item.delta)

    # 3. Path-Level Attribution (New & Removed Paths)
    base_paths = base_set.all_paths()
    cand_paths = cand_set.all_paths()

    new_paths: list[PathDiff] = []
    for path in sorted(cand_paths - base_paths):
        cy = cand_set.get_path_durations(path)
        cm = mean(cy)
        if cm >= min_effect_sec:
            new_paths.append(
                PathDiff(
                    path=path,
                    path_str=";".join(path),
                    baseline_mean=0.0,
                    candidate_mean=cm,
                    delta=cm,
                    pct_change=100.0,
                    p_value=0.0,
                    adjusted_p_value=0.0,
                    is_new=True,
                    is_removed=False,
                )
            )

    removed_paths: list[PathDiff] = []
    for path in sorted(base_paths - cand_paths):
        bx = base_set.get_path_durations(path)
        bm = mean(bx)
        if bm >= min_effect_sec:
            removed_paths.append(
                PathDiff(
                    path=path,
                    path_str=";".join(path),
                    baseline_mean=bm,
                    candidate_mean=0.0,
                    delta=-bm,
                    pct_change=-100.0,
                    p_value=0.0,
                    adjusted_p_value=0.0,
                    is_new=False,
                    is_removed=True,
                )
            )

    # 4. Verdict Determination
    is_stat_sig_reg = p_reg <= cfg.alpha
    is_stat_sig_speedup = p_speedup <= cfg.alpha
    exceeds_threshold = total_pct_change >= cfg.threshold_pct
    exceeds_noise = total_pct_change >= noise_floor

    # Check for inconclusive conditions:
    # Wide CI spanning both negative regression threshold and positive regression threshold with small sample size
    is_wide_ci = (pct_ci_low < -cfg.threshold_pct and pct_ci_high > cfg.threshold_pct)

    if (k_runs < 3 or m_runs < 3) and is_wide_ci:
        verdict = Verdict.INCONCLUSIVE
    elif total_pct_change <= -cfg.threshold_pct and is_stat_sig_speedup:
        verdict = Verdict.IMPROVEMENT
    elif (is_stat_sig_reg and exceeds_threshold and exceeds_noise) or (len(regressed_funcs) > 0 and exceeds_threshold and is_stat_sig_reg):
        verdict = Verdict.FAIL
    elif is_wide_ci and base_cv > 20.0:
        verdict = Verdict.INCONCLUSIVE
    else:
        verdict = Verdict.PASS

    return ComparisonResult(
        verdict=verdict,
        baseline_runs=k_runs,
        candidate_runs=m_runs,
        baseline_mean_duration=base_mean,
        candidate_mean_duration=cand_mean,
        baseline_std_duration=base_std,
        candidate_std_duration=cand_std,
        baseline_cv_pct=base_cv,
        candidate_cv_pct=cand_cv,
        total_delta=total_delta,
        total_pct_change=total_pct_change,
        total_p_value=p_reg,
        ci_lower=pct_ci_low,
        ci_upper=pct_ci_high,
        ci_level=cfg.ci_level,
        threshold_pct=cfg.threshold_pct,
        alpha=cfg.alpha,
        fdr_q=cfg.fdr_q,
        method=cfg.method,
        metric=cfg.metric,
        noise_floor_pct=noise_floor,
        regressed_functions=regressed_funcs,
        improved_functions=improved_funcs,
        neutral_functions_count=neutral_count,
        new_paths=new_paths,
        removed_paths=removed_paths,
        power_warning=power_warning,
        warnings=warnings,
    )
