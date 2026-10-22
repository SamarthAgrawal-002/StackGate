"""
Report formatters: Terminal (ANSI), GitHub Markdown PR comments, and HTML.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stackgate.models import ComparisonResult


def _ansi(code: str, text: str, enabled: bool) -> str:
    """Apply ANSI styling if enabled."""
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def format_terminal(result: ComparisonResult, use_color: bool = True) -> str:
    """Format ComparisonResult for CLI terminal output."""
    lines: list[str] = []

    # Verdict styling
    v = result.verdict
    if v == "FAIL":
        v_str = _ansi("1;31", "FAIL (Performance Regression Detected)", use_color)
    elif v == "IMPROVEMENT":
        v_str = _ansi("1;32", "PASS (Statistically Significant Speedup)", use_color)
    elif v == "INCONCLUSIVE":
        v_str = _ansi("1;33", "INCONCLUSIVE (High Noise or Low Power)", use_color)
    else:
        v_str = _ansi("1;32", "PASS (No Performance Regression)", use_color)

    lines.append(f"Verdict: {v_str}")

    # Total duration line
    sign = "+" if result.total_pct_change >= 0 else ""
    ci_sign_l = "+" if result.ci_lower >= 0 else ""
    ci_sign_u = "+" if result.ci_upper >= 0 else ""

    p_str = f"p={result.total_p_value:.4f}" if result.total_p_value >= 0.0001 else "p < 0.0001"
    stat_summary = (
        f"{result.baseline_mean_duration:.3f}s -> {result.candidate_mean_duration:.3f}s "
        f"({sign}{result.total_pct_change:.2f}%, {int(result.ci_level*100)}% CI "
        f"{ci_sign_l}{result.ci_lower:.1f}% .. {ci_sign_u}{result.ci_upper:.1f}%, "
        f"{p_str}, K={result.baseline_runs} vs {result.candidate_runs})"
    )
    lines.append(f"Total wall time:   {stat_summary}")
    lines.append(
        f"Runner noise floor: {result.noise_floor_pct:.2f}% | "
        f"Threshold: {result.threshold_pct:.2f}% | "
        f"Baseline CV: {result.baseline_cv_pct:.1f}%"
    )

    if result.warnings:
        lines.append("")
        for w in result.warnings:
            lines.append(_ansi("33", f"Warning: {w}", use_color))

    # Regressed functions
    if result.regressed_functions:
        lines.append("")
        header = f"Where it went ({result.metric} time, FDR q < {result.fdr_q}):"
        lines.append(_ansi("1;37", header, use_color))
        for fn in result.regressed_functions:
            fn_p_str = f"q={fn.adjusted_p_value:.4f}" if fn.adjusted_p_value >= 0.0001 else "q < 0.0001"
            attrib_str = f"({fn.attribution_pct:.1f}% of regression)" if fn.attribution_pct > 0 else ""
            lines.append(
                f"  {fn.function:<30} {fn.baseline_mean:6.3f}s -> {fn.candidate_mean:6.3f}s  "
                f"+{fn.delta:.3f}s  {attrib_str:<22} {fn_p_str}"
            )

    # Improved functions
    if result.improved_functions:
        lines.append("")
        header = f"Functions improved ({result.metric} time, FDR q < {result.fdr_q}):"
        lines.append(_ansi("1;32", header, use_color))
        for fn in result.improved_functions:
            fn_p_str = f"q={fn.adjusted_p_value:.4f}" if fn.adjusted_p_value >= 0.0001 else "q < 0.0001"
            attrib_str = f"({fn.attribution_pct:.1f}% of speedup)" if fn.attribution_pct > 0 else ""
            lines.append(
                f"  {fn.function:<30} {fn.baseline_mean:6.3f}s -> {fn.candidate_mean:6.3f}s  "
                f"{fn.delta:.3f}s  ({fn.pct_change:+.1f}%)  {attrib_str:<22} {fn_p_str}"
            )

    # New paths
    if result.new_paths:
        lines.append("")
        lines.append(_ansi("1;33", f"New call paths ({len(result.new_paths)}):", use_color))
        for p in result.new_paths[:5]:
            lines.append(f"  + {p.path_str:<50} +{p.delta:.3f}s")
        if len(result.new_paths) > 5:
            lines.append(f"  ... and {len(result.new_paths) - 5} more new paths")

    # Removed paths
    if result.removed_paths:
        lines.append("")
        lines.append(_ansi("1;36", f"Removed call paths ({len(result.removed_paths)}):", use_color))
        for p in result.removed_paths[:5]:
            lines.append(f"  - {p.path_str:<50} {p.delta:.3f}s")
        if len(result.removed_paths) > 5:
            lines.append(f"  ... and {len(result.removed_paths) - 5} more removed paths")

    lines.append("")
    lines.append(f"Not significant / below noise floor: {result.neutral_functions_count} functions")

    return "\n".join(lines)


def format_markdown(result: ComparisonResult) -> str:
    """Format ComparisonResult as a GitHub PR comment."""
    v = result.verdict
    if v == "FAIL":
        status_badge = "**FAIL** (Regression Detected)"
    elif v == "IMPROVEMENT":
        status_badge = "**PASS** (Performance Improvement)"
    elif v == "INCONCLUSIVE":
        status_badge = "**INCONCLUSIVE** (High Variance / Low Sample Count)"
    else:
        status_badge = "**PASS** (No Regression)"

    sign = "+" if result.total_pct_change >= 0 else ""
    p_str = f"{result.total_p_value:.4f}" if result.total_p_value >= 0.0001 else "< 0.0001"

    lines = [
        f"## StackGate Performance Gate: {status_badge}",
        "*StackGate Performance Report*",
        "",
        "### Overview",
        "",
        "| Metric | Baseline | Candidate | Change | 95% Confidence Interval | p-value | Verdict |",
        "|---|---|---|---|---|---|---|",
        f"| **Total Duration** | {result.baseline_mean_duration:.3f}s | {result.candidate_mean_duration:.3f}s | "
        f"**{sign}{result.total_pct_change:.2f}%** | [{result.ci_lower:+.1f}%, {result.ci_upper:+.1f}%] | "
        f"`{p_str}` | **{result.verdict}** |",
        "",
    ]

    if result.warnings:
        lines.append("> [!WARNING]")
        for w in result.warnings:
            lines.append(f"> {w}")
        lines.append("")

    if result.regressed_functions:
        lines.append(f"### Regressed Functions ({result.metric} time, FDR q < {result.fdr_q})")
        lines.append("")
        lines.append("| Function | Baseline | Candidate | Delta | Regression Share | FDR q-value |")
        lines.append("|---|---|---|---|---|---|")
        for fn in result.regressed_functions:
            q_str = f"{fn.adjusted_p_value:.4f}" if fn.adjusted_p_value >= 0.0001 else "< 0.0001"
            attrib = f"{fn.attribution_pct:.1f}%" if fn.attribution_pct > 0 else "—"
            esc_fn = fn.function.replace("|", r"\|")
            lines.append(
                f"| `{esc_fn}` | {fn.baseline_mean:.3f}s | {fn.candidate_mean:.3f}s | "
                f"+{fn.delta:.3f}s (+{fn.pct_change:.1f}%) | {attrib} | `{q_str}` |"
            )
        lines.append("")

    if result.improved_functions:
        lines.append(f"### Improved Functions ({result.metric} time)")
        lines.append("")
        lines.append("| Function | Baseline | Candidate | Delta | Speedup Share | FDR q-value |")
        lines.append("|---|---|---|---|---|---|")
        for fn in result.improved_functions:
            q_str = f"{fn.adjusted_p_value:.4f}" if fn.adjusted_p_value >= 0.0001 else "< 0.0001"
            attrib = f"{fn.attribution_pct:.1f}%" if fn.attribution_pct > 0 else "—"
            esc_fn = fn.function.replace("|", r"\|")
            lines.append(
                f"| `{esc_fn}` | {fn.baseline_mean:.3f}s | {fn.candidate_mean:.3f}s | "
                f"{fn.delta:.3f}s ({fn.pct_change:.1f}%) | {attrib} | `{q_str}` |"
            )
        lines.append("")

    if result.new_paths:
        lines.append(f"<details><summary><b>New Call Paths ({len(result.new_paths)})</b></summary>")
        lines.append("")
        lines.append("| Call Path | Duration |")
        lines.append("|---|---|")
        for p in result.new_paths:
            esc_path = p.path_str.replace("|", r"\|")
            lines.append(f"| `{esc_path}` | +{p.delta:.3f}s |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if result.removed_paths:
        lines.append(f"<details><summary><b>Removed Call Paths ({len(result.removed_paths)})</b></summary>")
        lines.append("")
        lines.append("| Call Path | Duration |")
        lines.append("|---|---|")
        for p in result.removed_paths:
            esc_path = p.path_str.replace("|", r"\|")
            lines.append(f"| `{esc_path}` | {p.delta:.3f}s |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("<details><summary><b>Statistical Details & Noise Floor</b></summary>")
    lines.append("")
    lines.append(f"- **Sample Counts**: K={result.baseline_runs} baseline runs vs M={result.candidate_runs} candidate runs")
    lines.append(f"- **Baseline CV**: {result.baseline_cv_pct:.2f}% | **Candidate CV**: {result.candidate_cv_pct:.2f}%")
    lines.append(f"- **Runner Noise Floor**: {result.noise_floor_pct:.2f}%")
    lines.append(f"- **Detection Threshold**: {result.threshold_pct:.2f}%")
    lines.append(f"- **Test Method**: {result.method}")
    lines.append(f"- **Neutral / Below Noise Floor Functions**: {result.neutral_functions_count}")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by [StackGate](https://github.com/samarthagrawal/stackgate) — Noise-Aware Sampling Profile Regression Gate.*")

    return "\n".join(lines)


def format_html(result: ComparisonResult) -> str:
    """Format ComparisonResult as a standalone HTML report."""
    v = html.escape(str(result.verdict))
    color = "#e53e3e" if v == "FAIL" else ("#38a169" if v in ("PASS", "IMPROVEMENT") else "#d69e2e")

    rows = []
    for fn in result.regressed_functions:
        rows.append(
            f"<tr><td><code>{html.escape(fn.function)}</code></td>"
            f"<td>{fn.baseline_mean:.3f}s</td>"
            f"<td>{fn.candidate_mean:.3f}s</td>"
            f"<td class='reg'>+{fn.delta:.3f}s</td>"
            f"<td>{fn.attribution_pct:.1f}%</td>"
            f"<td>{fn.adjusted_p_value:.4f}</td></tr>"
        )
    fn_table = "\n".join(rows) if rows else "<tr><td colspan='6'>No significant regressed functions</td></tr>"

    imp_rows = []
    for fn in result.improved_functions:
        imp_rows.append(
            f"<tr><td><code>{html.escape(fn.function)}</code></td>"
            f"<td>{fn.baseline_mean:.3f}s</td>"
            f"<td>{fn.candidate_mean:.3f}s</td>"
            f"<td class='speedup'>{fn.delta:.3f}s</td>"
            f"<td>{fn.attribution_pct:.1f}%</td>"
            f"<td>{fn.adjusted_p_value:.4f}</td></tr>"
        )
    imp_section = ""
    if imp_rows:
        imp_section = f"""
    <h2>Improved Functions</h2>
    <table>
      <thead>
        <tr><th>Function</th><th>Baseline</th><th>Candidate</th><th>Delta</th><th>Speedup Share</th><th>FDR q</th></tr>
      </thead>
      <tbody>
        {"".join(imp_rows)}
      </tbody>
    </table>"""

    warning_box = ""
    if result.warnings:
        w_items = "".join(f"<li>{html.escape(w)}</li>" for w in result.warnings)
        warning_box = f"""
    <div style="background: #fffaf0; border-left: 4px solid #dd6b20; padding: 1rem; margin-top: 1rem; border-radius: 4px;">
      <strong>Warnings:</strong>
      <ul style="margin: 0.5rem 0 0 1.5rem; padding: 0;">{w_items}</ul>
    </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>StackGate Performance Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 2rem; background: #f7fafc; color: #2d3748; }}
    .card {{ background: #fff; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); max-width: 900px; margin: auto; }}
    .badge {{ display: inline-block; padding: 0.4rem 0.8rem; border-radius: 4px; color: #fff; background: {color}; font-weight: bold; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #e2e8f0; }}
    th {{ background: #edf2f7; }}
    .reg {{ color: #e53e3e; font-weight: bold; }}
    .speedup {{ color: #38a169; font-weight: bold; }}
    code {{ background: #edf2f7; padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.9em; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>StackGate Performance Gate</h1>
    <div class="badge">{v}</div>
    {warning_box}
    <p>Total Wall Time: <strong>{result.baseline_mean_duration:.3f}s &rarr; {result.candidate_mean_duration:.3f}s</strong> ({result.total_pct_change:+.2f}%, p={result.total_p_value:.4f})</p>
    <p>Confidence Interval: [{result.ci_lower:+.1f}%, {result.ci_upper:+.1f}%] | Noise Floor: {result.noise_floor_pct:.2f}%</p>
    <h2>Function Attribution</h2>
    <table>
      <thead>
        <tr><th>Function</th><th>Baseline</th><th>Candidate</th><th>Delta</th><th>Attribution</th><th>FDR q</th></tr>
      </thead>
      <tbody>
        {fn_table}
      </tbody>
    </table>{imp_section}
  </div>
</body>
</html>"""
