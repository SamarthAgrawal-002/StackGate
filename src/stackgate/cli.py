"""
Command-Line Interface (CLI) for StackGate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from stackgate import __version__
from stackgate.calibrator import calibrate
from stackgate.comparator import ComparisonConfig, compare
from stackgate.exceptions import (
    ConfigurationError,
    InsufficientRunsError,
    InvalidDataError,
    ProfileParseError,
    StackGateError,
)
from stackgate.models import ComparisonResult, Verdict
from stackgate.recorder import record_command


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for stackgate CLI."""
    parser = argparse.ArgumentParser(
        prog="stackgate",
        description="StackGate: Statistically rigorous sampling profile regression detection.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Compare Subcommand
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare baseline profiles against candidate profiles.",
    )
    compare_parser.add_argument("baseline", help="Directory or file path containing baseline profile(s).")
    compare_parser.add_argument("candidate", help="Directory or file path containing candidate profile(s).")
    compare_parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Relative percentage regression threshold (default: 2.0%%).",
    )
    compare_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for total regression test (default: 0.05).",
    )
    compare_parser.add_argument(
        "--fdr-q",
        type=float,
        default=0.05,
        help="Multiple-comparison FDR q-value threshold for function attribution (default: 0.05).",
    )
    compare_parser.add_argument(
        "--method",
        choices=["permutation", "welch", "mann-whitney"],
        default="permutation",
        help="Statistical testing engine (default: permutation).",
    )
    compare_parser.add_argument(
        "--metric",
        choices=["self", "inclusive"],
        default="self",
        help="Metric used for ranking function regressions (default: self).",
    )
    compare_parser.add_argument(
        "--min-effect-pct",
        type=float,
        default=1.0,
        help="Minimum percentage change for a function to be flagged (default: 1.0%%).",
    )
    compare_parser.add_argument(
        "--min-effect-ms",
        type=float,
        default=5.0,
        help="Minimum absolute duration delta in ms for a function to be flagged (default: 5.0ms).",
    )
    compare_parser.add_argument(
        "--noise-floor",
        type=float,
        default=None,
        help="Custom noise floor override percentage.",
    )
    compare_parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence interval level (default: 0.95).",
    )
    compare_parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Number of Monte Carlo permutations (default: 10000).",
    )
    compare_parser.add_argument(
        "--rate",
        type=float,
        default=1000.0,
        help="Sampling rate in Hz if not embedded in profile (default: 1000.0).",
    )
    compare_parser.add_argument(
        "--format",
        choices=["terminal", "markdown", "json", "html"],
        default="terminal",
        help="Report output format (default: terminal).",
    )
    compare_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="File path to write report to (defaults to stdout).",
    )
    compare_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with exit code 2 if comparison is INCONCLUSIVE.",
    )
    compare_parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Do not exit with non-zero code even on regression (informational only).",
    )

    # 2. Calibrate Subcommand
    calib_parser = subparsers.add_parser(
        "calibrate",
        help="Run A/A test to measure noise floor and calculate required sample size.",
    )
    calib_parser.add_argument("runs", help="Directory containing baseline profile runs.")
    calib_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Target significance level (default: 0.05).",
    )
    calib_parser.add_argument(
        "--splits",
        type=int,
        default=100,
        help="Number of random A/A splits to evaluate (default: 100).",
    )
    calib_parser.add_argument(
        "--format",
        choices=["terminal", "json"],
        default="terminal",
        help="Output format (default: terminal).",
    )

    # 3. Record Subcommand
    rec_parser = subparsers.add_parser(
        "record",
        help="Run a command multiple times under a sampling profiler.",
    )
    rec_parser.add_argument("--runs", "-n", type=int, default=5, help="Number of runs to record (default: 5).")
    rec_parser.add_argument("--out", type=str, default="profiles", help="Output directory (default: profiles).")
    rec_parser.add_argument("--rate", type=float, default=100.0, help="Sampling rate in Hz (default: 100.0).")
    rec_parser.add_argument("--warmup", type=int, default=1, help="Warmup runs to discard (default: 1).")
    rec_parser.add_argument(
        "--engine",
        choices=["auto", "builtin", "pyspy", "tachyon"],
        default="auto",
        help="Sampling profiler engine (default: auto).",
    )
    rec_parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to execute (prefix with --).")

    # 4. Report Subcommand
    rep_parser = subparsers.add_parser(
        "report",
        help="Re-render a JSON comparison result into Markdown or HTML.",
    )
    rep_parser.add_argument("json_file", help="Path to input JSON comparison file.")
    rep_parser.add_argument(
        "--format",
        choices=["markdown", "html", "terminal"],
        default="markdown",
        help="Target report format (default: markdown).",
    )
    rep_parser.add_argument("--output", "-o", type=str, default=None, help="Output destination file.")

    return parser


def run_compare(args: argparse.Namespace) -> int:
    """Execute compare command."""
    cfg = ComparisonConfig(
        threshold_pct=args.threshold,
        alpha=args.alpha,
        fdr_q=args.fdr_q,
        min_effect_pct=args.min_effect_pct,
        min_effect_ms=args.min_effect_ms,
        noise_floor_pct=args.noise_floor,
        method=args.method,
        metric=args.metric,
        ci_level=args.ci,
        permutations=args.permutations,
        strict=args.strict,
        sampling_rate_hz=args.rate,
    )

    result = compare(
        baseline=args.baseline,
        candidate=args.candidate,
        config=cfg,
    )

    # Render format
    if args.format == "json":
        output_text = result.to_json(indent=2)
    elif args.format == "markdown":
        output_text = result.to_markdown()
    elif args.format == "html":
        output_text = result.to_html()
    else:
        # terminal
        output_text = result.to_terminal(use_color=sys.stdout.isatty())

    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)

    if args.no_fail:
        return 0

    if result.verdict == Verdict.FAIL:
        return 1
    if result.verdict == Verdict.INCONCLUSIVE and args.strict:
        return 2
    return 0


def run_calibrate(args: argparse.Namespace) -> int:
    """Execute calibrate command."""
    res = calibrate(runs=args.runs, alpha=args.alpha, num_splits=args.splits)
    if args.format == "json":
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(res.summary())
    return 0


def run_record(args: argparse.Namespace) -> int:
    """Execute record command."""
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("Error: No command specified to record. Use: stackgate record [opts] -- <command>", file=sys.stderr)
        return 3

    ps = record_command(
        command=cmd,
        runs=args.runs,
        out_dir=args.out,
        frequency_hz=args.rate,
        warmup=args.warmup,
        engine=args.engine,
    )
    print(f"Successfully recorded {ps.num_runs} runs to '{args.out}'.")
    return 0


def run_report(args: argparse.Namespace) -> int:
    """Execute report command."""
    in_path = Path(args.json_file)
    if not in_path.exists():
        print(f"Error: File not found: '{args.json_file}'", file=sys.stderr)
        return 3

    try:
        data = json.loads(in_path.read_text(encoding="utf-8"))
        res = ComparisonResult.from_dict(data)
    except Exception as e:
        print(f"Error: Invalid or corrupted JSON report '{args.json_file}': {e}", file=sys.stderr)
        return 3

    if args.format == "json":
        out = res.to_json(indent=2)
    elif args.format == "markdown":
        out = res.to_markdown()
    elif args.format == "html":
        out = res.to_html()
    else:
        out = res.to_terminal(use_color=sys.stdout.isatty())

    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
    else:
        print(out)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, (int, bool)) else (0 if e.code is None else 1)

    if not args.command:
        parser.print_help()
        return 0

    try:
        if args.command == "compare":
            return run_compare(args)
        elif args.command == "calibrate":
            return run_calibrate(args)
        elif args.command == "record":
            return run_record(args)
        elif args.command == "report":
            return run_report(args)
        else:
            parser.print_help()
            return 3
    except ProfileParseError as e:
        print(f"Parse Error: {e}", file=sys.stderr)
        return 3
    except (InvalidDataError, InsufficientRunsError, ConfigurationError) as e:
        print(f"Data Error: {e}", file=sys.stderr)
        return 3
    except StackGateError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected Error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
