"""
Recorder engine: captures multi-run sampling profiles from commands or callables.
Outputs folded stack files and manifest metadata.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from stackgate.exceptions import SamplingError
from stackgate.models import Profile, ProfileSet
from stackgate.sampler import sample_callable

logger = logging.getLogger(__name__)


def write_folded(profile: Profile, dest_path: str | Path) -> Path:
    """Write a Profile object to a Brendan Gregg collapsed/folded text file."""
    path = Path(dest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for stack_tuple, count in sorted(profile.stacks.items()):
        stack_str = ";".join(stack_tuple)
        lines.append(f"{stack_str} {count}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def record_callable(
    target: Callable[[], Any],
    runs: int = 5,
    out_dir: str | Path | None = None,
    frequency_hz: float = 200.0,
    warmup: int = 1,
    name_prefix: str = "run",
) -> ProfileSet:
    """
    Profile a callable multiple times and optionally persist folded profiles.

    Args:
        target: Parameterless function to benchmark.
        runs: Number of benchmark runs to collect (recommended >= 5).
        out_dir: Optional output directory to write .folded files.
        frequency_hz: Sampling frequency in Hertz.
        warmup: Number of warmup runs to discard before recording.
        name_prefix: Prefix for run files.

    Returns:
        ProfileSet containing all measured runs.
    """
    if runs < 1:
        raise SamplingError("Must execute at least 1 run")

    for _ in range(max(0, warmup)):
        target()

    profiles: list[Profile] = []
    out_path = Path(out_dir) if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    for i in range(1, runs + 1):
        run_name = f"{name_prefix}_{i:03d}"
        _, prof = sample_callable(target, frequency_hz=frequency_hz, name=run_name)
        profiles.append(prof)
        if out_path:
            write_folded(prof, out_path / f"{run_name}.folded")

    if out_path:
        manifest = {
            "runs": runs,
            "warmup": warmup,
            "frequency_hz": frequency_hz,
            "engine": "builtin_sampler",
            "timestamp": time.time(),
        }
        (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return ProfileSet(name=out_path.name if out_path else "recorded_runs", profiles=profiles)


def record_command(
    command: Sequence[str] | str,
    runs: int = 5,
    out_dir: str | Path = "profiles",
    frequency_hz: float = 100.0,
    warmup: int = 1,
    engine: str = "auto",
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> ProfileSet:
    """
    Run an external CLI command or Python script multiple times under a sampling profiler.

    Args:
        command: Command string or sequence of arguments to run.
        runs: Number of runs to capture.
        out_dir: Output directory for profile files.
        frequency_hz: Sampling frequency in Hz.
        warmup: Warmup runs to discard.
        engine: Profiler engine ('auto', 'pyspy', 'tachyon', or 'builtin').
        cwd: Working directory.
        env: Environment variables.

    Returns:
        ProfileSet containing the recorded runs.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd_list = [command] if isinstance(command, str) else list(command)

    # Detect engine
    chosen_engine = engine.lower()
    if chosen_engine == "auto":
        if shutil.which("py-spy"):
            chosen_engine = "pyspy"
        else:
            # Check Python 3.15 Tachyon
            try:
                import profiling.sampling  # type: ignore # noqa: F401
                chosen_engine = "tachyon"
            except ImportError:
                chosen_engine = "builtin"

    # Discard warmup
    for _ in range(max(0, warmup)):
        subprocess.run(cmd_list, cwd=cwd, env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    profiles: list[Profile] = []

    for i in range(1, runs + 1):
        run_name = f"run_{i:03d}"
        folded_file = out_path / f"{run_name}.folded"

        if chosen_engine == "pyspy":
            # py-spy record -o file.folded -f folded -r freq -- cmd...
            spy_cmd = [
                "py-spy",
                "record",
                "-o",
                str(folded_file),
                "-f",
                "folded",
                "-r",
                str(int(frequency_hz)),
                "--",
            ] + cmd_list
            res = subprocess.run(spy_cmd, cwd=cwd, env=env, capture_output=True, text=True)
            if res.returncode != 0:
                raise SamplingError(f"py-spy failed with exit code {res.returncode}: {res.stderr}")

        elif chosen_engine == "tachyon":
            # Python 3.15 Tachyon
            bin_file = out_path / f"{run_name}.bin"
            tachyon_cmd = [
                sys.executable,
                "-m",
                "profiling.sampling",
                "record",
                "-o",
                str(bin_file),
                "--frequency",
                str(int(frequency_hz)),
                "--",
            ] + cmd_list
            res = subprocess.run(tachyon_cmd, cwd=cwd, env=env, capture_output=True, text=True)
            if res.returncode != 0:
                raise SamplingError(f"Tachyon failed with exit code {res.returncode}: {res.stderr}")
            # Replay to folded stack
            replay_cmd = [
                sys.executable,
                "-m",
                "profiling.sampling",
                "replay",
                "--folded",
                str(bin_file),
            ]
            replay_res = subprocess.run(replay_cmd, cwd=cwd, capture_output=True, text=True)
            if replay_res.returncode != 0:
                raise SamplingError(f"Tachyon replay failed: {replay_res.stderr}")
            folded_file.write_text(replay_res.stdout, encoding="utf-8")

        else:
            # Built-in wrapper runner via Python subprocess
            # Wraps the target command inside a python runner script that uses Sampler
            sub_env = dict(os.environ) if env is None else dict(env)
            src_dir = str(Path(__file__).resolve().parent.parent)
            current_pp = sub_env.get("PYTHONPATH", "")
            sub_env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{current_pp}" if current_pp else src_dir

            runner_code = (
                "import sys, subprocess\n"
                "from stackgate.sampler import Sampler\n"
                "from stackgate.recorder import write_folded\n"
                f"sampler = Sampler(frequency_hz={frequency_hz}, name='{run_name}')\n"
                f"cmd = {repr(cmd_list)}\n"
                "with sampler:\n"
                "    p = subprocess.run(cmd)\n"
                f"write_folded(sampler.to_profile(), {repr(str(folded_file))})\n"
                "sys.exit(p.returncode)\n"
            )
            res = subprocess.run([sys.executable, "-c", runner_code], cwd=cwd, env=sub_env, capture_output=True, text=True)
            if res.returncode != 0:
                raise SamplingError(f"Command execution failed with code {res.returncode}: {res.stderr}")

        from stackgate.parser import parse_collapsed
        prof = parse_collapsed(folded_file, sampling_rate_hz=frequency_hz, name=run_name)
        profiles.append(prof)

    manifest = {
        "command": cmd_list,
        "runs": runs,
        "warmup": warmup,
        "frequency_hz": frequency_hz,
        "engine": chosen_engine,
        "timestamp": time.time(),
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return ProfileSet(name=out_path.name, profiles=profiles)
