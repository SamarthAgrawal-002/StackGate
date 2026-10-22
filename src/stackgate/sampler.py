"""
Pure-Python, thread-safe sampling profiler for runtime execution capture.
Zero external dependencies, works across all Python versions (3.10+).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any, Callable, TypeVar

from stackgate.models import Profile

T = TypeVar("T")


def _extract_stack(frame: FrameType | None, target_thread_ident: int) -> tuple[str, ...] | None:
    """Extract collapsed call path from a Python execution frame."""
    if frame is None:
        return None

    frames: list[str] = []
    curr: FrameType | None = frame
    while curr is not None:
        code = curr.f_code
        fn_name = code.co_name
        filename = Path(code.co_filename).name
        # Format as filename:function_name
        frames.append(f"{filename}:{fn_name}")
        curr = curr.f_back

    # Root to leaf
    frames.reverse()
    return tuple(frames) if frames else None


class Sampler:
    """
    Lightweight sampling profiler that inspects Python threads at regular intervals.
    """

    def __init__(
        self,
        frequency_hz: float = 100.0,
        target_thread: threading.Thread | None = None,
        name: str = "sampled_run",
    ) -> None:
        self.frequency_hz = max(1.0, float(frequency_hz))
        self.interval = 1.0 / self.frequency_hz
        self.target_thread = target_thread or threading.main_thread()
        self.name = name
        self.stacks: dict[tuple[str, ...], int] = {}
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start background sampling thread."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return  # Already running
            self._stop_event.clear()
            self.stacks.clear()
            self._start_time = time.perf_counter()
            self._end_time = 0.0
            self._worker = threading.Thread(
                target=self._sampling_loop,
                name="StackGate-Sampler",
                daemon=True,
            )
            self._worker.start()

    def stop(self) -> None:
        """Stop sampling and wait for sampler thread to terminate."""
        self._stop_event.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        if self._start_time > 0.0:
            self._end_time = time.perf_counter()
        else:
            self._end_time = 0.0

    def __enter__(self) -> Sampler:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def _sampling_loop(self) -> None:
        interval = self.interval
        stop_event = self._stop_event
        next_tick = time.perf_counter() + interval

        while not stop_event.is_set():
            now = time.perf_counter()
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_tick = max(time.perf_counter() + interval, next_tick + interval)

            target_ident = self.target_thread.ident
            if target_ident is None:
                continue

            frames = sys._current_frames()
            if target_ident in frames:
                frame = frames[target_ident]
                stack = _extract_stack(frame, target_ident)
                if stack:
                    with self._lock:
                        self.stacks[stack] = self.stacks.get(stack, 0) + 1

    def to_profile(self, name: str | None = None) -> Profile:
        """Export accumulated samples as a Profile object."""
        with self._lock:
            copied_stacks = dict(self.stacks)

        wall_duration = (
            max(0.0001, self._end_time - self._start_time)
            if (self._end_time > 0.0 and self._start_time > 0.0)
            else 0.0
        )
        metadata = {
            "wall_duration_s": wall_duration,
            "target_thread": self.target_thread.name,
            "frequency_hz": self.frequency_hz,
        }

        return Profile(
            name=name or self.name,
            stacks=copied_stacks,
            sampling_rate_hz=self.frequency_hz,
            metadata=metadata,
        )


def sample_callable(
    target: Callable[..., T],
    *args: Any,
    frequency_hz: float = 200.0,
    name: str = "run",
    **kwargs: Any,
) -> tuple[T, Profile]:
    """
    Execute a callable under the sampling profiler and return (result, profile).
    """
    sampler = Sampler(frequency_hz=frequency_hz, name=name)
    with sampler:
        result = target(*args, **kwargs)
    return result, sampler.to_profile()
