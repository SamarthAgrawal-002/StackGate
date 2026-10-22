"""
Robust profile parsing for collapsed/folded stacks, JSON profiles, and directories.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import IO, Any, Sequence, Union

from stackgate.exceptions import InvalidDataError, ProfileParseError
from stackgate.models import Profile, ProfileSet

logger = logging.getLogger(__name__)

# Matches integer or floating point sample count at end of line, including optional sign
_COUNT_REGEX = re.compile(r"\s+([-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*(?:samples?|s|ms)?\s*$")


def parse_line(
    line: str,
    line_number: int = 1,
    file_path: str | None = None,
    strict: bool = False,
) -> tuple[tuple[str, ...], int] | None:
    """
    Parse a single line of collapsed stack format:
      'func1;func2;func3 100' -> (('func1', 'func2', 'func3'), 100)

    Returns None for empty lines or comments.
    Raises ProfileParseError in strict mode for malformed lines.
    """
    raw = line.strip().lstrip("\ufeff")
    if not raw or raw.startswith(("#", "//")):
        return None

    match = _COUNT_REGEX.search(raw)
    if not match:
        if strict:
            raise ProfileParseError(
                "Line missing valid sample count at end",
                line_number=line_number,
                line_content=line,
                file_path=file_path,
            )
        logger.warning(
            "Skipping malformed profile line %d: missing numeric count in %s",
            line_number,
            file_path or "<string>",
        )
        return None

    count_str = match.group(1)
    try:
        # Round float counts to nearest int if fractional samples exist
        count = int(round(float(count_str)))
    except ValueError:
        if strict:
            raise ProfileParseError(
                f"Invalid sample count value '{count_str}'",
                line_number=line_number,
                line_content=line,
                file_path=file_path,
            )
        return None

    if count < 0:
        if strict:
            raise ProfileParseError(
                f"Negative sample count {count} is invalid",
                line_number=line_number,
                line_content=line,
                file_path=file_path,
            )
        return None

    stack_part = raw[: match.start()].strip()
    if not stack_part:
        if strict:
            raise ProfileParseError(
                "Line has count but empty stack trace",
                line_number=line_number,
                line_content=line,
                file_path=file_path,
            )
        return None

    frames = tuple(f.strip() for f in stack_part.split(";") if f.strip())
    if not frames:
        if strict:
            raise ProfileParseError(
                "Stack path produced zero non-empty frames",
                line_number=line_number,
                line_content=line,
                file_path=file_path,
            )
        return None

    return frames, count


def parse_collapsed(
    source: str | Path | IO[str],
    name: str | None = None,
    sampling_rate_hz: float = 1000.0,
    strict: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Profile:
    """
    Parse a collapsed/folded stack file, string, or open stream into a Profile.

    Args:
        source: File path, open file stream, or string containing collapsed stacks.
        name: Optional name for this profile run.
        sampling_rate_hz: Sampling frequency in Hertz (default: 1000.0 Hz).
        strict: If True, raises ProfileParseError on any malformed line.
        metadata: Optional metadata dictionary.

    Returns:
        Profile object containing parsed stacks and sample counts.
    """
    file_path: str | None = None
    lines: list[str]

    if isinstance(source, Path):
        file_path = str(source)
        if not source.exists():
            raise ProfileParseError(f"File not found: '{source}'", file_path=file_path)
        content = source.read_text(encoding="utf-8-sig", errors="replace")
        lines = content.splitlines()
        if name is None:
            name = source.stem
    elif isinstance(source, str):
        # Check if source is a path on disk
        if "\n" not in source and (os.path.exists(source) or (len(source) < 512 and ("/" in source or "\\" in source))):
            p = Path(source)
            if p.is_file():
                file_path = str(p)
                content = p.read_text(encoding="utf-8-sig", errors="replace")
                lines = content.splitlines()
                if name is None:
                    name = p.stem
            elif not p.exists() and not source.strip().startswith("{") and not ";" in source:
                raise ProfileParseError(f"File not found: '{source}'", file_path=source)
            else:
                lines = source.splitlines()
                if name is None:
                    name = "profile"
        else:
            lines = source.splitlines()
            if name is None:
                name = "profile"
    elif hasattr(source, "readlines"):
        file_path = getattr(source, "name", None)
        lines = source.readlines()
        if name is None:
            name = Path(file_path).stem if file_path else "profile"
    else:
        raise ProfileParseError(f"Unsupported source type: {type(source)}")

    # Check for JSON profile
    first_non_empty = next((l.strip() for l in lines if l.strip()), None)
    if first_non_empty and first_non_empty.startswith("{"):
        raw_text = "\n".join(lines)
        try:
            data = json.loads(raw_text)
            if isinstance(data, dict) and ("stacks" in data or "samples" in data):
                return Profile.from_dict(data)
        except json.JSONDecodeError:
            pass

    stacks: dict[tuple[str, ...], int] = {}
    for idx, line in enumerate(lines, start=1):
        parsed = parse_line(line, line_number=idx, file_path=file_path, strict=strict)
        if parsed is not None:
            path, count = parsed
            stacks[path] = stacks.get(path, 0) + count

    if not stacks:
        if strict:
            raise ProfileParseError(
                "Profile contains no valid stack samples",
                file_path=file_path,
            )
        # In non-strict mode, return empty profile
        logger.warning("Empty profile parsed from %s", file_path or "<content>")

    return Profile(
        name=name or "profile",
        stacks=stacks,
        sampling_rate_hz=sampling_rate_hz,
        metadata=metadata or {},
    )


def parse_directory(
    dir_path: str | Path,
    pattern: str = "*",
    sampling_rate_hz: float = 1000.0,
    strict: bool = False,
    set_name: str | None = None,
) -> ProfileSet:
    """
    Parse all profile runs from a directory into a ProfileSet.

    Args:
        dir_path: Path to directory containing profile files.
        pattern: Glob pattern to filter files (default: '*').
        sampling_rate_hz: Sampling frequency in Hertz.
        strict: If True, raise errors on malformed files.
        set_name: Name for the ProfileSet (defaults to directory name).

    Returns:
        ProfileSet containing all parsed Profile objects.
    """
    path = Path(dir_path)
    if not path.exists():
        raise InvalidDataError(f"Directory does not exist: '{dir_path}'")
    if not path.is_dir():
        raise InvalidDataError(f"Expected directory path, got file: '{dir_path}'")

    valid_extensions = {".folded", ".txt", ".collapsed", ".stacks", ".prof", ".json", ""}
    files = sorted(
        [
            f
            for f in path.glob(pattern)
            if f.is_file()
            and (f.suffix.lower() in valid_extensions or not f.suffix)
            and not f.name.startswith((".", "manifest"))
        ],
        key=lambda p: p.name,
    )

    if not files:
        raise InvalidDataError(
            f"No profile files found in '{dir_path}' matching pattern '{pattern}'"
        )

    profiles: list[Profile] = []
    for f in files:
        try:
            prof = parse_collapsed(f, sampling_rate_hz=sampling_rate_hz, strict=strict)
            if prof.total_samples > 0 or not strict:
                profiles.append(prof)
        except ProfileParseError as e:
            if strict:
                raise
            logger.warning("Skipping invalid profile file %s: %s", f, e)

    if not profiles:
        raise InvalidDataError(
            f"No valid profiles could be loaded from '{dir_path}'"
        )

    return ProfileSet(name=set_name or path.name, profiles=profiles)


def load_runs(
    source: Union[str, Path, list[str | Path | Profile], ProfileSet, Profile],
    name: str = "runs",
    sampling_rate_hz: float = 1000.0,
    strict: bool = False,
) -> ProfileSet:
    """
    Universal loader that converts paths, directories, lists, or Profile objects into a ProfileSet.
    """
    if isinstance(source, ProfileSet):
        return source

    if isinstance(source, Profile):
        return ProfileSet(name=name, profiles=[source])

    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            return parse_directory(p, sampling_rate_hz=sampling_rate_hz, strict=strict, set_name=name)
        if p.is_file():
            prof = parse_collapsed(p, sampling_rate_hz=sampling_rate_hz, strict=strict, name=p.stem)
            return ProfileSet(name=name, profiles=[prof])
        # If it's a multiline string of collapsed stacks
        if "\n" in str(source) or ";" in str(source):
            prof = parse_collapsed(str(source), sampling_rate_hz=sampling_rate_hz, strict=strict, name=name)
            return ProfileSet(name=name, profiles=[prof])
        raise InvalidDataError(f"Source path does not exist: '{source}'")

    if isinstance(source, (list, tuple, Sequence)) and not isinstance(source, (str, bytes)):
        if not source:
            raise InvalidDataError("Cannot create ProfileSet from empty list or sequence")
        profiles: list[Profile] = []
        for idx, item in enumerate(source):
            if isinstance(item, Profile):
                profiles.append(item)
            elif isinstance(item, (str, Path)):
                item_path = Path(item)
                if item_path.is_file():
                    profiles.append(
                        parse_collapsed(
                            item_path,
                            sampling_rate_hz=sampling_rate_hz,
                            strict=strict,
                            name=item_path.stem,
                        )
                    )
                elif item_path.is_dir():
                    dir_set = parse_directory(
                        item_path,
                        sampling_rate_hz=sampling_rate_hz,
                        strict=strict,
                        set_name=f"{name}_{item_path.name}",
                    )
                    profiles.extend(dir_set.profiles)
                else:
                    profiles.append(
                        parse_collapsed(
                            str(item),
                            sampling_rate_hz=sampling_rate_hz,
                            strict=strict,
                            name=f"{name}_{idx+1}",
                        )
                    )
            elif isinstance(item, ProfileSet):
                profiles.extend(item.profiles)
            else:
                raise InvalidDataError(f"Unsupported item type in profile list: {type(item)}")
        return ProfileSet(name=name, profiles=profiles)

    raise InvalidDataError(f"Unsupported profile source type: {type(source)}")
