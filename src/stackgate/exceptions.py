"""
Exceptions for StackGate.
"""

from __future__ import annotations


class StackGateError(Exception):
    """Base exception for all errors raised by StackGate."""


class ProfileParseError(StackGateError):
    """Raised when parsing a profile or collapsed stack fails."""

    def __init__(
        self,
        message: str,
        line_number: int | None = None,
        line_content: str | None = None,
        file_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.line_number = line_number
        self.line_content = line_content
        self.file_path = file_path

    def __str__(self) -> str:
        parts = [self.message]
        if self.file_path:
            parts.append(f"in file '{self.file_path}'")
        if self.line_number is not None:
            parts.append(f"at line {self.line_number}")
        if self.line_content:
            snippet = self.line_content.strip()
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            parts.append(f"-> {repr(snippet)}")
        return " ".join(parts)


class InsufficientRunsError(StackGateError):
    """Raised when there are not enough profile runs to perform statistical testing."""


class InvalidDataError(StackGateError):
    """Raised when profile data is empty, inconsistent, or corrupted."""


class RegressionDetectedError(StackGateError):
    """Raised when a statistically significant regression is detected in assertion gates."""

    def __init__(self, message: str, result: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.result = result


class SamplingError(StackGateError):
    """Raised when profiling or recording process execution fails."""


class ConfigurationError(StackGateError):
    """Raised when configuration parameters or CLI flags are invalid."""
