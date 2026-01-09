"""
Progress callback interface for RAPAS pipeline operations.

This module provides a unified interface for reporting progress from
pipeline functions, supporting both Streamlit UI and Celery background tasks.

Usage in pipeline functions:
    from src.progress import ProgressReporter, get_default_reporter

    def my_function(..., progress: ProgressReporter = None):
        progress = progress or get_default_reporter()
        progress.info("Starting processing...")
        progress.update(0.5, "Halfway done")
        progress.success("Complete!")

The ProgressReporter is a simple callable protocol that can be:
- StreamlitReporter: Uses st.write/st.progress for UI display
- CeleryReporter: Stores events in database for polling
- NullReporter: Discards all messages (for testing/silent mode)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Protocol, Union


class MessageLevel(str, Enum):
    """Log levels for progress messages."""

    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ProgressUpdate:
    """A single progress update."""

    progress: Optional[float]  # 0.0 to 1.0, None for indeterminate
    message: str
    level: MessageLevel = MessageLevel.INFO


class ProgressReporter(ABC):
    """
    Abstract base class for progress reporting.

    Implementations should handle progress updates appropriately
    for their target (Streamlit UI, Celery job events, etc.).
    """

    @abstractmethod
    def update(self, progress: Optional[float], message: str) -> None:
        """
        Report a progress update.

        Args:
            progress: Progress value from 0.0 to 1.0, or None for indeterminate
            message: Human-readable progress message
        """
        pass

    @abstractmethod
    def info(self, message: str) -> None:
        """Report an informational message."""
        pass

    @abstractmethod
    def success(self, message: str) -> None:
        """Report a success message."""
        pass

    @abstractmethod
    def warning(self, message: str) -> None:
        """Report a warning message."""
        pass

    @abstractmethod
    def error(self, message: str) -> None:
        """Report an error message."""
        pass

    @abstractmethod
    def write(self, message: str) -> None:
        """Write a plain message (like st.write)."""
        pass


class StreamlitReporter(ProgressReporter):
    """
    Progress reporter that uses Streamlit for display.

    This is the default reporter when running in Streamlit context.
    """

    def __init__(self):
        """Initialize with lazy Streamlit import."""
        self._st = None

    @property
    def st(self):
        """Lazy import of streamlit to avoid import errors in non-UI context."""
        if self._st is None:
            import streamlit as st

            self._st = st
        return self._st

    def update(self, progress: Optional[float], message: str) -> None:
        """Update progress bar and display message."""
        if progress is not None:
            # Streamlit progress expects 0-100 or 0.0-1.0
            self.st.progress(progress, text=message)
        else:
            self.st.info(message)

    def info(self, message: str) -> None:
        self.st.info(message)

    def success(self, message: str) -> None:
        self.st.success(message)

    def warning(self, message: str) -> None:
        self.st.warning(message)

    def error(self, message: str) -> None:
        self.st.error(message)

    def write(self, message: str) -> None:
        self.st.write(message)


class NullReporter(ProgressReporter):
    """
    Progress reporter that discards all messages.

    Useful for testing or when no progress reporting is needed.
    """

    def update(self, progress: Optional[float], message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def success(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass

    def write(self, message: str) -> None:
        pass


class LoggingReporter(ProgressReporter):
    """
    Progress reporter that uses Python logging.

    Useful for background tasks or CLI usage.
    """

    def __init__(self, logger_name: str = "rapas.pipeline"):
        import logging

        self._logger = logging.getLogger(logger_name)

    def update(self, progress: Optional[float], message: str) -> None:
        if progress is not None:
            self._logger.info(f"[{progress * 100:.0f}%] {message}")
        else:
            self._logger.info(message)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def success(self, message: str) -> None:
        self._logger.info(f"✓ {message}")

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    def write(self, message: str) -> None:
        self._logger.info(message)


class CallbackReporter(ProgressReporter):
    """
    Progress reporter that calls a callback function.

    Used by Celery tasks to report progress via job events.
    """

    def __init__(self, callback: Callable[[float, str, str], None]):
        """
        Initialize with a callback function.

        Args:
            callback: Function taking (progress, message, level) arguments
        """
        self._callback = callback

    def update(self, progress: Optional[float], message: str) -> None:
        self._callback(progress or 0.0, message, MessageLevel.INFO.value)

    def info(self, message: str) -> None:
        self._callback(0.0, message, MessageLevel.INFO.value)

    def success(self, message: str) -> None:
        self._callback(1.0, message, MessageLevel.SUCCESS.value)

    def warning(self, message: str) -> None:
        self._callback(0.0, message, MessageLevel.WARNING.value)

    def error(self, message: str) -> None:
        self._callback(0.0, message, MessageLevel.ERROR.value)

    def write(self, message: str) -> None:
        self._callback(0.0, message, MessageLevel.INFO.value)


# Global default reporter - can be overridden
_default_reporter: Optional[ProgressReporter] = None


def set_default_reporter(reporter: ProgressReporter) -> None:
    """Set the default progress reporter globally."""
    global _default_reporter
    _default_reporter = reporter


def get_default_reporter() -> ProgressReporter:
    """
    Get the default progress reporter.

    Returns StreamlitReporter if running in Streamlit context,
    otherwise returns LoggingReporter.
    """
    global _default_reporter

    if _default_reporter is not None:
        return _default_reporter

    # Try to detect Streamlit context
    try:
        import streamlit as st

        # Check if we're actually running in Streamlit
        # This will fail if not in Streamlit context
        ctx = st.runtime.scriptrunner.get_script_run_ctx()
        if ctx is not None:
            return StreamlitReporter()
    except Exception:
        pass

    # Fall back to logging reporter
    return LoggingReporter()


def is_streamlit_context() -> bool:
    """Check if code is running within Streamlit."""
    try:
        import streamlit as st

        ctx = st.runtime.scriptrunner.get_script_run_ctx()
        return ctx is not None
    except Exception:
        return False
