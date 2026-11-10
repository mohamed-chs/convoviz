"""Defensive readers for untyped export JSON, shared by the provider adapters.

Exports have no published schema, so every value is checked before use: a wrong
type reads as missing, never as an exception.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from convoviz.model import AttachmentKind


def as_str(value: Any) -> str | None:
    """The value if it is a string."""
    return value if isinstance(value, str) else None


def as_dict(value: Any) -> dict[str, Any]:
    """The value if it is an object, else an empty one."""
    return value if isinstance(value, dict) else {}


def as_time(value: Any) -> datetime | None:
    """Parse epoch seconds, epoch milliseconds, or ISO-8601 strings."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        seconds = float(value)
        if seconds > 1e11:  # milliseconds
            seconds /= 1000
        if seconds <= 0:
            return None
        try:
            return datetime.fromtimestamp(seconds, UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            try:
                return as_time(float(value))
            except ValueError:
                return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def kind_for(mime: str | None, name: str | None) -> AttachmentKind:
    """Attachment kind from a MIME type or, failing that, a file name."""
    probe = (mime or "").lower()
    lowered = (name or "").lower()
    if probe.startswith("image/") or lowered.endswith(
        (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".heic")
    ):
        return "image"
    if probe.startswith("audio/") or lowered.endswith((".wav", ".mp3", ".m4a", ".ogg")):
        return "audio"
    if probe.startswith("video/") or lowered.endswith((".mp4", ".mov", ".webm")):
        return "video"
    return "file"
