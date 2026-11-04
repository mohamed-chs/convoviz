"""Collects everything noteworthy that happened during a run.

The export format is undocumented and changes without notice, so convoviz never
fails silently: every skipped record, unknown content type, and missing file is
counted here and summarized at the end of a run.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger("convoviz")

_MAX_EXAMPLES = 5


@dataclass(slots=True)
class Diagnostics:
    """Accumulated warnings, errors and counters."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    unknown: Counter[str] = field(default_factory=Counter)
    unknown_examples: dict[str, list[str]] = field(default_factory=dict)
    counters: Counter[str] = field(default_factory=Counter)

    def error(self, message: str) -> None:
        """Something was lost and the user should know."""
        logger.error(message)
        self.errors.append(message)

    def warn(self, message: str) -> None:
        """Something unusual that was handled."""
        logger.warning(message)
        self.warnings.append(message)

    def skip(self, message: str) -> None:
        """A record that was intentionally not imported."""
        logger.info("skipped: %s", message)
        self.skipped.append(message)

    def unknown_kind(self, kind: str, where: str) -> None:
        """A structure convoviz does not recognize (rendered generically)."""
        self.unknown[kind] += 1
        examples = self.unknown_examples.setdefault(kind, [])
        if len(examples) < _MAX_EXAMPLES:
            examples.append(where)
        logger.debug("unknown %s at %s", kind, where)

    def count(self, key: str, n: int = 1) -> None:
        """Increment a named counter."""
        self.counters[key] += n
