"""Usage statistics over a set of conversations (provider-neutral).

Everything is computed from the main branch of each conversation, so edited
and regenerated messages are not double counted.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from convoviz.model import (
    AttachmentBlock,
    CodeBlock,
    DocumentBlock,
    ReasoningBlock,
    TextBlock,
    strip_citations,
)
from convoviz.render import count_messages
from convoviz.stopwords import STOPWORDS
from convoviz.tree import linearize

if TYPE_CHECKING:
    from datetime import tzinfo

    from convoviz.model import Conversation, Message

FEATURES = (
    ("reasoning", "Reasoning"),
    ("web", "Web sources cited"),
    ("code", "Code run"),
    ("image_gen", "Images generated"),
    ("upload", "Files or images uploaded"),
    ("canvas", "Canvas"),
    ("voice", "Voice"),
)

LENGTH_BUCKETS = ((1, 2), (3, 4), (5, 10), (11, 20), (21, 50), (51, 100), (101, None))


@dataclass(slots=True)
class Stats:
    """Aggregates for the insights report."""

    conversations: int = 0
    prompts: int = 0
    replies: int = 0
    first: date | None = None
    last: date | None = None
    per_day: Counter[date] = field(default_factory=Counter)
    week_hour: list[list[int]] = field(default_factory=lambda: [[0] * 24 for _ in range(7)])
    per_month: Counter[str] = field(default_factory=Counter)  # "2026-03"
    models_per_month: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    lengths: Counter[str] = field(default_factory=Counter)
    features: Counter[str] = field(default_factory=Counter)
    words_by_year: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    @property
    def active_days(self) -> int:
        """Days with at least one prompt."""
        return len(self.per_day)

    @property
    def longest_streak(self) -> int:
        """Most consecutive days with at least one prompt."""
        best = run = 0
        previous: date | None = None
        for day in sorted(self.per_day):
            run = run + 1 if previous and day - previous == timedelta(days=1) else 1
            best = max(best, run)
            previous = day
        return best

    def months(self) -> list[str]:
        """Every month from the first to the last activity, including empty ones."""
        if self.first is None or self.last is None:
            return []
        out: list[str] = []
        year, month = self.first.year, self.first.month
        while (year, month) <= (self.last.year, self.last.month):
            out.append(f"{year:04d}-{month:02d}")
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return out


def bucket_label(low: int, high: int | None) -> str:
    """Human label of a length bucket."""
    if high is None:
        return f"{low}+"
    return f"{low}–{high}" if low != high else str(low)


def compute(
    conversations: list[Conversation],
    tz: tzinfo | None,
    ignored_words: frozenset[str] = frozenset(),
) -> Stats:
    """Aggregate statistics over ``conversations``."""
    stats = Stats()
    for conversation in conversations:
        thread = linearize(conversation, include_forks=False)
        messages = thread.messages()
        if not messages:
            continue
        stats.conversations += 1
        length = count_messages(thread)
        for low, high in LENGTH_BUCKETS:
            if length >= low and (high is None or length <= high):
                stats.lengths[bucket_label(low, high)] += 1
                break
        used: set[str] = set()
        for message in messages:
            used |= _features(message)
            when = (message.time or conversation.created).astimezone(tz)
            if message.role == "user":
                _count_prompt(stats, message, when, ignored_words)
            elif message.role == "assistant":
                stats.replies += 1
                if message.model:
                    stats.models_per_month[f"{when:%Y-%m}"][message.model] += 1
        stats.features.update(used)
    if stats.per_day:
        stats.first, stats.last = min(stats.per_day), max(stats.per_day)
    return stats


def _count_prompt(stats: Stats, message: Message, when: datetime, ignored: frozenset[str]) -> None:
    stats.prompts += 1
    stats.per_day[when.date()] += 1
    stats.week_hour[when.weekday()][when.hour] += 1
    stats.per_month[f"{when:%Y-%m}"] += 1
    stats.words_by_year[when.year].update(tokenize(message.text, ignored))


def _features(message: Message) -> set[str]:
    found: set[str] = set()
    for block in message.blocks:
        match block:
            case ReasoningBlock():
                found.add("reasoning")
            case TextBlock() if block.sources and message.role != "user":
                found.add("web")
            case CodeBlock():
                found.add("code")
            case DocumentBlock():
                found.add("canvas")
            case AttachmentBlock():
                kind = block.attachment.kind
                if kind == "audio":
                    found.add("voice")
                elif message.role == "user":
                    found.add("upload")
                elif kind == "image":
                    found.add("image_gen")
            case _:
                pass
    return found


# --------------------------------------------------------------------------- words

_WORD_RE = re.compile(r"[^\W\d_](?:[\w'’-]*[^\W_])?")
_CODE_RE = re.compile(r"```.*?(?:```|\Z)|`[^`\n]*`", re.DOTALL)
_URL_RE = re.compile(r"https?://\S+")

# French/Italian elisions: "l'histoire" counts as "histoire".
_ELISION_RE = re.compile(r"\b(?:qu|[cdjlmnst])'(?=\w)")


def tokenize(text: str, ignored: frozenset[str] = frozenset()) -> list[str]:
    """Lower-cased words of prose (code and URLs removed), stop words and ``ignored`` dropped."""
    text = strip_citations(text)
    if "`" in text:
        text = _CODE_RE.sub(" ", text)
    if "://" in text:
        text = _URL_RE.sub(" ", text)
    text = _ELISION_RE.sub("", text.lower().replace("’", "'"))
    return [
        w for w in _WORD_RE.findall(text) if len(w) > 2 and w not in STOPWORDS and w not in ignored
    ]


def frequent_words(stats: Stats, limit: int = 90) -> list[tuple[str, int]]:
    """The most used words over all years (words used only once are left out)."""
    total: Counter[str] = Counter()
    for counts in stats.words_by_year.values():
        total.update(counts)
    return [(word, count) for word, count in total.most_common(limit) if count > 1]


def distinctive_words(stats: Stats, per_year: int = 24) -> dict[int, list[tuple[str, float]]]:
    """Words typical of each year compared with the other years (tf-idf-like).

    With a single year, words are ranked by plain frequency.
    """
    years = sorted(stats.words_by_year)
    document_frequency: Counter[str] = Counter()
    for year in years:
        document_frequency.update(stats.words_by_year[year].keys())
    result: dict[int, list[tuple[str, float]]] = {}
    for year in years:
        counts = stats.words_by_year[year]
        total = sum(counts.values()) or 1
        scored: list[tuple[str, float]] = []
        for word, count in counts.items():
            if count < 3:
                continue
            idf = math.log((1 + len(years)) / (1 + document_frequency[word])) + 1
            scored.append((word, count / total * idf))
        scored.sort(key=lambda item: (-item[1], item[0]))
        result[year] = scored[:per_year]
    return result
