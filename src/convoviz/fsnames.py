"""Cross-platform safe file names that keep the title readable in any script.

Only characters that are actually unsafe are removed: path separators, Windows
reserved characters, characters that break Obsidian/Markdown links, control and
invisible formatting characters. Cyrillic, CJK, Arabic, accents and emoji stay.
"""

from __future__ import annotations

import re
import unicodedata

# Windows-reserved + characters that break wiki links / Markdown links / tags.
_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\[\]#^\x00-\x1f\x7f]')
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in [*range(1, 10), "¹", "²", "³"]}
    | {f"LPT{i}" for i in [*range(1, 10), "¹", "²", "³"]}
)
MAX_STEM_BYTES = 120  # leaves room for folders inside Windows' 260-char MAX_PATH


def safe_stem(title: str, *, fallback: str = "Untitled", max_bytes: int = MAX_STEM_BYTES) -> str:
    """Turn an arbitrary title into a safe file stem (no extension)."""
    text = unicodedata.normalize("NFC", title)
    # Drop invisible format characters (bidi controls, ZWJ, variation selectors...).
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("Cf", "Cc", "Cs", "Co"))
    text = _UNSAFE_RE.sub(" ", text)
    text = " ".join(text.split())
    text = text.strip(" .")
    text = _truncate_bytes(text, max_bytes).rstrip(" .")
    if not text:
        return fallback
    if text.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        text = f"{text}_"
    return text


def _truncate_bytes(text: str, max_bytes: int) -> str:
    """Truncate to ``max_bytes`` of UTF-8 without splitting a character; prefer word breaks."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    cut = encoded[:max_bytes].decode("utf-8", errors="ignore")
    space = cut.rfind(" ")
    if space > len(cut) * 0.6:
        cut = cut[:space]
    return cut.rstrip() + "…"


_SLUG_RE = re.compile(r"[^\w.-]+")


def slug(text: str, *, max_len: int = 60) -> str:
    """A compact, space-free name for attachments (Unicode letters kept)."""
    stem = safe_stem(text, fallback="", max_bytes=max_len * 4)
    stem = _SLUG_RE.sub("-", stem).strip("-.")
    return stem[:max_len].rstrip("-.")
