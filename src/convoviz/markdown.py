r"""Code-aware Markdown text utilities.

Most transformations (math delimiters, escaping) must never touch code. These
helpers split text into prose and code segments (fenced blocks and inline code
spans) so a transformation can be applied to prose only.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_FENCE_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
_FENCE_LINE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\n]*)$", re.MULTILINE)


def split_fenced(text: str) -> list[tuple[bool, str]]:
    """Split into (is_code, chunk) by fenced code blocks; unclosed fences run to the end."""
    if "```" not in text and "~~~" not in text:
        return [(False, text)] if text else []
    chunks: list[tuple[bool, str]] = []
    start = 0
    fence: str | None = None
    for match in _FENCE_LINE_RE.finditer(text):
        marker, info = match.group(1), match.group(2).rstrip("\r")
        if fence is None:
            if marker[0] == "`" and "`" in info:
                continue  # not a fence: backticks in a backtick info string
            if match.start() > start:
                chunks.append((False, text[start : match.start()]))
            start, fence = match.start(), marker
        elif marker[0] == fence[0] and len(marker) >= len(fence) and not info.strip():
            end = match.end() + (1 if text.startswith("\n", match.end()) else 0)
            chunks.append((True, text[start:end]))
            start, fence = end, None
    if start < len(text):
        chunks.append((fence is not None, text[start:]))
    return chunks


def close_fences(text: str) -> str:
    """Close a dangling code fence so it cannot swallow the rest of the document."""
    if "```" not in text and "~~~" not in text:
        return text
    chunks = split_fenced(text)
    if not chunks or not chunks[-1][0]:
        return text
    last = chunks[-1][1]
    match = _FENCE_RE.match(last.splitlines()[0])
    fence = match.group(2) if match else "```"
    lines = last.splitlines()
    closing = _FENCE_RE.match(lines[-1]) if len(lines) > 1 else None
    if (
        closing
        and closing.group(2)[0] == fence[0]
        and len(closing.group(2)) >= len(fence)
        and not closing.group(3).strip()
    ):
        return text
    return text.rstrip("\n") + "\n" + fence


_INLINE_CODE_RE = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)", re.DOTALL)


def map_prose(text: str, *transforms: Callable[[str], str]) -> str:
    """Apply ``transforms``, in order, to everything except fenced and inline code.

    The text is split once, so the transforms must not create code (backticks or
    fences) themselves.
    """

    def apply(segment: str) -> str:
        for transform in transforms:
            segment = transform(segment)
        return segment

    out: list[str] = []
    for is_code, chunk in split_fenced(text):
        if is_code:
            out.append(chunk)
            continue
        if "`" not in chunk:
            out.append(apply(chunk))
            continue
        position = 0
        for match in _INLINE_CODE_RE.finditer(chunk):
            out.append(apply(chunk[position : match.start()]))
            out.append(match.group(0))
            position = match.end()
        out.append(apply(chunk[position:]))
    return "".join(out)


_DISPLAY_MATH_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def math_to_dollars(segment: str) -> str:
    """Prose-level part of :func:`convert_math`."""
    if "\\[" in segment:
        segment = _DISPLAY_MATH_RE.sub(lambda m: f"$${m.group(1)}$$", segment)
    if "\\(" in segment:
        segment = _INLINE_MATH_RE.sub(lambda m: f"${m.group(1).strip()}$", segment)
    return segment


def convert_math(text: str) -> str:
    r"""Rewrite ``\[ … \]`` and ``\( … \)`` to ``$$ … $$`` and ``$ … $``.

    Obsidian and GitHub only understand dollar delimiters.
    """
    return map_prose(text, math_to_dollars)


_CURRENCY_RE = re.compile(r"(?<![\\$\w])\$(?=\d)")


def currency_escaped(segment: str) -> str:
    """Prose-level part of :func:`escape_currency`."""
    return _CURRENCY_RE.sub(r"\\$", segment) if "$" in segment else segment


def escape_currency(text: str) -> str:
    """Escape ``$5``-style amounts so they don't open inline math in Obsidian."""
    if "$" not in text:
        return text
    return map_prose(text, currency_escaped)


_HTML_TAG_RE = re.compile(r"<(?=[A-Za-z/!?])")


def html_escaped(segment: str) -> str:
    """Prose-level part of :func:`escape_html`."""
    return _HTML_TAG_RE.sub("&lt;", segment) if "<" in segment else segment


def escape_html(text: str) -> str:
    """Neutralize raw HTML in user-typed text so it shows literally."""
    if "<" not in text:
        return text
    return map_prose(text, html_escaped)


def link_url(url: str) -> str:
    """A URL that survives as a Markdown link destination."""
    url = url.strip().replace(" ", "%20").replace("<", "%3C").replace(">", "%3E")
    if url.count("(") != url.count(")"):
        url = url.replace("(", "%28").replace(")", "%29")
    return url


def fence_for(content: str, char: str = "`") -> str:
    """A code fence long enough to wrap ``content`` safely."""
    longest = max((len(m) for m in re.findall(re.escape(char) + "{3,}", content)), default=0)
    return char * max(3, longest + 1)


def blockquote(text: str) -> str:
    """Prefix every line with ``> ``."""
    return "\n".join(f"> {line}" if line else ">" for line in text.strip("\n").split("\n"))


def escape_inline(text: str) -> str:
    """Make a short string safe inside link text, headings, or ``<summary>``."""
    text = " ".join(text.split())
    return re.sub(r"([\\`*_\[\]<>#|])", r"\\\1", text)
