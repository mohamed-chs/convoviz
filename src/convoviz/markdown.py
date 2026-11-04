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


def split_fenced(text: str) -> list[tuple[bool, str]]:
    """Split into (is_code, chunk) by fenced code blocks; unclosed fences run to the end."""
    chunks: list[tuple[bool, str]] = []
    buffer: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        match = _FENCE_RE.match(line.rstrip("\r\n"))
        if fence is None:
            if match and not (match.group(2)[0] == "`" and "`" in match.group(3)):
                if buffer:
                    chunks.append((False, "".join(buffer)))
                buffer = [line]
                fence = match.group(2)
            else:
                buffer.append(line)
        else:
            buffer.append(line)
            if (
                match
                and match.group(2)[0] == fence[0]
                and len(match.group(2)) >= len(fence)
                and not match.group(3).strip()
            ):
                chunks.append((True, "".join(buffer)))
                buffer = []
                fence = None
    if buffer:
        chunks.append((fence is not None, "".join(buffer)))
    return chunks


def close_fences(text: str) -> str:
    """Close a dangling code fence so it cannot swallow the rest of the document."""
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


def map_prose(text: str, transform: Callable[[str], str]) -> str:
    """Apply ``transform`` to everything except fenced and inline code."""
    out: list[str] = []
    for is_code, chunk in split_fenced(text):
        if is_code:
            out.append(chunk)
            continue
        position = 0
        for match in _INLINE_CODE_RE.finditer(chunk):
            out.append(transform(chunk[position : match.start()]))
            out.append(match.group(0))
            position = match.end()
        out.append(transform(chunk[position:]))
    return "".join(out)


_DISPLAY_MATH_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def convert_math(text: str) -> str:
    r"""Rewrite ``\[ … \]`` and ``\( … \)`` to ``$$ … $$`` and ``$ … $``.

    Obsidian and GitHub only understand dollar delimiters.
    """

    def prose(segment: str) -> str:
        if "\\[" in segment:
            segment = _DISPLAY_MATH_RE.sub(lambda m: f"$${m.group(1)}$$", segment)
        if "\\(" in segment:
            segment = _INLINE_MATH_RE.sub(lambda m: f"${m.group(1).strip()}$", segment)
        return segment

    return map_prose(text, prose)


_CURRENCY_RE = re.compile(r"(?<![\\$\w])\$(?=\d)")


def escape_currency(text: str) -> str:
    """Escape ``$5``-style amounts so they don't open inline math in Obsidian."""
    if "$" not in text:
        return text
    return map_prose(text, lambda s: _CURRENCY_RE.sub(r"\\$", s))


_HTML_TAG_RE = re.compile(r"<(?=[A-Za-z/!?])")


def escape_html(text: str) -> str:
    """Neutralize raw HTML in user-typed text so it shows literally."""
    if "<" not in text:
        return text
    return map_prose(text, lambda s: _HTML_TAG_RE.sub("&lt;", s))


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
