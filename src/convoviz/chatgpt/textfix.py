"""ChatGPT-specific cleanups of assistant prose."""

from __future__ import annotations

import re
from dataclasses import dataclass

from convoviz.markdown import map_prose, split_fenced

# Canvas-style inline blocks: :::writing{variant="email" subject="..." id="..."} ... :::
_DIRECTIVE_OPEN_RE = re.compile(r"^[ \t]*:::([A-Za-z_][\w-]*)\{([^}\n]*)\}[ \t]*$")
_DIRECTIVE_CLOSE_RE = re.compile(r"^[ \t]*:::[ \t]*$")
_ATTRIBUTE_RE = re.compile(r'([A-Za-z_][\w-]*)\s*=\s*"([^"]*)"')
# Links to files in the code-interpreter sandbox can't work outside ChatGPT.
_SANDBOX_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(sandbox:[^)\s]+\)")

_VARIANTS = {
    "document": "Document",
    "email": "Email",
    "social_post": "Social post",
    "standard": "Draft",
}


@dataclass(frozen=True, slots=True)
class Directive:
    """A ``:::writing`` block: a draft ChatGPT shows as a card."""

    label: str
    title: str
    body: str


def split_directives(text: str) -> list[str | Directive]:
    """Split assistant text into prose and ``:::name{…}`` blocks (outside code).

    An unclosed block runs to the end of the text.
    """
    if ":::" not in text:
        return [text]
    out: list[str | Directive] = []
    prose: list[str] = []
    body: list[str] | None = None
    header: tuple[str, str] = ("", "")

    def finish() -> None:
        name, attributes = header
        attrs = dict(_ATTRIBUTE_RE.findall(attributes))
        label = _VARIANTS.get(attrs.get("variant", ""), "Canvas") if name == "writing" else "Canvas"
        title = attrs.get("subject") or attrs.get("title") or ""
        out.append(Directive(label, title, "".join(body or []).strip("\n")))

    for is_code, chunk in split_fenced(text):
        if is_code:
            (body if body is not None else prose).append(chunk)
            continue
        for line in chunk.splitlines(keepends=True):
            bare = line.rstrip("\r\n")
            if body is None and (match := _DIRECTIVE_OPEN_RE.match(bare)):
                if prose:
                    out.append("".join(prose))
                    prose = []
                header, body = (match.group(1), match.group(2)), []
            elif body is not None and _DIRECTIVE_CLOSE_RE.match(bare):
                finish()
                body = None
            else:
                (body if body is not None else prose).append(line)
    if body is not None:
        finish()
    if prose:
        out.append("".join(prose))
    return out


def clean_assistant_text(text: str) -> str:
    """Remove UI-only syntax that would render as noise outside ChatGPT."""
    if "sandbox:" in text:
        text = map_prose(text, lambda s: _SANDBOX_LINK_RE.sub(r"`\1`", s))
    return text
