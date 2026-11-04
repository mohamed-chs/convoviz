"""ChatGPT-specific cleanups of assistant prose."""

from __future__ import annotations

import re

from convoviz.markdown import map_prose

# Canvas-style inline blocks: :::writing{variant="email" id="..."} ... :::
_DIRECTIVE_OPEN_RE = re.compile(r"^[ \t]*:::[A-Za-z_][\w-]*\{[^}\n]*\}[ \t]*$", re.MULTILINE)
_DIRECTIVE_CLOSE_RE = re.compile(r"^[ \t]*:::[ \t]*$", re.MULTILINE)
# Links to files in the code-interpreter sandbox can't work outside ChatGPT.
_SANDBOX_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(sandbox:[^)\s]+\)")


def clean_assistant_text(text: str) -> str:
    """Remove UI-only syntax that would render as noise outside ChatGPT."""
    if ":::" in text:
        text = _DIRECTIVE_OPEN_RE.sub("", text)
        text = _DIRECTIVE_CLOSE_RE.sub("", text)
    if "sandbox:" in text:
        text = map_prose(text, lambda s: _SANDBOX_LINK_RE.sub(r"`\1`", s))
    return text
