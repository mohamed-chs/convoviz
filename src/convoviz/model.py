"""Provider-neutral conversation model.

Provider adapters (``convoviz.chatgpt``, ``convoviz.claude``) translate raw export
JSON into these types; renderers and analytics only ever see these. The model is
deliberately small: it keeps what a reader of the conversation would care about,
already interpreted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime

Role = Literal["user", "assistant", "tool", "system"]

# How each provider's assistant is called, by ``Conversation.provider``.
ASSISTANT_NAMES = {"chatgpt": "ChatGPT", "claude": "Claude"}
AttachmentKind = Literal["image", "audio", "video", "file"]

# Inline source references inside TextBlock.text are encoded as
# CITE_OPEN + "3,4" + CITE_CLOSE, where the numbers index TextBlock.sources.
# These are ASCII control characters, which never occur in real chat text.
CITE_OPEN = "\x1e"
CITE_CLOSE = "\x1f"


@dataclass(frozen=True, slots=True)
class Source:
    """A web page (or other document) cited by the assistant."""

    url: str | None
    title: str | None = None
    attribution: str | None = None


@dataclass(frozen=True, slots=True)
class Attachment:
    """A file referenced by a message (upload, generated image, voice clip...)."""

    file_id: str
    kind: AttachmentKind
    name: str | None = None
    mime: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True, slots=True)
class TextBlock:
    """Markdown prose. May contain encoded citation markers (see ``CITE_OPEN``)."""

    text: str
    sources: tuple[Source, ...] = ()


@dataclass(frozen=True, slots=True)
class ReasoningBlock:
    """The model's visible reasoning summary ("Thought for 12s")."""

    text: str
    title: str | None = None


@dataclass(frozen=True, slots=True)
class CodeBlock:
    """Code the assistant ran or sent to a tool."""

    code: str
    language: str = ""
    tool: str | None = None
    title: str | None = None  # what the fold says instead of "Code (language)"


@dataclass(frozen=True, slots=True)
class OutputBlock:
    """Output returned by a tool (code execution result, etc.)."""

    text: str
    tool: str | None = None
    title: str | None = None  # what the fold says instead of "Output"


@dataclass(frozen=True, slots=True)
class QuoteBlock:
    """A quoted excerpt from a web page."""

    text: str
    title: str | None = None
    url: str | None = None


@dataclass(frozen=True, slots=True)
class AttachmentBlock:
    """An inline file: rendered as an embedded image or a link."""

    attachment: Attachment
    caption: str | None = None


@dataclass(frozen=True, slots=True)
class FileTextBlock:
    """An attached file kept only as the text the provider extracted from it."""

    text: str
    name: str | None = None
    file_type: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentBlock:
    """A document the assistant wrote or edited, with its full current content.

    ``doc_type`` is ``"document"`` (Markdown prose) or ``"code/<language>"``.
    ``kind`` is what the provider calls it: a ChatGPT "Canvas", a Claude
    "Artifact" or "File", or an inline draft ("Email", "Document", ...).
    """

    title: str
    content: str
    doc_type: str | None = None
    version: int = 1
    kind: str = "Canvas"


@dataclass(frozen=True, slots=True)
class NoticeBlock:
    """A short status line (memory updated, tool error...)."""

    text: str


Block = (
    TextBlock
    | ReasoningBlock
    | CodeBlock
    | OutputBlock
    | QuoteBlock
    | AttachmentBlock
    | FileTextBlock
    | DocumentBlock
    | NoticeBlock
)


@dataclass(slots=True)
class Message:
    """One visible message. Hidden/internal messages are never materialized."""

    id: str
    role: Role
    blocks: list[Block]
    time: datetime | None = None
    model: str | None = None
    author_name: str | None = None
    scheduled: bool = False  # a run of a scheduled task, a reply of its own

    @property
    def text(self) -> str:
        """Plain prose of the message (for search and statistics)."""
        return "\n\n".join(strip_citations(b.text) for b in self.blocks if isinstance(b, TextBlock))


@dataclass(slots=True)
class Node:
    """A position in the conversation tree. ``message`` is None for hidden nodes."""

    id: str
    parent: str | None
    message: Message | None = None
    time: datetime | None = None  # kept even for hidden nodes, for branch ordering
    children_hint: tuple[str, ...] = ()  # legacy exports list children explicitly


@dataclass(slots=True)
class Conversation:
    """A conversation: a tree of nodes plus metadata."""

    id: str
    title: str
    created: datetime
    updated: datetime
    nodes: dict[str, Node]
    current_node: str | None
    url: str | None = None
    default_model: str | None = None
    starred: bool = False
    archived: bool = False
    gizmo_id: str | None = None
    project_id: str | None = None
    provider: str = "chatgpt"
    # Files that belong to the conversation but appear in no message (e.g. the
    # export's file library); rendered at the end of the note.
    files: list[Attachment] = field(default_factory=list)
    extra: dict[str, object] = field(default_factory=dict)


def strip_citations(text: str) -> str:
    """Remove encoded citation markers from text."""
    if CITE_OPEN not in text:
        return text
    out: list[str] = []
    i = 0
    while True:
        start = text.find(CITE_OPEN, i)
        if start < 0:
            out.append(text[i:])
            break
        end = text.find(CITE_CLOSE, start)
        if end < 0:
            out.append(text[i:])
            break
        out.append(text[i:start])
        i = end + 1
    return "".join(out)
