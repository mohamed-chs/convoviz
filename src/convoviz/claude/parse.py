"""Translate a conversation of the claude.ai data export into the neutral model.

The format is documented in ``docs/dev/claude-export-format.md``. In short: a
conversation is a flat list of messages (``sender`` ``human`` or ``assistant``),
linked into a tree by ``parent_message_uuid`` since 2026. Each message holds
typed content blocks: text, thinking, tool calls and their results. Tool names
drift (``bash_tool`` became ``Bash`` in October 2026), so they are normalized
first. As in the ChatGPT adapter, nothing here raises on odd data: unknown
shapes degrade to text or nothing, and are counted in the diagnostics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from convoviz.model import (
    CITE_CLOSE,
    CITE_OPEN,
    Attachment,
    AttachmentBlock,
    Block,
    CodeBlock,
    Conversation,
    DocumentBlock,
    FileTextBlock,
    Message,
    Node,
    NoticeBlock,
    OutputBlock,
    ReasoningBlock,
    Role,
    Source,
    TextBlock,
)
from convoviz.values import as_dict, as_str, as_time, kind_for

if TYPE_CHECKING:
    from convoviz.diagnostics import Diagnostics

# Root messages point to the zero UUID (version 4 flavor); accept the nil UUID too.
_ROOT_PARENTS = frozenset(
    {"00000000-0000-4000-8000-000000000000", "00000000-0000-0000-0000-000000000000"}
)
_ROLES: dict[str, Role] = {"human": "user", "user": "user", "assistant": "assistant"}

# Tools renamed in October 2026 (after Claude Code's names) → their earlier names.
_ALIASES = {
    "Bash": "bash_tool",
    "WebSearch": "web_search",
    "WebFetch": "web_fetch",
    "ToolSearch": "tool_search",
    "SearchMcpRegistry": "search_mcp_registry",
    "Write": "create_file",
    "Edit": "str_replace",
    "Read": "view",
}
_SHELL_TOOLS = frozenset({"bash_tool", "bash"})
_FILE_CREATE = frozenset({"create_file"})
_FILE_EDIT = frozenset({"str_replace"})
_WIDGET = "visualize:show_widget"
_MEMORY_EDITS = "memory_user_edits"
_RESEARCH = "launch_extended_search_task"
# Content blocks that are context claude.ai added or bookkeeping, never shown.
_HIDDEN_BLOCKS = frozenset({"injected_prompt_block", "token_budget", "redacted_thinking"})

_ARTIFACT_TYPES = {
    "text/markdown": "document",
    "text/html": "code/html",
    "image/svg+xml": "code/svg",
    "application/vnd.ant.react": "code/react",
    "application/vnd.ant.mermaid": "code/mermaid",
}
_EXTENSION_TYPES = {
    ".md": "document",
    ".markdown": "document",
    ".py": "code/python",
    ".js": "code/javascript",
    ".mjs": "code/javascript",
    ".jsx": "code/react",
    ".ts": "code/typescript",
    ".tsx": "code/tsx",
    ".html": "code/html",
    ".htm": "code/html",
    ".css": "code/css",
    ".json": "code/json",
    ".sh": "code/bash",
    ".sql": "code/sql",
    ".svg": "code/svg",
    ".yaml": "code/yaml",
    ".yml": "code/yaml",
    ".tex": "code/latex",
    ".csv": "code/csv",
    ".txt": "code/text",
    ".mmd": "code/mermaid",
    ".java": "code/java",
    ".go": "code/go",
    ".rs": "code/rust",
    ".rb": "code/ruby",
    ".php": "code/php",
    ".c": "code/c",
    ".cpp": "code/cpp",
    ".cs": "code/csharp",
    ".swift": "code/swift",
    ".kt": "code/kotlin",
    ".r": "code/r",
}
_TITLE_LENGTH = 60


def parse_conversation(raw: dict[str, Any], diagnostics: Diagnostics) -> Conversation | None:
    """Build a :class:`Conversation`, or return None if it is unusable."""
    conv_id = as_str(raw.get("uuid")) or as_str(raw.get("id"))
    raw_messages = raw.get("chat_messages", raw.get("messages"))
    if not conv_id or not isinstance(raw_messages, list):
        diagnostics.skip("conversation without an id or messages")
        return None

    ctx = _Context(conv_id, diagnostics)
    nodes, raw_by_id = _build_nodes(raw_messages, ctx)
    _materialize_messages(nodes, raw_by_id, ctx)

    times = [n.time for n in nodes.values() if n.time is not None]
    created = as_time(raw.get("created_at")) or (min(times) if times else None)
    updated = as_time(raw.get("updated_at")) or (max(times) if times else None) or created
    if created is None or updated is None:
        created = updated = datetime.fromtimestamp(0, UTC)
        diagnostics.warn(f"conversation {conv_id}: no timestamps; dated 1970-01-01")

    leaf = as_str(raw.get("current_leaf_message_uuid"))
    project = as_str(raw.get("project_uuid")) or as_str(as_dict(raw.get("project")).get("uuid"))
    return Conversation(
        id=conv_id,
        title=_title(raw, nodes),
        created=created,
        updated=updated,
        nodes=nodes,
        current_node=None if leaf in _ROOT_PARENTS else leaf,
        url=f"https://claude.ai/chat/{conv_id}",
        default_model=as_str(raw.get("model")),
        starred=raw.get("is_starred") is True,
        project_id=project,
        provider="claude",
    )


# --------------------------------------------------------------------------- context


@dataclass(slots=True)
class _Document:
    """An artifact or file as it stands at one point of a branch."""

    title: str
    content: str
    doc_type: str | None
    kind: str
    version: int = 1


@dataclass(slots=True)
class _Context:
    conv_id: str
    diagnostics: Diagnostics
    # Page titles from search and fetch results, for naming cited sources.
    titles: dict[str, str] = field(default_factory=dict)
    # Tool name by tool_use id: results since 2026-10 often have an empty name.
    tool_names: dict[str, str] = field(default_factory=dict)

    def where(self, message_id: str) -> str:
        return f"conversation {self.conv_id}, message {message_id}"


def _build_nodes(
    raw_messages: list[Any], ctx: _Context
) -> tuple[dict[str, Node], dict[str, dict[str, Any]]]:
    """One node per message, linked by ``parent_message_uuid``.

    Exports from before 2026 have no parent links: their messages are chained in
    order. A parent that isn't in the conversation is replaced by the message
    before it, so the message stays on its branch rather than starting a new one.
    """
    entries: list[tuple[str, dict[str, Any]]] = []
    for index, item in enumerate(raw_messages):
        if not isinstance(item, dict):
            ctx.diagnostics.skip(f"conversation {ctx.conv_id}: message {index} is not an object")
            continue
        message_id = as_str(item.get("uuid")) or f"{ctx.conv_id}#{index}"
        entries.append((message_id, item))
    linked = any("parent_message_uuid" in item for _, item in entries)
    if not linked:
        entries.sort(key=lambda entry: as_time(entry[1].get("created_at")) or _EPOCH)
    ids = {message_id for message_id, _ in entries}

    nodes: dict[str, Node] = {}
    raw_by_id: dict[str, dict[str, Any]] = {}
    previous: str | None = None
    for message_id, item in entries:
        if message_id in nodes:
            ctx.diagnostics.count("claude_duplicate_messages")
            continue
        parent = as_str(item.get("parent_message_uuid")) if linked else previous
        if parent in _ROOT_PARENTS or parent == message_id:
            parent = None
        elif parent is not None and parent not in ids:
            ctx.diagnostics.count("claude_orphan_messages")
            parent = previous
        nodes[message_id] = Node(id=message_id, parent=parent, time=as_time(item.get("created_at")))
        raw_by_id[message_id] = item
        previous = message_id
    return nodes, raw_by_id


_EPOCH = datetime.min.replace(tzinfo=UTC)


def _materialize_messages(
    nodes: dict[str, Node], raw_by_id: dict[str, dict[str, Any]], ctx: _Context
) -> None:
    """Parse messages in tree order, so each branch sees its own artifact versions."""
    children: dict[str | None, list[str]] = {}
    for node in nodes.values():
        children.setdefault(node.parent if node.parent in nodes else None, []).append(node.id)
    for ids in children.values():
        ids.sort(key=lambda i: (nodes[i].time or _EPOCH, i))

    visited: set[str] = set()
    stack: list[tuple[str, dict[str, _Document]]] = [
        (root, {}) for root in reversed(children.get(None, []))
    ]
    while stack:
        node_id, documents = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        _parse_into(nodes[node_id], raw_by_id[node_id], documents, ctx)
        kids = children.get(node_id, [])
        stack.extend(
            (kid, _copy(documents) if len(kids) > 1 else documents) for kid in reversed(kids)
        )
    # Messages in parent cycles are unreachable from any root; parse them standalone.
    for node_id, node in nodes.items():
        if node_id not in visited:
            _parse_into(node, raw_by_id[node_id], {}, ctx)


def _copy(documents: dict[str, _Document]) -> dict[str, _Document]:
    return {
        key: _Document(d.title, d.content, d.doc_type, d.kind, d.version)
        for key, d in documents.items()
    }


def _parse_into(
    node: Node, raw: dict[str, Any], documents: dict[str, _Document], ctx: _Context
) -> None:
    try:
        node.message = _parse_message(node.id, raw, documents, ctx)
    except Exception as exc:  # noqa: BLE001 - one bad message must not sink the rest
        ctx.diagnostics.error(f"{ctx.where(node.id)}: could not parse message ({exc!r})")


# --------------------------------------------------------------------------- messages


def _parse_message(
    message_id: str, raw: dict[str, Any], documents: dict[str, _Document], ctx: _Context
) -> Message | None:
    sender = as_str(raw.get("sender")) or as_str(raw.get("role")) or ""
    role = _ROLES.get(sender)
    if role is None:
        ctx.diagnostics.unknown_kind(f"claude-sender:{sender or '<none>'}", ctx.where(message_id))
        return None

    blocks: list[Block] = _files(raw, ctx)
    content = raw.get("content")
    has_text = False
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            has_text = has_text or item.get("type") in ("text", "voice_note")
            blocks.extend(_block(item, message_id, documents, ctx))
    elif isinstance(content, str) and content.strip():
        blocks.append(TextBlock(content))
        has_text = True
    text = as_str(raw.get("text"))
    if not has_text and text and text.strip():
        # Exports before content blocks, or a message whose blocks lost their text.
        blocks.append(TextBlock(text))
    if not blocks:
        return None
    return Message(
        id=message_id,
        role=role,
        blocks=blocks,
        time=as_time(raw.get("created_at")),
    )


def _files(raw: dict[str, Any], ctx: _Context) -> list[Block]:
    """Attachments first, as claude.ai shows them above the prompt.

    ``attachments`` carry the text Claude extracted from a pasted or uploaded
    document. ``files`` name uploads (mostly images) whose bytes the export never
    includes. A file listed in both is shown once, with its text.
    """
    blocks: list[Block] = []
    with_text: set[str] = set()
    attachments = raw.get("attachments")
    for item in attachments if isinstance(attachments, list) else ():
        if not isinstance(item, dict):
            continue
        name = (as_str(item.get("file_name")) or "").strip() or None
        text = as_str(item.get("extracted_content")) or ""
        if text.strip():
            blocks.append(FileTextBlock(text, name, as_str(item.get("file_type"))))
            if name:
                with_text.add(name)
        elif name:
            blocks.append(_missing_file(name, None, ctx))
    files = raw.get("files")
    if not isinstance(files, list) or not files:
        files = raw.get("files_v2")
    for item in files if isinstance(files, list) else ():
        if not isinstance(item, dict):
            continue
        name = (as_str(item.get("file_name")) or "").strip() or None
        if name in with_text:
            continue
        file_id = as_str(item.get("file_uuid")) or as_str(item.get("uuid"))
        if name or file_id:
            blocks.append(_missing_file(name, file_id, ctx))
    return blocks


def _missing_file(name: str | None, file_id: str | None, ctx: _Context) -> AttachmentBlock:
    """A file the export names but doesn't contain (it never contains uploads)."""
    ctx.diagnostics.count("claude_files_without_payload")
    file_id = f"claude-file:{file_id or name}"
    return AttachmentBlock(Attachment(file_id, kind_for(None, name), name=name))


# --------------------------------------------------------------------------- blocks


def _block(
    item: dict[str, Any], message_id: str, documents: dict[str, _Document], ctx: _Context
) -> list[Block]:
    kind = as_str(item.get("type")) or ""
    match kind:
        case "text":
            return _text(item, ctx)
        case "thinking":
            return _thinking(item)
        case "tool_use":
            return _tool_use(item, message_id, documents, ctx)
        case "tool_result":
            return _tool_result(item, documents, ctx)
        case "voice_note":
            text = as_str(item.get("text")) or ""
            return [TextBlock(text)] if text.strip() else []
        case _ if kind in _HIDDEN_BLOCKS:
            ctx.diagnostics.count(f"hidden:{kind}")
            return []
        case "image" | "document" if as_str(item.get("file_name")) or as_str(item.get("file_uuid")):
            name = as_str(item.get("file_name"))
            return [_missing_file(name, as_str(item.get("file_uuid")), ctx)]
    ctx.diagnostics.unknown_kind(f"claude-block:{kind or '<none>'}", ctx.where(message_id))
    text = as_str(item.get("text")) or ""
    return [TextBlock(text)] if text.strip() else []


def _text(item: dict[str, Any], ctx: _Context) -> list[Block]:
    text = as_str(item.get("text")) or ""
    if not text.strip():
        return []
    citations = item.get("citations")
    if not isinstance(citations, list) or not citations:
        return [TextBlock(text)]
    sources: list[Source] = []
    index_by_url: dict[str, int] = {}
    markers: dict[int, list[int]] = {}
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        details = as_dict(citation.get("details"))
        url = as_str(details.get("url")) or as_str(citation.get("url"))
        end = citation.get("end_index")
        if not url or not isinstance(end, int) or isinstance(end, bool):
            continue
        if url not in index_by_url:
            title = as_str(details.get("title")) or ctx.titles.get(url)
            index_by_url[url] = len(sources)
            sources.append(Source(url, title))
        position = _word_end(text, max(0, min(end, len(text))))
        refs = markers.setdefault(position, [])
        if index_by_url[url] not in refs:
            refs.append(index_by_url[url])
    if not sources:
        return [TextBlock(text)]
    for position in sorted(markers, reverse=True):
        ids = ",".join(str(i) for i in markers[position])
        text = f"{text[:position]}{CITE_OPEN}{ids}{CITE_CLOSE}{text[position:]}"
    return [TextBlock(text, tuple(sources))]


def _word_end(text: str, position: int) -> int:
    """Move a citation that would split a word to the end of that word.

    Indices could count UTF-16 units rather than characters; this keeps the
    footnote readable when they are off by a little.
    """
    while 0 < position < len(text) and text[position - 1].isalnum() and text[position].isalnum():
        position += 1
    return position


def _thinking(item: dict[str, Any]) -> list[Block]:
    text = (as_str(item.get("thinking")) or "").strip()
    if not text:
        return []
    summaries = [
        summary.strip()
        for entry in item.get("summaries") or ()
        if isinstance(entry, dict) and (summary := as_str(entry.get("summary"))) and summary.strip()
    ]
    return [ReasoningBlock(text, summaries[-1] if summaries else None)]


# --------------------------------------------------------------------------- tools


def _tool_name(item: dict[str, Any]) -> str:
    name = as_str(item.get("name")) or ""
    return _ALIASES.get(name, name)


def _is_mcp(name: str, item: dict[str, Any]) -> bool:
    if name.startswith("mcp__"):
        return True
    if name.startswith("visualize:"):
        return False
    return ":" in name or bool(as_str(item.get("mcp_server_url")))


def _mcp_label(name: str, item: dict[str, Any]) -> str:
    """``mcp__github__create_issue`` or ``GitHub:create_issue`` → ``create_issue (GitHub)``."""
    if name.startswith("mcp__") and "__" in name[5:]:
        server, tool = name[5:].split("__", 1)
    elif ":" in name:
        server, tool = name.split(":", 1)
    else:
        server, tool = "", name
    server = as_str(item.get("integration_name")) or server
    return f"{tool} ({server})" if server else tool


def _tool_use(
    item: dict[str, Any], message_id: str, documents: dict[str, _Document], ctx: _Context
) -> list[Block]:
    name = _tool_name(item)
    tool_id = as_str(item.get("id"))
    if tool_id:
        ctx.tool_names[tool_id] = name
    data = as_dict(item.get("input"))
    if name == "artifacts":
        return _artifact(data, message_id, documents, ctx)
    if name in _FILE_CREATE:
        return _file_create(data, message_id, documents, ctx)
    if name in _FILE_EDIT:
        return _file_edit(data, documents, ctx)
    if name == _WIDGET:
        return _widget(data, documents)
    if name in _SHELL_TOOLS:
        command = as_str(data.get("command")) or ""
        return [CodeBlock(command, "bash", name)] if command.strip() else []
    if name == "repl":
        code = as_str(data.get("code")) or ""
        return (
            [CodeBlock(code, "javascript", name, "Analysis (javascript)")] if code.strip() else []
        )
    if name == _MEMORY_EDITS:
        command = as_str(data.get("command"))
        return [] if command in (None, "view") else [NoticeBlock("Memory updated")]
    if name == _RESEARCH:
        return [NoticeBlock("Started research")]
    if _is_mcp(name, item):
        payload = json.dumps(data, ensure_ascii=False, indent=2) if data else "{}"
        return [CodeBlock(payload, "json", name, _mcp_label(name, item))]
    # Searches, file reads, memory lookups, tool discovery…: plumbing.
    ctx.diagnostics.count(f"tool_call:{name or '<none>'}")
    return []


def _tool_result(
    item: dict[str, Any], documents: dict[str, _Document], ctx: _Context
) -> list[Block]:
    name = _tool_name(item) or ctx.tool_names.get(as_str(item.get("tool_use_id")) or "", "")
    content = item.get("content")
    items = content if isinstance(content, list) else []
    texts: list[str] = []
    blocks: list[Block] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        etype = as_str(entry.get("type"))
        if etype == "knowledge":
            url = as_str(entry.get("url"))
            title = (as_str(entry.get("title")) or "").strip()
            if url and title:
                ctx.titles.setdefault(url, title)
        elif etype == "text":
            texts.append(as_str(entry.get("text")) or "")
        elif etype == "local_resource" and name == "present_files":
            blocks.extend(_presented(entry, documents, ctx))
    if isinstance(content, str):
        texts.append(content)
    shown = name in _SHELL_TOOLS or name == "repl" or _is_mcp(name, item)
    output = "\n".join(t for t in texts if t).strip()
    if shown and output:
        blocks.append(OutputBlock(output, name, "Error" if item.get("is_error") is True else None))
    return blocks


def _presented(
    entry: dict[str, Any], documents: dict[str, _Document], ctx: _Context
) -> list[Block]:
    """A file Claude made and offered for download.

    Text files Claude wrote are already shown as File blocks; the others (docx,
    xlsx, pdf…) aren't in the export.
    """
    path = as_str(entry.get("file_path")) or ""
    name = as_str(entry.get("name")) or PurePosixPath(path).name or None
    if f"file:{path}" in documents:
        return []
    ctx.diagnostics.count("claude_files_without_payload")
    file_id = f"claude-file:{as_str(entry.get('uuid')) or path or name}"
    mime = as_str(entry.get("mime_type"))
    return [AttachmentBlock(Attachment(file_id, kind_for(mime, name), name=name, mime=mime))]


# --------------------------------------------------------------------------- documents


def _artifact(
    data: dict[str, Any], message_id: str, documents: dict[str, _Document], ctx: _Context
) -> list[Block]:
    """The 2024–2025 ``artifacts`` tool: create, rewrite or patch an artifact."""
    command = as_str(data.get("command")) or "create"
    artifact_id = as_str(data.get("id")) or as_str(data.get("title")) or "artifact"
    key = f"artifact:{artifact_id}"
    current = documents.get(key)
    if command == "view":
        return []
    if command == "update":
        if current is None:
            ctx.diagnostics.unknown_kind("artifact:update-without-artifact", ctx.where(message_id))
            return []
        content = _replace(current.content, data.get("old_str"), data.get("new_str"), ctx)
    else:
        content = as_str(data.get("content"))
        if content is None:
            ctx.diagnostics.unknown_kind(f"artifact:{command}", ctx.where(message_id))
            return []
    mime = as_str(data.get("type"))
    language = (as_str(data.get("language")) or "").strip().lower()
    if mime == "application/vnd.ant.code" or (mime is None and language):
        doc_type = f"code/{language or 'text'}"
    else:
        doc_type = _ARTIFACT_TYPES.get(mime or "", current.doc_type if current else "document")
    title = (as_str(data.get("title")) or "").strip() or (current.title if current else "")
    return [_store(documents, key, title or "Untitled artifact", content, doc_type, "Artifact")]


def _file_create(
    data: dict[str, Any], message_id: str, documents: dict[str, _Document], ctx: _Context
) -> list[Block]:
    path = as_str(data.get("path")) or as_str(data.get("file_path")) or ""
    content = as_str(data.get("file_text"))
    if content is None:
        content = as_str(data.get("content"))
    if not path or content is None:
        ctx.diagnostics.unknown_kind("file:create-without-text", ctx.where(message_id))
        return []
    return [
        _store(
            documents, f"file:{path}", PurePosixPath(path).name, content, _file_type(path), "File"
        )
    ]


def _file_edit(data: dict[str, Any], documents: dict[str, _Document], ctx: _Context) -> list[Block]:
    path = as_str(data.get("path")) or as_str(data.get("file_path")) or ""
    current = documents.get(f"file:{path}")
    if current is None:
        # An uploaded or earlier file Claude never showed in full: nothing to patch.
        ctx.diagnostics.count("claude_edits_of_unknown_files")
        return []
    old = data.get("old_str", data.get("old_string"))
    new = data.get("new_str", data.get("new_string"))
    content = _replace(current.content, old, new, ctx, every=data.get("replace_all") is True)
    return [_store(documents, f"file:{path}", current.title, content, current.doc_type, "File")]


def _widget(data: dict[str, Any], documents: dict[str, _Document]) -> list[Block]:
    code = as_str(data.get("widget_code")) or ""
    if not code.strip():
        return []
    title = (as_str(data.get("title")) or "").strip() or "Widget"
    doc_type = "code/svg" if code.lstrip().startswith("<svg") else "code/html"
    return [_store(documents, f"widget:{title}", title, code, doc_type, "Widget")]


def _store(
    documents: dict[str, _Document],
    key: str,
    title: str,
    content: str,
    doc_type: str | None,
    kind: str,
) -> DocumentBlock:
    current = documents.get(key)
    version = current.version + 1 if current else 1
    documents[key] = _Document(title, content, doc_type, kind, version)
    return DocumentBlock(title, content, doc_type, version, kind)


def _replace(content: str, old: Any, new: Any, ctx: _Context, *, every: bool = False) -> str:
    if not isinstance(old, str) or not isinstance(new, str) or not old:
        ctx.diagnostics.count("claude_edits_unreadable")
        return content
    if old not in content:
        ctx.diagnostics.count("claude_edits_not_found")
        return content
    return content.replace(old, new) if every else content.replace(old, new, 1)


def _file_type(path: str) -> str:
    return _EXTENSION_TYPES.get(PurePosixPath(path).suffix.lower(), "code/text")


# --------------------------------------------------------------------------- title


_SPACE_RE = re.compile(r"\s+")


def _title(raw: dict[str, Any], nodes: dict[str, Node]) -> str:
    """The conversation's name or, when claude.ai never named it, its first prompt."""
    name = (as_str(raw.get("name")) or "").strip()
    if name:
        return name
    prompts = sorted(
        (n for n in nodes.values() if n.message and n.message.role == "user" and n.message.text),
        key=lambda n: (n.time or _EPOCH, n.id),
    )
    if not prompts:
        return "Untitled"
    text = _SPACE_RE.sub(" ", prompts[0].message.text if prompts[0].message else "").strip()
    if len(text) <= _TITLE_LENGTH:
        return text or "Untitled"
    cut = text[:_TITLE_LENGTH].rsplit(" ", 1)[0] or text[:_TITLE_LENGTH]
    return cut.rstrip(" ,.;:") + "…"
