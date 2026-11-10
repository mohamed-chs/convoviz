"""Translate raw ChatGPT conversation JSON into the neutral model.

Every accessor here is defensive: the export has no published schema and keys come
and go between releases. A malformed message degrades to nothing (and a diagnostic),
never to an exception that aborts the conversation, let alone the run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import orjson

from convoviz.chatgpt.citations import render_text
from convoviz.chatgpt.textfix import Directive, clean_assistant_text, split_directives
from convoviz.model import (
    Attachment,
    AttachmentBlock,
    AttachmentKind,
    Block,
    CodeBlock,
    Conversation,
    DocumentBlock,
    Message,
    Node,
    NoticeBlock,
    OutputBlock,
    QuoteBlock,
    ReasoningBlock,
    Role,
    TextBlock,
    strip_citations,
)
from convoviz.values import as_dict as _dict
from convoviz.values import as_str as _str
from convoviz.values import as_time as _time
from convoviz.values import kind_for

if TYPE_CHECKING:
    from convoviz.diagnostics import Diagnostics

# Content types that are model plumbing, never conversation content.
_REASONING = frozenset({"thoughts", "reasoning_recap"})
_HIDDEN_CONTENT = frozenset(
    {
        "tether_browsing_display",
        "tether_browsing_code",
        "sonic_webpage",
        "system_error",
        "user_editable_context",
        "model_editable_context",
        "about_user_message",
        "about_model_message",
    }
)
# Tools whose replies are plumbing (search results, memory writes, plans...).
_CODE_TOOLS = frozenset({"python", "container.exec", "jupyter", "code_interpreter"})
_ASSET_PART_KIND: dict[str, AttachmentKind] = {
    "image_asset_pointer": "image",
    "audio_asset_pointer": "audio",
    "video_container_asset_pointer": "video",
}


def parse_conversation(raw: dict[str, Any], diagnostics: Diagnostics) -> Conversation | None:
    """Build a :class:`Conversation`, or return None if it is unusable."""
    conv_id = _str(raw.get("conversation_id")) or _str(raw.get("id"))
    mapping = raw.get("mapping")
    if not conv_id or not isinstance(mapping, dict):
        diagnostics.skip("conversation without an id or message mapping")
        return None

    ctx = _Context(conv_id, diagnostics)
    raw_nodes: dict[str, dict[str, Any]] = {}
    nodes: dict[str, Node] = {}
    for key, value in mapping.items():
        if not isinstance(value, dict):
            continue
        node_id = _str(value.get("id")) or _str(key)
        if not node_id:
            continue
        message = value.get("message") if isinstance(value.get("message"), dict) else None
        children = value.get("children")
        nodes[node_id] = Node(
            id=node_id,
            parent=_str(value.get("parent")),
            time=_time(message.get("create_time")) if message else None,
            children_hint=tuple(c for c in children if isinstance(c, str))
            if isinstance(children, list)
            else (),
        )
        if message is not None:
            raw_nodes[node_id] = message

    _materialize_messages(nodes, raw_nodes, ctx)

    times = [n.time for n in nodes.values() if n.time is not None]
    created = _time(raw.get("create_time")) or (min(times) if times else None)
    updated = _time(raw.get("update_time")) or (max(times) if times else None) or created
    if created is None or updated is None:
        created = updated = datetime.fromtimestamp(0, UTC)
        diagnostics.warn(f"conversation {conv_id}: no timestamps; dated 1970-01-01")

    gizmo = _str(raw.get("gizmo_id"))
    conversation = Conversation(
        id=conv_id,
        title=(_str(raw.get("title")) or "").strip() or "Untitled",
        created=created,
        updated=updated,
        nodes=nodes,
        current_node=_str(raw.get("current_node")),
        url=f"https://chatgpt.com/c/{conv_id}",
        default_model=_str(raw.get("default_model_slug")),
        starred=raw.get("is_starred") is True,
        archived=raw.get("is_archived") is True,
        gizmo_id=gizmo if gizmo and not gizmo.startswith("g-p-") else None,
        project_id=gizmo if gizmo and gizmo.startswith("g-p-") else None,
    )
    if ctx.custom_instructions:
        conversation.extra["custom_instructions"] = ctx.custom_instructions
    return conversation


# --------------------------------------------------------------------------- context


@dataclass(slots=True)
class _Canvas:
    title: str
    content: str
    doc_type: str | None
    version: int = 1


@dataclass(slots=True)
class _Context:
    conv_id: str
    diagnostics: Diagnostics
    custom_instructions: dict[str, str] = field(default_factory=dict)
    # Image prompts by the node of the tool call that sent them.
    prompts: dict[str, str] = field(default_factory=dict)
    parent: str | None = None  # of the message being parsed
    # Replies followed by a reasoning step: thinking preambles (see _materialize_messages).
    preambles: set[str] = field(default_factory=set)

    def where(self, node_id: str) -> str:
        return f"conversation {self.conv_id}, node {node_id}"


def _materialize_messages(
    nodes: dict[str, Node], raw_nodes: dict[str, dict[str, Any]], ctx: _Context
) -> None:
    """Parse messages in tree order so Canvas edits apply to the right version."""
    # The 2026 export drops ``is_thinking_preamble_message`` but keeps the
    # preambles. A reply followed by a reasoning step is one.
    ctx.preambles = {
        parent
        for node_id, raw in raw_nodes.items()
        if (parent := nodes[node_id].parent) is not None
        and _str(_dict(raw.get("content")).get("content_type")) in _REASONING
    }
    children: dict[str | None, list[str]] = {}
    for node in nodes.values():
        parent = node.parent if node.parent in nodes else None
        children.setdefault(parent, []).append(node.id)
    for ids in children.values():
        ids.sort(key=lambda i: (nodes[i].time or datetime.min.replace(tzinfo=UTC), i))

    visited: set[str] = set()
    stack: list[tuple[str, dict[str, _Canvas]]] = [
        (root, {}) for root in reversed(children.get(None, []))
    ]
    while stack:
        node_id, canvases = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        raw = raw_nodes.get(node_id)
        if raw is not None:
            ctx.parent = nodes[node_id].parent
            try:
                nodes[node_id].message = _parse_message(node_id, raw, canvases, ctx)
            except Exception as exc:  # noqa: BLE001 - one bad message must not sink the rest
                ctx.diagnostics.error(f"{ctx.where(node_id)}: could not parse message ({exc!r})")
        kids = children.get(node_id, [])
        stack.extend((kid, dict(canvases) if len(kids) > 1 else canvases) for kid in reversed(kids))
    # Nodes in parent cycles are unreachable from any root; parse them standalone.
    for node_id, raw in raw_nodes.items():
        if node_id in visited:
            continue
        ctx.parent = nodes[node_id].parent
        try:
            nodes[node_id].message = _parse_message(node_id, raw, {}, ctx)
        except Exception as exc:  # noqa: BLE001
            ctx.diagnostics.error(f"{ctx.where(node_id)}: could not parse message ({exc!r})")


# --------------------------------------------------------------------------- messages


def _parse_message(
    node_id: str, raw: dict[str, Any], canvases: dict[str, _Canvas], ctx: _Context
) -> Message | None:
    author = _dict(raw.get("author"))
    role_raw = _str(author.get("role")) or "assistant"
    name = _str(author.get("name"))
    content = _dict(raw.get("content"))
    ctype = _str(content.get("content_type")) or "text"
    meta = _dict(raw.get("metadata"))
    recipient = _str(raw.get("recipient")) or "all"

    _capture_instructions(content, meta, ctx)
    if meta.get("is_visually_hidden_from_conversation") is True:
        return None
    if role_raw == "system" or ctype in _HIDDEN_CONTENT:
        return None

    role: Role = role_raw if role_raw in ("user", "assistant", "tool") else "tool"
    blocks: list[Block]
    if role == "assistant" and recipient != "all":
        blocks = _tool_call(recipient, content, canvases, node_id, ctx)
        role = "tool"
    elif role == "tool":
        blocks = _tool_result(name, content, meta, node_id, ctx)
    elif ctype == "thoughts":
        blocks = _thoughts(content)
    elif role == "assistant" and (
        meta.get("is_thinking_preamble_message") is True
        or (ctype == "text" and node_id in ctx.preambles)
    ):
        # Progress notes written between reasoning steps; ChatGPT shows them
        # inside the thinking panel, not as part of the answer.
        if meta.get("is_thinking_preamble_message") is not True:
            ctx.diagnostics.count("inferred_preambles")
        note = _join_parts(content.get("parts")).strip()
        blocks = [ReasoningBlock(note)] if note else []
    elif ctype == "reasoning_recap":
        recap = _str(content.get("content"))
        blocks = [ReasoningBlock("", title=recap)] if recap else []
    else:
        if role == "assistant":
            content = _interactive_fallback(content, meta, ctx)
        blocks = _content_blocks(role, ctype, content, meta, node_id, ctx)
        if role == "user":
            blocks.extend(_metadata_attachments(meta, blocks))

    if not blocks:
        return None
    return Message(
        id=node_id,
        role=role,
        blocks=blocks,
        time=_time(raw.get("create_time")),
        model=_str(meta.get("model_slug")) or _str(meta.get("resolved_model_slug")),
        author_name=name,
        # Each run of a scheduled task is a reply of its own, with no prompt before it.
        scheduled=role == "assistant" and bool(_str(meta.get("automation_id"))),
    )


_ESCAPE_RE = re.compile(r"<escape>(.*?)</escape>", re.DOTALL)


def _interactive_fallback(
    content: dict[str, Any], meta: dict[str, Any], ctx: _Context
) -> dict[str, Any]:
    """Use the Markdown version of an interactive reply instead of its widget markup.

    Interactive replies keep JSX-like markup (``<box>``, ``DIL.useState(...)``) in
    ``parts`` and the Markdown ChatGPT falls back to in
    ``metadata.model_dil_v2.fallbackMarkdown``.
    """
    fallback = _dict(meta.get("model_dil_v2")).get("fallbackMarkdown")
    if not isinstance(fallback, str) or not fallback.strip():
        return content
    ctx.diagnostics.count("interactive_replies")
    # ``<escape>`` wraps a literal character, such as ``<`` in inline code.
    return {**content, "parts": [_ESCAPE_RE.sub(r"\1", fallback)]}


def _content_blocks(
    role: Role,
    ctype: str,
    content: dict[str, Any],
    meta: dict[str, Any],
    node_id: str,
    ctx: _Context,
) -> list[Block]:
    match ctype:
        case "text" | "multimodal_text":
            return _parts(role, content.get("parts"), meta, node_id, ctx)
        case "code":
            code = _str(content.get("text")) or _join_parts(content.get("parts"))
            return [CodeBlock(code, _str(content.get("language")) or "")] if code else []
        case "execution_output":
            text = _str(content.get("text"))
            return [OutputBlock(text)] if text else []
        case "tether_quote":
            text = _str(content.get("text"))
            if not text:
                return []
            return [QuoteBlock(text, _str(content.get("title")), _str(content.get("url")))]
        case _:
            ctx.diagnostics.unknown_kind(f"content_type:{ctype}", ctx.where(node_id))
            text = _generic_text(content)
            return [TextBlock(text)] if text else []


def _parts(
    role: Role, parts: Any, meta: dict[str, Any], node_id: str, ctx: _Context
) -> list[Block]:
    if not isinstance(parts, list):
        return []
    blocks: list[Block] = []
    text_buffer: list[str] = []

    def flush() -> None:
        text = "".join(text_buffer)
        text_buffer.clear()
        if not text.strip():
            return
        if role == "assistant":
            # Citations first: content_references index into the raw text.
            block = render_text(text, meta, ctx.diagnostics)
            for piece in split_directives(block.text):
                if isinstance(piece, Directive):
                    content = clean_assistant_text(strip_citations(piece.body))
                    blocks.append(DocumentBlock(piece.title, content, kind=piece.label))
                elif piece.strip():
                    blocks.append(TextBlock(clean_assistant_text(piece), block.sources))
        else:
            blocks.append(TextBlock(text))

    for part in parts:
        if isinstance(part, str):
            text_buffer.append(part)
            continue
        if not isinstance(part, dict):
            continue
        ptype = _str(part.get("content_type")) or ""
        if ptype in ("text", "audio_transcription") or (
            not ptype and isinstance(part.get("text"), str)
        ):
            text_buffer.append(_str(part.get("text")) or "")
            continue
        attachment = _asset_part(part, ptype)
        if attachment is not None:
            flush()
            blocks.append(attachment)
            continue
        if ptype == "real_time_user_audio_video_asset_pointer":
            for key in ("audio_asset_pointer", "video_container_asset_pointer"):
                inner = part.get(key)
                if isinstance(inner, dict):
                    nested = _asset_part(inner, _str(inner.get("content_type")) or key)
                    if nested is not None:
                        flush()
                        blocks.append(nested)
            continue
        ctx.diagnostics.unknown_kind(f"part:{ptype or '?'}", ctx.where(node_id))
    flush()
    return blocks


def _asset_part(part: dict[str, Any], ptype: str) -> AttachmentBlock | None:
    kind = _ASSET_PART_KIND.get(ptype)
    pointer = _str(part.get("asset_pointer"))
    if kind is None or not pointer:
        return None
    pmeta = _dict(part.get("metadata"))
    caption = _str(_dict(pmeta.get("dalle")).get("prompt")) or _str(
        _dict(pmeta.get("generation")).get("prompt")
    )
    return AttachmentBlock(
        Attachment(
            file_id=pointer.split("://", 1)[-1],
            kind=kind,
            width=part.get("width") if isinstance(part.get("width"), int) else None,
            height=part.get("height") if isinstance(part.get("height"), int) else None,
        ),
        caption=caption,
    )


def _metadata_attachments(meta: dict[str, Any], blocks: list[Block]) -> list[Block]:
    """Uploaded files listed in metadata but not already shown as parts."""
    shown = {b.attachment.file_id for b in blocks if isinstance(b, AttachmentBlock)}
    extra: list[Block] = []
    attachments = meta.get("attachments")
    if not isinstance(attachments, list):
        return extra
    named: dict[str, str] = {}
    for att in attachments:
        if not isinstance(att, dict):
            continue
        file_id = _str(att.get("id"))
        name = _str(att.get("name"))
        if not file_id:
            continue
        if name:
            named[file_id] = name
        if file_id in shown:
            continue
        mime = _str(att.get("mime_type"))
        extra.append(
            AttachmentBlock(Attachment(file_id, kind_for(mime, name), name=name, mime=mime))
        )
    # Give inline images their upload names, when known.
    for i, block in enumerate(blocks):
        if isinstance(block, AttachmentBlock) and block.attachment.file_id in named:
            blocks[i] = replace(
                block,
                attachment=replace(block.attachment, name=named[block.attachment.file_id]),
            )
    return extra


def _thoughts(content: dict[str, Any]) -> list[Block]:
    thoughts = content.get("thoughts")
    if not isinstance(thoughts, list):
        return []
    sections: list[str] = []
    for thought in thoughts:
        if not isinstance(thought, dict):
            continue
        summary = (_str(thought.get("summary")) or "").strip()
        body = (_str(thought.get("content")) or "").strip()
        if summary and body:
            sections.append(f"**{summary}**\n\n{body}")
        elif summary or body:
            sections.append(summary or body)
    text = "\n\n".join(sections)
    return [ReasoningBlock(text)] if text else []


# --------------------------------------------------------------------------- tools


def _tool_call(
    recipient: str,
    content: dict[str, Any],
    canvases: dict[str, _Canvas],
    node_id: str,
    ctx: _Context,
) -> list[Block]:
    payload = _str(content.get("text")) or _join_parts(content.get("parts"))
    if recipient in _CODE_TOOLS or recipient.startswith("python"):
        if not payload:
            return []
        return [CodeBlock(payload, _str(content.get("language")) or "python", recipient)]
    if recipient == "canmore.create_textdoc":
        return _canvas_create(payload, canvases, node_id, ctx)
    if recipient == "canmore.update_textdoc":
        return _canvas_update(payload, canvases, node_id, ctx)
    # Search queries, image prompts, memory writes, research plans: plumbing.
    ctx.diagnostics.count(f"tool_call:{recipient}")
    if payload.lstrip().startswith("{"):
        prompt = _str(_dict(_load_payload(payload)).get("prompt"))
        if prompt and prompt.strip():
            ctx.prompts[node_id] = prompt.strip()
    return []


def _tool_result(
    name: str | None,
    content: dict[str, Any],
    meta: dict[str, Any],
    node_id: str,
    ctx: _Context,
) -> list[Block]:
    ctype = _str(content.get("content_type")) or "text"
    tool = name or "tool"
    if ctype == "execution_output":
        text = _str(content.get("text"))
        blocks: list[Block] = [OutputBlock(text, tool)] if text else []
        blocks.extend(_images_from_aggregate(meta))
        return blocks
    if ctype == "multimodal_text":
        # Generated images and charts: keep the pictures, drop the status chatter.
        parts = content.get("parts")
        prompt = ctx.prompts.get(ctx.parent or "")
        return [
            b if b.caption or not prompt else replace(b, caption=prompt)
            for b in _parts("tool", parts, meta, node_id, ctx)
            if isinstance(b, AttachmentBlock)
        ]
    if ctype == "tether_quote":
        return _content_blocks("tool", ctype, content, meta, node_id, ctx)
    if tool == "bio":
        return [NoticeBlock("Memory updated")]
    ctx.diagnostics.count(f"tool_result:{tool}")
    return []


def _images_from_aggregate(meta: dict[str, Any]) -> list[Block]:
    """Charts produced by code execution (``aggregate_result.messages``)."""
    result = _dict(meta.get("aggregate_result"))
    messages = result.get("messages")
    blocks: list[Block] = []
    if not isinstance(messages, list):
        return blocks
    for item in messages:
        if isinstance(item, dict) and item.get("message_type") == "image":
            pointer = _str(item.get("image_url"))
            if pointer:
                file_id = pointer.split("://", 1)[-1]
                blocks.append(AttachmentBlock(Attachment(file_id, "image")))
    return blocks


def _load_payload(payload: str) -> dict[str, Any] | None:
    try:
        data = orjson.loads(payload)
    except orjson.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _canvas_create(
    payload: str, canvases: dict[str, _Canvas], node_id: str, ctx: _Context
) -> list[Block]:
    data = _load_payload(payload)
    if data is None or not isinstance(data.get("content"), str):
        ctx.diagnostics.unknown_kind("canvas:create-payload", ctx.where(node_id))
        return []
    canvas = _Canvas(
        title=_str(data.get("name")) or "Untitled document",
        content=data["content"],
        doc_type=_str(data.get("type")),
    )
    canvases["__current__"] = canvas
    canvases[canvas.title] = canvas
    return [DocumentBlock(canvas.title, canvas.content, canvas.doc_type, canvas.version)]


def _canvas_update(
    payload: str, canvases: dict[str, _Canvas], node_id: str, ctx: _Context
) -> list[Block]:
    data = _load_payload(payload)
    current = canvases.get("__current__")
    updates = data.get("updates") if data else None
    if current is None or not isinstance(updates, list):
        ctx.diagnostics.unknown_kind("canvas:update-without-document", ctx.where(node_id))
        return []
    content = current.content
    for update in updates:
        if not isinstance(update, dict):
            continue
        pattern = _str(update.get("pattern"))
        replacement = update.get("replacement")
        if pattern is None or not isinstance(replacement, str):
            continue
        content = _apply_canvas_update(
            content, pattern, replacement, multiple=update.get("multiple") is True
        )
    updated = _Canvas(current.title, content, current.doc_type, current.version + 1)
    canvases["__current__"] = updated
    canvases[updated.title] = updated
    return [DocumentBlock(updated.title, updated.content, updated.doc_type, updated.version)]


def _apply_canvas_update(content: str, pattern: str, replacement: str, *, multiple: bool) -> str:
    """Apply one Canvas edit: a regex ``pattern`` and its ``replacement``.

    Which regex dialect ChatGPT uses isn't documented (reports say Python, others
    ECMAScript), so the default flags shared by both are used (``.`` doesn't match
    newlines), and replacement references are understood in both spellings.
    """
    if pattern in (".*", "^.*$", "[\\s\\S]*", "(?s).*", "^[\\s\\S]*$"):
        return replacement  # the usual "rewrite everything"
    count = 0 if multiple else 1
    try:
        compiled = re.compile(pattern)
    except re.error:
        return content.replace(pattern, replacement, -1 if multiple else 1)
    return compiled.sub(lambda match: _expand(replacement, match), content, count=count)


# \g<1> \g<name> \1 (Python) and $1 $& $$ $<name> (ECMAScript).
_REFERENCE_RE = re.compile(r"\\g<(\w+)>|\\(\d{1,2})|\$(\d{1,2}|&|\$)|\$<(\w+)>")


def _expand(template: str, match: re.Match[str]) -> str:
    """Fill group references in ``template``; every other character is literal."""

    def reference(ref: re.Match[str]) -> str:
        name = ref.group(1) or ref.group(2) or ref.group(3) or ref.group(4)
        if name == "$":
            return "$"
        if name == "&":
            return match.group(0)
        try:
            return match.group(int(name) if name.isdigit() else name) or ""
        except (IndexError, re.error):
            return ref.group(0)  # not a group of this pattern: keep the text

    return _REFERENCE_RE.sub(reference, template)


# --------------------------------------------------------------------------- helpers


def _capture_instructions(content: dict[str, Any], meta: dict[str, Any], ctx: _Context) -> None:
    """Remember custom instructions (shown once per conversation, never inline)."""
    if content.get("content_type") == "user_editable_context":
        for key in ("user_profile", "user_instructions"):
            value = _str(content.get(key))
            if value and value.strip():
                ctx.custom_instructions[key] = value.strip()
    data = meta.get("user_context_message_data")
    if meta.get("is_user_system_message") is True and isinstance(data, dict):
        for key, target in (
            ("about_user_message", "user_profile"),
            ("about_model_message", "user_instructions"),
        ):
            value = _str(data.get(key))
            if value and value.strip():
                ctx.custom_instructions.setdefault(target, value.strip())


def _generic_text(content: dict[str, Any]) -> str:
    for key in ("text", "result", "content"):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return _join_parts(content.get("parts"))


def _join_parts(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    texts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "".join(texts)
