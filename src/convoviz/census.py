"""An anonymous census of an export's *structure*, for bug reports.

The export format is undocumented and changes often. When convoviz misreads an
export, this report shows maintainers what the export looks like without
revealing anything in it: only field names, type tags and counts are kept,
never titles, message text, URLs, file names or ids.
"""

from __future__ import annotations

import platform
import re
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from convoviz.diagnostics import Diagnostics
from convoviz.export import iter_raw_conversations
from convoviz.load import load_export

if TYPE_CHECKING:
    from collections.abc import Sequence

    from convoviz.source import ExportSource

# Only identifier-like strings are ever reported.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:\-]{0,63}$")
_MARKER_KIND_RE = re.compile(r"\ue200([A-Za-z_]{1,32})\ue202")


def _ident(value: object) -> str:
    if isinstance(value, str) and _IDENT_RE.match(value) and not _looks_like_id(value):
        return value
    return "<other>" if isinstance(value, str) else f"<{type(value).__name__}>"


def _looks_like_id(value: str) -> bool:
    return bool(re.search(r"[0-9a-f]{8}-[0-9a-f]{4}|file[-_][A-Za-z0-9]{6,}|[0-9]{6,}", value))


def normalize_member_name(path: str) -> str:
    """``conversations-007.json`` → ``conversations-N.json``; ids and names → ``*``."""
    parts = path.split("/")
    out = []
    for part in parts:
        stem, dot, ext = part.rpartition(".")
        if not dot:
            stem, ext = part, ""
        if re.fullmatch(r"conversations-\d+", stem):
            stem = "conversations-N"
        elif stem in _KNOWN_STEMS:
            pass
        elif re.match(r"file[-_]", stem):
            stem = "file_*"
        else:
            stem = "*"
        ext = ext.lower() if re.fullmatch(r"[A-Za-z0-9]{1,8}", ext) else ("*" if ext else "")
        out.append(f"{stem}.{ext}" if ext else stem)
    return "/".join(out)


_KNOWN_STEMS = frozenset(
    {
        "conversations",
        "export_manifest",
        "conversation_asset_file_names",
        "library_files",
        "library-files",
        "user_settings",
        "ads",
        "sectioned_conversations",
        "shared_conversations",
        "message_feedback",
        "model_comparisons",
        "user",
        "users",
        "chat",
        "shopping",
        "group_chats",
        "dalle-generations",
        "audio",
        "projects",
        "memories",
        "login_history",
    }
)


def census(sources: Sequence[ExportSource]) -> dict[str, Any]:
    """Collect the anonymous structure report: one section per provider found."""
    files: Counter[str] = Counter(normalize_member_name(m.path) for s in sources for m in s)
    chatgpt = _ChatGPTCensus()
    claude = _ClaudeCensus()
    diagnostics = Diagnostics()
    for source in sources:
        for _, provider, raw in iter_raw_conversations(source, diagnostics):
            (chatgpt if provider == "chatgpt" else claude).add(raw)

    parse_diagnostics = Diagnostics()
    loaded = load_export(sources, parse_diagnostics)
    report: dict[str, Any] = {
        "convoviz": _version(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "files": dict(sorted(files.items())),
    }
    if chatgpt.shapes["conversations"]:
        report["chatgpt"] = chatgpt.report()
    if claude.shapes["conversations"]:
        report["claude"] = claude.report()
    report["parsed"] = {
        "conversations": len(loaded.conversations),
        "attachments_indexed": len(loaded.assets.by_member),
        "errors": len(parse_diagnostics.errors),
        "warnings": len(parse_diagnostics.warnings),
        "skipped": len(parse_diagnostics.skipped),
        "unknown": _top(Counter(_anonymous_kinds(parse_diagnostics.unknown))),
        "counters": dict(sorted(_anonymous_kinds(parse_diagnostics.counters).items())),
    }
    return report


class _ChatGPTCensus:
    """Keys and type tags of ChatGPT's ``mapping`` trees."""

    def __init__(self) -> None:
        self.shapes: Counter[str] = Counter()
        self.counters: dict[str, Counter[str]] = {
            name: Counter()
            for name in (
                "conversation_keys",
                "node_keys",
                "message_keys",
                "metadata_keys",
                "author_roles",
                "content_types",
                "part_types",
                "recipients",
                "channels",
                "content_reference_types",
                "inline_marker_kinds",
            )
        }

    def add(self, raw: dict[str, Any]) -> None:
        c = self.counters
        self.shapes["conversations"] += 1
        c["conversation_keys"].update(_ident(k) for k in raw)
        mapping = raw.get("mapping")
        if not isinstance(mapping, dict):
            self.shapes["conversations_without_mapping"] += 1
            return
        for node in mapping.values():
            if not isinstance(node, dict):
                self.shapes["non_object_nodes"] += 1
                continue
            self.shapes["nodes"] += 1
            c["node_keys"].update(_ident(k) for k in node)
            message = node.get("message")
            if isinstance(message, dict):
                self._message(message)

    def _message(self, message: dict[str, Any]) -> None:
        c = self.counters
        self.shapes["messages"] += 1
        c["message_keys"].update(_ident(k) for k in message)
        author = _obj(message.get("author"))
        c["author_roles"][_ident(author.get("role"))] += 1
        content = _obj(message.get("content"))
        c["content_types"][_ident(content.get("content_type"))] += 1
        for part in content.get("parts") or []:
            if isinstance(part, dict):
                c["part_types"][_ident(part.get("content_type"))] += 1
            else:
                c["part_types"][f"<{type(part).__name__}>"] += 1
                if isinstance(part, str):
                    c["inline_marker_kinds"].update(
                        _ident(k) for k in _MARKER_KIND_RE.findall(part)
                    )
        recipient = message.get("recipient")
        if recipient not in (None, "all"):
            c["recipients"][_ident(recipient)] += 1
        if "channel" in message:
            c["channels"][f"{_ident(author.get('role'))}:{_ident(message.get('channel'))}"] += 1
        meta = _obj(message.get("metadata"))
        c["metadata_keys"].update(_ident(k) for k in meta)
        for ref in meta.get("content_references") or []:
            if isinstance(ref, dict):
                c["content_reference_types"][_ident(ref.get("type"))] += 1

    def report(self) -> dict[str, Any]:
        limits = {"metadata_keys": 80}
        return {
            "shape": dict(self.shapes),
            **{name: _top(c, limits.get(name, 50)) for name, c in self.counters.items()},
        }


class _ClaudeCensus:
    """Keys and type tags of claude.ai's ``chat_messages`` and their content blocks.

    Connector (MCP) tool names are reported only as ``<mcp>``: a server's name
    can say where someone works.
    """

    def __init__(self) -> None:
        self.shapes: Counter[str] = Counter()
        self.counters: dict[str, Counter[str]] = {
            name: Counter()
            for name in (
                "conversation_keys",
                "message_keys",
                "senders",
                "block_types",
                "block_keys",
                "tool_names",
                "tool_input_keys",
                "tool_result_item_types",
                "citation_detail_types",
                "attachment_keys",
                "attachment_file_types",
                "file_keys",
            )
        }

    def add(self, raw: dict[str, Any]) -> None:
        c = self.counters
        self.shapes["conversations"] += 1
        c["conversation_keys"].update(_ident(k) for k in raw)
        messages = raw.get("chat_messages", raw.get("messages"))
        parents: Counter[str] = Counter()
        for message in messages if isinstance(messages, list) else ():
            if not isinstance(message, dict):
                self.shapes["non_object_messages"] += 1
                continue
            self._message(message)
            parent = message.get("parent_message_uuid")
            if isinstance(parent, str):
                parents[parent] += 1
        if any(n > 1 for n in parents.values()):
            self.shapes["conversations_with_branches"] += 1

    def _message(self, message: dict[str, Any]) -> None:
        c = self.counters
        self.shapes["messages"] += 1
        c["message_keys"].update(_ident(k) for k in message)
        c["senders"][_ident(message.get("sender"))] += 1
        for attachment in _objects(message.get("attachments")):
            c["attachment_keys"].update(_ident(k) for k in attachment)
            c["attachment_file_types"][_ident(attachment.get("file_type"))] += 1
        for file in _objects(message.get("files")):
            c["file_keys"].update(_ident(k) for k in file)
        for block in _objects(message.get("content")):
            kind = _ident(block.get("type"))
            c["block_types"][kind] += 1
            c["block_keys"].update(f"{kind}.{_ident(k)}" for k in block)
            if kind == "tool_use":
                name = _tool_ident(block.get("name"))
                c["tool_names"][name] += 1
                if name != "<mcp>":
                    c["tool_input_keys"].update(
                        f"{name}.{_ident(k)}" for k in _obj(block.get("input"))
                    )
            elif kind == "tool_result":
                for item in _objects(block.get("content")):
                    c["tool_result_item_types"][_ident(item.get("type"))] += 1
            for citation in _objects(block.get("citations")):
                details = _obj(citation.get("details"))
                c["citation_detail_types"][_ident(details.get("type"))] += 1

    def report(self) -> dict[str, Any]:
        return {
            "shape": dict(self.shapes),
            **{
                name: _top(c, 80 if name == "block_keys" else 50)
                for name, c in self.counters.items()
            },
        }


def _tool_ident(name: object) -> str:
    if isinstance(name, str) and (
        name.startswith("mcp__") or (":" in name and not name.startswith("visualize:"))
    ):
        return "<mcp>"
    return _ident(name)


def _obj(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _objects(value: object) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _anonymous_kinds(unknown: Counter[str]) -> Counter[str]:
    """``content_type:<value>`` keys carry export values: keep only identifier-like ones."""
    out: Counter[str] = Counter()
    for key, n in unknown.items():
        prefix, sep, value = key.partition(":")
        out[f"{prefix}:{_ident(value)}" if sep else _ident(key)] += n
    return out


def _top(counter: Counter[str], limit: int = 50) -> dict[str, int]:
    return dict(counter.most_common(limit))


def _version() -> str:
    try:
        return version("convoviz")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        return "unknown"
