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

from convoviz.chatgpt import load_export
from convoviz.chatgpt.files import iter_raw_conversations
from convoviz.diagnostics import Diagnostics

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
    }
)


def census(sources: Sequence[ExportSource]) -> dict[str, Any]:
    """Collect the anonymous structure report."""
    files: Counter[str] = Counter(normalize_member_name(m.path) for s in sources for m in s)
    conv_keys: Counter[str] = Counter()
    node_keys: Counter[str] = Counter()
    message_keys: Counter[str] = Counter()
    metadata_keys: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    content_types: Counter[str] = Counter()
    part_types: Counter[str] = Counter()
    recipients: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    reference_types: Counter[str] = Counter()
    marker_kinds: Counter[str] = Counter()
    shapes: Counter[str] = Counter()

    diagnostics = Diagnostics()
    raws = (raw for source in sources for _, raw in iter_raw_conversations(source, diagnostics))
    for raw in raws:
        shapes["conversations"] += 1
        conv_keys.update(_ident(k) for k in raw)
        mapping = raw.get("mapping")
        if not isinstance(mapping, dict):
            shapes["conversations_without_mapping"] += 1
            continue
        for node in mapping.values():
            if not isinstance(node, dict):
                shapes["non_object_nodes"] += 1
                continue
            shapes["nodes"] += 1
            node_keys.update(_ident(k) for k in node)
            message = node.get("message")
            if not isinstance(message, dict):
                continue
            shapes["messages"] += 1
            message_keys.update(_ident(k) for k in message)
            author = message.get("author") if isinstance(message.get("author"), dict) else {}
            roles[_ident(author.get("role"))] += 1
            content = message.get("content") if isinstance(message.get("content"), dict) else {}
            content_types[_ident(content.get("content_type"))] += 1
            for part in content.get("parts") or []:
                if isinstance(part, dict):
                    part_types[_ident(part.get("content_type"))] += 1
                else:
                    part_types[f"<{type(part).__name__}>"] += 1
                    if isinstance(part, str):
                        marker_kinds.update(_ident(k) for k in _MARKER_KIND_RE.findall(part))
            recipient = message.get("recipient")
            if recipient not in (None, "all"):
                recipients[_ident(recipient)] += 1
            if "channel" in message:
                channels[f"{_ident(author.get('role'))}:{_ident(message.get('channel'))}"] += 1
            meta = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            metadata_keys.update(_ident(k) for k in meta)
            for ref in meta.get("content_references") or []:
                if isinstance(ref, dict):
                    reference_types[_ident(ref.get("type"))] += 1

    parse_diagnostics = Diagnostics()
    loaded = load_export(sources, parse_diagnostics)
    return {
        "convoviz": _version(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "files": dict(sorted(files.items())),
        "shape": dict(shapes),
        "conversation_keys": _top(conv_keys),
        "node_keys": _top(node_keys),
        "message_keys": _top(message_keys),
        "metadata_keys": _top(metadata_keys, 80),
        "author_roles": _top(roles),
        "content_types": _top(content_types),
        "part_types": _top(part_types),
        "recipients": _top(recipients),
        "channels": _top(channels),
        "content_reference_types": _top(reference_types),
        "inline_marker_kinds": _top(marker_kinds),
        "parsed": {
            "conversations": len(loaded.conversations),
            "attachments_indexed": len(loaded.assets.by_member),
            "errors": len(parse_diagnostics.errors),
            "warnings": len(parse_diagnostics.warnings),
            "skipped": len(parse_diagnostics.skipped),
            "unknown": _top(Counter(_anonymous_kinds(parse_diagnostics.unknown))),
            "counters": dict(sorted(parse_diagnostics.counters.items())),
        },
    }


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
