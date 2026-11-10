"""Find and load the conversation files of an export, whatever the provider.

ChatGPT puts conversations in ``conversations.json`` or in numbered shards
(``conversations-000.json`` …) described by ``export_manifest.json``: either
``logical_files`` or an ``export_files`` inventory of ``{path, size_bytes}``.
Claude puts them in ``conversations.json``, in its own ZIP since August 2026.
Both are JSON arrays of conversation objects; :func:`provider_of` tells them
apart by shape, so a file is never trusted by its name alone.
"""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

import orjson

from convoviz.source import CONVERSATION_FILE_RE

if TYPE_CHECKING:
    from collections.abc import Iterator

    from convoviz.diagnostics import Diagnostics
    from convoviz.source import ExportSource, Member

Provider = Literal["chatgpt", "claude"]

_SHARD_NUMBER_RE = re.compile(r"(\d+)")


def load_json(member: Member, diagnostics: Diagnostics) -> Any:
    """Parse a JSON member, repairing broken Unicode if strict parsing fails.

    Some exports contain emoji encoded as separately-encoded UTF-16 surrogate
    halves (CESU-8) or lone surrogate escapes. Rather than failing the run, those
    are repaired and the file is flagged in the diagnostics.
    """
    try:
        raw = member.read_bytes()
    except OSError as exc:
        diagnostics.error(f"could not read {member.path}: {exc}")
        return None
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError:
        pass
    try:
        text = raw.decode("utf-8", errors="surrogatepass")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        diagnostics.error(f"{member.path}: not valid JSON ({exc.msg}, line {exc.lineno})")
        return None
    diagnostics.warn(f"{member.path}: repaired invalid Unicode (e.g. broken emoji)")
    return _repair_strings(data)


def _repair_text(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        # Re-pair surrogate halves; anything still unpaired becomes U+FFFD.
        return value.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return value


def _repair_strings(value: Any) -> Any:
    if isinstance(value, str):
        return _repair_text(value)
    if isinstance(value, list):
        return [_repair_strings(v) for v in value]
    if isinstance(value, dict):
        return {_repair_text(k): _repair_strings(v) for k, v in value.items()}
    return value


def conversation_files(source: ExportSource, diagnostics: Diagnostics) -> list[Member]:
    """The conversation data files of an export, in canonical order."""
    manifest_files = _files_from_manifest(source, diagnostics)
    if manifest_files is not None:
        return manifest_files

    candidates = [m for m in source if CONVERSATION_FILE_RE.match(m.name)]
    if not candidates:
        # A bare JSON file (e.g. a single conversation saved by a browser tool).
        # Full paths: the one JSON file of a Claude ``projects-000.zip`` is in a folder.
        root = source.root_prefix
        candidates = [m for m in source if m.name.endswith(".json") and "/" not in root + m.path]
        if len(candidates) != 1:
            return []
        return candidates
    # Prefer the shallowest location; an export never mixes depths.
    depth = min(m.path.count("/") for m in candidates)
    candidates = [m for m in candidates if m.path.count("/") == depth]
    single = [m for m in candidates if m.name.lower() == "conversations.json"]
    shards = sorted((m for m in candidates if m not in single), key=lambda m: _shard_number(m.name))
    if single and shards:
        diagnostics.warn("Found both conversations.json and numbered shards; reading all of them.")
    return single + shards


def _shard_number(name: str) -> int:
    match = _SHARD_NUMBER_RE.search(name)
    return int(match.group(1)) if match else -1


def root_file(source: ExportSource, name: str) -> Member | None:
    """A top-level file, ignoring case and ``-`` vs ``_`` (both spellings are reported)."""
    exact = source.get(name)
    if exact is not None:
        return exact
    key = _name_key(name)
    return next((m for m in source if "/" not in m.path and _name_key(m.name) == key), None)


def _name_key(name: str) -> str:
    return name.lower().replace("-", "_")


def _files_from_manifest(source: ExportSource, diagnostics: Diagnostics) -> list[Member] | None:
    manifest_member = root_file(source, "export_manifest.json")
    if manifest_member is None:
        return None
    manifest = load_json(manifest_member, diagnostics)
    if not isinstance(manifest, dict):
        return None
    logical = manifest.get("logical_files")
    entry = logical.get("conversations.json") if isinstance(logical, dict) else None
    names = entry.get("files") if isinstance(entry, dict) else None
    if not names and isinstance(manifest.get("export_files"), list):
        # Inventory form: every file in the export, conversation shards among them.
        names = [
            item["path"]
            for item in manifest["export_files"]
            if isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and CONVERSATION_FILE_RE.match(PurePosixPath(item["path"]).name)
        ]
        names.sort(key=lambda n: _shard_number(PurePosixPath(n).name))
    if not isinstance(names, list) or not names:
        return None
    members: list[Member] = []
    for name in names:
        member = (
            source.get(name) or root_file(source, PurePosixPath(name).name)
            if isinstance(name, str)
            else None
        )
        if member is None:
            diagnostics.error(
                f"Export manifest lists {name!r}, but it is missing from the export. "
                "Conversations in it cannot be recovered."
            )
            continue
        members.append(member)
    _report_missing_files(source, manifest, diagnostics)
    count = entry.get("shard_count") if isinstance(entry, dict) else None
    if isinstance(count, int) and count != len(names):
        diagnostics.warn(
            f"Export manifest says {count} conversation shards but lists {len(names)}."
        )
    return members


def _report_missing_files(
    source: ExportSource, manifest: dict[str, Any], diagnostics: Diagnostics
) -> None:
    """Warn when an inventory manifest lists files (other than shards) the export lacks.

    A split download lists every part's files in one manifest, so a missing part
    shows up here as missing attachments.
    """
    inventory = manifest.get("export_files")
    if not isinstance(inventory, list):
        return
    missing, size = 0, 0
    for item in inventory:
        path = item.get("path") if isinstance(item, dict) else None
        if not isinstance(path, str) or CONVERSATION_FILE_RE.match(PurePosixPath(path).name):
            continue
        if source.get(path) is None and root_file(source, PurePosixPath(path).name) is None:
            missing += 1
            size += item["size_bytes"] if isinstance(item.get("size_bytes"), int) else 0
    if missing:
        amount = f" ({size / 1_000_000:.1f} MB)" if size else ""
        diagnostics.warn(
            f"The export manifest lists {missing} files that are not in the export{amount}. "
            "If your download came in several parts, pass them all; their attachments "
            "will show as missing otherwise."
        )


def provider_of(item: object) -> Provider | None:
    """Which provider's conversation ``item`` is, judged by its shape."""
    if not isinstance(item, dict):
        return None
    if isinstance(item.get("mapping"), dict):
        return "chatgpt"
    messages = item.get("chat_messages", item.get("messages"))
    if isinstance(messages, list) and ("uuid" in item or "chat_messages" in item):
        return "claude"
    return None


def iter_raw_conversations(
    source: ExportSource, diagnostics: Diagnostics
) -> Iterator[tuple[Member, Provider, dict[str, Any]]]:
    """Yield every conversation object with its provider, file by file."""
    for member in conversation_files(source, diagnostics):
        data = load_json(member, diagnostics)
        if isinstance(data, dict) and isinstance(data.get("conversations"), list):
            data = data["conversations"]
        if provider_of(data) is not None:
            data = [data]  # a single conversation object
        if not isinstance(data, list):
            if data is not None:
                diagnostics.error(f"{member.path}: unexpected top-level JSON shape")
            continue
        for index, item in enumerate(data):
            provider = provider_of(item)
            if provider is not None:
                yield member, provider, item
            elif item:
                diagnostics.skip(f"{member.path}[{index}]: not a conversation object")
            else:
                diagnostics.skip(f"{member.path}[{index}]: empty entry")
