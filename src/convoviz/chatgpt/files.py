"""Locate and load the data files of a ChatGPT export.

Layouts seen in 2026:

* ``conversations.json`` (small accounts) or ``conversations-000.json`` ... shards
  (large accounts), described by ``export_manifest.json``: either
  ``logical_files`` or an ``export_files`` inventory of ``{path, size_bytes}``.
* Attachments at the root as ``file_<hex>.dat`` / ``file-<id>.dat``, with original
  names in ``conversation_asset_file_names.json`` and richer metadata in
  ``library_files.json``.
* Legacy layouts: ``file-<id>-<name>.<ext>`` at the root, ``dalle-generations/``
  and ``user-<id>/`` folders.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import orjson

from convoviz.source import CONVERSATION_FILE_RE

if TYPE_CHECKING:
    from collections.abc import Iterator

    from convoviz.diagnostics import Diagnostics
    from convoviz.source import ExportSource, Member

_SHARD_NUMBER_RE = re.compile(r"(\d+)")


def load_json(member: Member, diagnostics: Diagnostics) -> Any:
    """Parse a JSON member, repairing broken Unicode if strict parsing fails.

    Some exports contain emoji encoded as separately-encoded UTF-16 surrogate
    halves (CESU-8) or lone surrogate escapes. Rather than failing the run, those
    are repaired and the file is flagged in the diagnostics.
    """
    raw = member.read_bytes()
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
        candidates = [m for m in source if m.name.endswith(".json") and "/" not in m.path]
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


def iter_raw_conversations(
    source: ExportSource, diagnostics: Diagnostics
) -> Iterator[tuple[Member, dict[str, Any]]]:
    """Yield every conversation-shaped object, file by file."""
    for member in conversation_files(source, diagnostics):
        data = load_json(member, diagnostics)
        if isinstance(data, dict) and isinstance(data.get("conversations"), list):
            data = data["conversations"]
        if isinstance(data, dict) and isinstance(data.get("mapping"), dict):
            data = [data]  # a single conversation object
        if not isinstance(data, list):
            if data is not None:
                diagnostics.error(f"{member.path}: unexpected top-level JSON shape")
            continue
        for index, item in enumerate(data):
            if isinstance(item, dict) and isinstance(item.get("mapping"), dict):
                yield member, item
            elif item:
                diagnostics.skip(f"{member.path}[{index}]: not a conversation object")
            else:
                diagnostics.skip(f"{member.path}[{index}]: empty entry")


# --------------------------------------------------------------------------- assets

_FILE_ID_RE = re.compile(r"^(file[-_][A-Za-z0-9]+)")


@dataclass(slots=True)
class AssetInfo:
    """What the export tells us about one stored file."""

    member: Member
    original_name: str | None = None
    mime: str | None = None


@dataclass(frozen=True, slots=True)
class LibraryFile:
    """An entry of ``library_files.json`` that names where it came from."""

    file_id: str
    conversation_id: str | None
    message_id: str | None
    name: str | None
    mime: str | None
    artifact_type: str | None = None
    created: str = ""  # ISO 8601 or epoch digits, only used for ordering


@dataclass(slots=True)
class AssetIndex:
    """Resolve asset pointers (``sediment://file_…``) to export members."""

    by_id: dict[str, AssetInfo] = field(default_factory=dict)
    by_member: dict[str, AssetInfo] = field(default_factory=dict)
    library: list[LibraryFile] = field(default_factory=list)

    @classmethod
    def build(cls, source: ExportSource, diagnostics: Diagnostics) -> AssetIndex:
        """Index every file that could be an attachment."""
        index = cls()
        for member in source:
            if member.name.endswith(".json") or member.name == "chat.html":
                continue
            stem = PurePosixPath(member.name).stem if member.name.endswith(".dat") else member.name
            info = AssetInfo(member)
            index.by_member[member.path] = info
            match = _FILE_ID_RE.match(stem)
            keys = {stem}
            if match:
                keys.add(match.group(1))
            for key in keys:
                existing = index.by_id.get(key)
                # Prefer root-level files over folder copies, then shorter names.
                if existing is None or (
                    existing.member.path.count("/"),
                    len(existing.member.path),
                ) > (member.path.count("/"), len(member.path)):
                    index.by_id[key] = info
        index._apply_name_index(source, diagnostics)
        index._apply_library(source, diagnostics)
        return index

    def _apply_name_index(self, source: ExportSource, diagnostics: Diagnostics) -> None:
        member = root_file(source, "conversation_asset_file_names.json")
        if member is None:
            return
        data = load_json(member, diagnostics)
        if not isinstance(data, dict):
            return
        for stored, original in data.items():
            info = self.by_member.get(stored)
            if info is None or not isinstance(original, str) or not original:
                continue
            name = PurePosixPath(original).name
            if not _looks_generated_name(name):
                info.original_name = name
            elif "." in name:
                # e.g. dalle-generations/<uuid>.webp: only the extension is useful.
                info.mime = info.mime or _mime_from_name(name)

    def _apply_library(self, source: ExportSource, diagnostics: Diagnostics) -> None:
        member = root_file(source, "library_files.json")
        if member is None:
            return
        data = load_json(member, diagnostics)
        entries = (
            data
            if isinstance(data, list)
            else (
                data.get("files") or data.get("library_files") if isinstance(data, dict) else None
            )
        )
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("deleted_at") or entry.get("trashed_at"):
                continue
            file_id = entry.get("file_id")
            if not isinstance(file_id, str) or not file_id:
                continue
            raw_name, raw_mime = entry.get("file_name"), entry.get("mime_type")
            name = (
                PurePosixPath(raw_name).name
                if isinstance(raw_name, str) and raw_name and not _looks_generated_name(raw_name)
                else None
            )
            mime = raw_mime if isinstance(raw_mime, str) and raw_mime else None
            thread, message = (
                _nonempty(entry.get("origination_thread_id")),
                _nonempty(entry.get("origination_message_id")),
            )
            if thread or message:
                created = next(
                    (
                        str(value)
                        for key in ("created_at", "record_creation_time", "version_created_at")
                        if isinstance(value := entry.get(key), str | int | float) and value
                    ),
                    "",
                )
                self.library.append(
                    LibraryFile(
                        file_id,
                        thread,
                        message,
                        name,
                        mime,
                        _nonempty(entry.get("library_artifact_type")),
                        created,
                    )
                )
            info = self.by_id.get(file_id)
            if info is None:
                continue
            info.original_name = info.original_name or name
            info.mime = info.mime or mime

    def merge(self, other: AssetIndex) -> None:
        """Add ``other``'s files; entries already known here win."""
        for key, info in other.by_id.items():
            self.by_id.setdefault(key, info)
        for key, info in other.by_member.items():
            self.by_member.setdefault(key, info)
        known = {f.file_id for f in self.library}
        self.library.extend(f for f in other.library if f.file_id not in known)

    def resolve(self, pointer: str) -> AssetInfo | None:
        """Find the stored file for an asset pointer or file id."""
        file_id = pointer.split("://", 1)[-1].strip()
        if not file_id or "/" in file_id:
            file_id = PurePosixPath(file_id).name
        info = self.by_id.get(file_id)
        if info is not None:
            return info
        match = _FILE_ID_RE.match(file_id)
        return self.by_id.get(match.group(1)) if match else None


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?:\.\w+)?$", re.IGNORECASE
)


def _nonempty(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _looks_generated_name(name: str) -> bool:
    """True for names that carry no meaning for a human (UUIDs, file ids)."""
    return bool(_UUID_RE.match(name) or _FILE_ID_RE.match(name))


_EXT_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "mp4": "video/mp4",
    "pdf": "application/pdf",
}


def _mime_from_name(name: str) -> str | None:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return _EXT_MIME.get(ext)
