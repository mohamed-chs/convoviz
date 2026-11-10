"""Locate the attachments of a ChatGPT export.

Layouts seen in 2026 (conversation files are found by ``convoviz.export``):

* Attachments at the root as ``file_<hex>.dat`` / ``file-<id>.dat``, with original
  names in ``conversation_asset_file_names.json`` and richer metadata in
  ``library_files.json``.
* Legacy layouts: ``file-<id>-<name>.<ext>`` at the root, ``dalle-generations/``
  and ``user-<id>/`` folders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from convoviz.export import load_json, root_file

if TYPE_CHECKING:
    from convoviz.diagnostics import Diagnostics
    from convoviz.source import ExportSource, Member

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
