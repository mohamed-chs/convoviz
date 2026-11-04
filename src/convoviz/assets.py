"""Copy attachments out of the export under readable, stable names."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from convoviz.fsnames import slug

if TYPE_CHECKING:
    from convoviz.chatgpt.files import AssetIndex, AssetInfo
    from convoviz.diagnostics import Diagnostics
    from convoviz.model import Attachment

ATTACHMENTS_DIR = "attachments"

_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "image/heic": ".heic",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "application/pdf": ".pdf",
    "application/json": ".json",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/csv": ".csv",
}


def sniff_extension(head: bytes) -> str | None:
    """Guess a file extension from its first bytes (exports store many files as ``.dat``)."""
    signatures: list[tuple[bytes, int, str]] = [
        (b"\x89PNG\r\n\x1a\n", 0, ".png"),
        (b"\xff\xd8\xff", 0, ".jpg"),
        (b"GIF8", 0, ".gif"),
        (b"%PDF-", 0, ".pdf"),
        (b"ID3", 0, ".mp3"),
        (b"OggS", 0, ".ogg"),
        (b"\x1a\x45\xdf\xa3", 0, ".webm"),
        (b"fLaC", 0, ".flac"),
    ]
    for magic, offset, ext in signatures:
        if head[offset : offset + len(magic)] == magic:
            return ext
    if head[:4] == b"RIFF":
        return {b"WEBP": ".webp", b"WAVE": ".wav"}.get(head[8:12])
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand.startswith((b"heic", b"heix")):
            return ".heic"
        if brand.startswith((b"M4A", b"M4B")):
            return ".m4a"
        return ".mp4"
    if head[:2] == b"\xff\xfb" or head[:2] == b"\xff\xf3":
        return ".mp3"
    if head[:4] == b"PK\x03\x04":
        return ".zip"
    if head and b"\x00" not in head:
        try:
            head.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return ".txt"
    return None


def _extension(info: AssetInfo) -> str:
    if info.original_name:
        suffix = PurePosixPath(info.original_name).suffix.lower()
        if suffix and suffix != ".dat":
            return suffix
    stored = PurePosixPath(info.member.name).suffix.lower()
    if stored and stored != ".dat":
        return stored
    if info.mime and info.mime in _MIME_EXT:
        return _MIME_EXT[info.mime]
    return sniff_extension(info.member.head(16)) or ".bin"


class AssetStore:
    """Resolves attachments to files under ``<output>/attachments``, copying on demand."""

    def __init__(self, output: Path, index: AssetIndex, diagnostics: Diagnostics) -> None:
        self.output = output
        self.directory = output / ATTACHMENTS_DIR
        self.index = index
        self.diagnostics = diagnostics
        self._placed: dict[str, Path | None] = {}
        self.copied = 0
        self.missing = 0

    def path_for(self, attachment: Attachment) -> Path | None:
        """Absolute path of the copied file, or None when the export lacks it."""
        if attachment.file_id in self._placed:
            return self._placed[attachment.file_id]
        info = self.index.resolve(attachment.file_id)
        destination: Path | None = None
        if info is None:
            self.missing += 1
            self.diagnostics.count(f"missing_{attachment.kind}")
        else:
            destination = self._copy(attachment, info)
        self._placed[attachment.file_id] = destination
        return destination

    def link_from(self, note: Path, attachment: Attachment) -> str | None:
        """Relative link from ``note`` to the attachment (POSIX separators)."""
        target = self.path_for(attachment)
        if target is None:
            return None
        return Path(os.path.relpath(target, note.parent)).as_posix()

    def _copy(self, attachment: Attachment, info: AssetInfo) -> Path | None:
        name = info.original_name or attachment.name
        stem = slug(PurePosixPath(name).stem) if name else ""
        tail = attachment.file_id.rsplit("-", 1)[-1].rsplit("_", 1)[-1][-8:]
        filename = f"{stem or attachment.kind}_{tail}{_extension(info)}"
        destination = self.directory / filename
        try:
            if not destination.exists() or destination.stat().st_size != info.member.size:
                self.directory.mkdir(parents=True, exist_ok=True)
                info.member.copy_to(destination)
                self.copied += 1
        except OSError as exc:
            self.diagnostics.error(f"could not copy {info.member.path}: {exc}")
            return None
        return destination
