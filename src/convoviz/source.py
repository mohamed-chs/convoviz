"""Read-only access to an export, wherever it lives.

An export can arrive as a ZIP, an extracted folder, a ZIP nested inside another ZIP
(the Privacy Portal path), several ``part-N.zip`` archives that together form one
export, or a single JSON file. :class:`ExportSource` hides all of that behind a flat
namespace of POSIX-style member paths.

Nothing is ever extracted to its original member path, so ZIP path-traversal tricks
have nothing to exploit: member names are only used as dictionary keys.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator

_NESTED_ZIP_MAX_DEPTH = 2


class SourceError(Exception):
    """The input cannot be read as an export."""


class DamagedMemberError(OSError):
    """A file inside the export can't be read back (a corrupt or cut-short ZIP)."""


# What zipfile raises for corrupt data: bad CRC, broken deflate stream, early end.
_DAMAGED = (zipfile.BadZipFile, zlib.error, EOFError)


@dataclass(frozen=True, slots=True)
class Member:
    """One file inside an export."""

    path: str  # POSIX path relative to the export root
    size: int
    _origin: _Origin = field(repr=False, compare=False)
    _key: str = field(default="", repr=False, compare=False)

    @property
    def key(self) -> str:
        """Identifier the origin uses to open this member."""
        return self._key or self.path

    @property
    def name(self) -> str:
        """Final path component."""
        return PurePosixPath(self.path).name

    def open(self) -> IO[bytes]:
        """Open the member for binary reading."""
        return self._origin.open(self)

    def read_bytes(self) -> bytes:
        """Read the whole member."""
        with self._reading(), self.open() as handle:
            return handle.read()

    def head(self, n: int = 64) -> bytes:
        """Read the first ``n`` bytes (for type sniffing)."""
        with self._reading(), self.open() as handle:
            return handle.read(n)

    def copy_to(self, dest: Path) -> None:
        """Stream the member into ``dest`` (nothing is left behind if that fails)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._reading(), self.open() as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        except BaseException:
            dest.unlink(missing_ok=True)
            raise

    @contextmanager
    def _reading(self) -> Generator[None]:
        try:
            yield
        except _DAMAGED as exc:
            raise DamagedMemberError("damaged in the archive; download the export again") from exc


class _Origin:
    def open(self, member: Member) -> IO[bytes]:  # pragma: no cover - interface
        raise NotImplementedError


class _DirOrigin(_Origin):
    def __init__(self, root: Path) -> None:
        self.root = root

    def open(self, member: Member) -> IO[bytes]:
        return (self.root / member.key).open("rb")

    def members(self) -> Iterator[Member]:
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                rel = path.relative_to(self.root).as_posix()
                yield Member(rel, path.stat().st_size, self)


class _ZipOrigin(_Origin):
    def __init__(self, archive: zipfile.ZipFile, label: str) -> None:
        self.archive = archive
        self.label = label
        self._infos: dict[str, zipfile.ZipInfo] = {}

    def open(self, member: Member) -> IO[bytes]:
        return self.archive.open(self._infos[member.key])

    def members(self) -> Iterator[Member]:
        for info in self.archive.infolist():
            if info.is_dir():
                continue
            path = _clean_member_path(info.filename)
            if path is None:
                continue
            self._infos[path] = info
            yield Member(path, info.file_size, self)


class _FileOrigin(_Origin):
    def __init__(self, path: Path) -> None:
        self.path = path

    def open(self, member: Member) -> IO[bytes]:  # noqa: ARG002
        return self.path.open("rb")


def _clean_member_path(name: str) -> str | None:
    """Normalize a ZIP member name; drop anything that isn't a plain relative path."""
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or ".." in parts or parts[0].endswith(":"):
        return None
    if parts[0] == "__MACOSX" or parts[-1] == ".DS_Store":
        return None
    return "/".join(parts)


_PART_RE = re.compile(r"(?i)^(?P<prefix>.*?)[-_ ]?part[-_ ]?(?P<number>\d+)\.zip$")
_NESTED_EXPORT_RE = re.compile(r"(?i)\.zip$")


class ExportSource:
    """A merged, read-only view over one or more export inputs.

    Later inputs never shadow earlier ones: when two inputs contain the same member
    path, the first one wins. Use separate sources for separate exports.
    """

    def __init__(self, members: Iterable[Member], labels: list[str]) -> None:
        self._members: dict[str, Member] = {}
        for member in members:
            self._members.setdefault(member.path, member)
        self.labels = labels
        self._root_prefix = _common_wrapper_dir(self._members)
        if self._root_prefix:
            self._members = {
                path.removeprefix(self._root_prefix): replace(
                    m, path=path.removeprefix(self._root_prefix), _key=m.key
                )
                for path, m in self._members.items()
            }

    # -- construction -----------------------------------------------------------

    @classmethod
    def open(cls, path: Path, workdir: Path) -> ExportSource:
        """Open a single input path (ZIP, folder, or JSON file).

        Args:
            path: The user-supplied input.
            workdir: Scratch space for unpacking nested archives. Must outlive
                the returned source.
        """
        if not path.exists():
            raise SourceError(f"Not found: {path}")
        if path.is_dir():
            origin = _DirOrigin(path)
            source = cls(origin.members(), [str(path)])
            return source.unnest(workdir)
        if path.suffix.lower() == ".json":
            origin = _FileOrigin(path)
            return cls([Member(path.name, path.stat().st_size, origin)], [str(path)])
        if zipfile.is_zipfile(path):
            try:
                archive = zipfile.ZipFile(path)
            except zipfile.BadZipFile as exc:
                raise SourceError(f"Corrupt ZIP archive: {path}") from exc
            origin = _ZipOrigin(archive, str(path))
            source = cls(origin.members(), [str(path)])
            return source.unnest(workdir)
        if path.suffix.lower() == ".zip":
            raise SourceError(
                f"{path} isn't a complete ZIP archive. The download may have been cut "
                "short: download the export again."
            )
        raise SourceError(f"Not a ZIP archive, folder, or JSON file: {path}")

    @classmethod
    def open_many(cls, paths: list[Path], workdir: Path) -> ExportSource:
        """Open several inputs that together form a single export (e.g. part ZIPs)."""
        sources = [cls.open(p, workdir / f"in{i}") for i, p in enumerate(paths)]
        members = [m for s in sources for m in s]
        return cls(members, [label for s in sources for label in s.labels])

    def unnest(self, workdir: Path, depth: int = 0) -> ExportSource:
        """Expand ZIPs inside the export when the export proper lives in them.

        Handles the Privacy Portal layout (``Conversations__*.zip`` inside the outer
        archive) and multi-part layouts (``part-1.zip``, ``part-2.zip`` ...). Only
        triggers when no conversation data is visible at the current level, so
        user-uploaded ZIP attachments are never unpacked.
        """
        if depth >= _NESTED_ZIP_MAX_DEPTH or self.has_conversation_data():
            return self
        nested = [m for m in self if _NESTED_EXPORT_RE.search(m.name)]
        if not nested:
            return self
        merged: list[Member] = [m for m in self if m not in nested]
        labels = list(self.labels)
        for index, member in enumerate(sorted(nested, key=_part_sort_key)):
            target = workdir / f"nested-{depth}-{index}.zip"
            target.parent.mkdir(parents=True, exist_ok=True)
            member.copy_to(target)
            if not zipfile.is_zipfile(target):
                continue
            origin = _ZipOrigin(zipfile.ZipFile(target), f"{labels[0]}!{member.path}")
            merged.extend(origin.members())
            labels.append(origin.label)
        return ExportSource(merged, labels).unnest(workdir, depth + 1)

    # -- queries ----------------------------------------------------------------

    def __iter__(self) -> Iterator[Member]:
        return iter(self._members.values())

    def __len__(self) -> int:
        return len(self._members)

    def get(self, path: str) -> Member | None:
        """Look up a member by exact path."""
        return self._members.get(path)

    def has_conversation_data(self) -> bool:
        """Whether this looks like it holds conversations (ChatGPT or Claude)."""
        return any(_is_conversation_file(m.path) for m in self)

    @property
    def label(self) -> str:
        """Human-readable description of where the data came from."""
        return self.labels[0] if self.labels else "<unknown>"


def _part_sort_key(member: Member) -> tuple[int, str]:
    numbers = re.findall(r"\d+", member.name)
    return (int(numbers[-1]) if numbers else 0, member.name)


CONVERSATION_FILE_RE = re.compile(r"(?i)^conversations(?:-\d+)?\.json$")


def _is_conversation_file(path: str) -> bool:
    name = PurePosixPath(path).name
    return CONVERSATION_FILE_RE.match(name) is not None or name == "export_manifest.json"


def _common_wrapper_dir(members: dict[str, Member]) -> str:
    """Return ``"dir/"`` if every member sits inside one top-level folder."""
    tops = {path.split("/", 1)[0] for path in members}
    if len(tops) != 1:
        return ""
    (top,) = tops
    if all("/" in path for path in members):
        return top + "/"
    return ""


def make_workdir() -> tempfile.TemporaryDirectory[str]:
    """Create a scratch directory for nested-archive extraction."""
    return tempfile.TemporaryDirectory(prefix="convoviz-")


# --------------------------------------------------------------------------- discovery


def with_sibling_parts(path: Path) -> list[Path]:
    """``export-part-0001.zip`` → every ``export-part-NNNN.zip`` next to it, in order.

    Large exports are split into numbered parts (later parts hold attachments);
    they only make sense together. Any other path is returned as is.
    """
    match = _PART_RE.match(path.name)
    if not match or not path.is_file():
        return [path]
    prefix = match.group("prefix")
    parts: list[tuple[int, Path]] = []
    for candidate in path.parent.iterdir():
        other = _PART_RE.match(candidate.name)
        if other and other.group("prefix") == prefix and candidate.is_file():
            parts.append((int(other.group("number")), candidate))
    return [p for _, p in sorted(parts)] or [path]


def group_inputs(paths: Iterable[Path]) -> list[list[Path]]:
    """Group input paths into exports: the parts of a split export belong together."""
    groups: list[list[Path]] = []
    for path in paths:
        group = with_sibling_parts(path)
        if group not in groups:
            groups.append(group)
    return groups


def looks_like_export(path: Path) -> bool:
    """Cheap check: a ZIP whose listing contains conversation data."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False
    for name in names:
        base = PurePosixPath(name).name
        if _is_conversation_file(base) or re.match(r"(?i)^conversations__.*\.zip$", base):
            return True
    return False


def find_recent_exports(folder: Path, limit: int = 30) -> list[Path]:
    """Export ZIPs in ``folder``, newest first (checks the ``limit`` newest ZIPs)."""
    try:
        zips = [p for p in folder.iterdir() if p.suffix.lower() == ".zip" and p.is_file()]
    except OSError:
        return []
    zips.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    # Later parts of a split export hold only attachments, so they don't match.
    return [p for p in zips[:limit] if looks_like_export(p)]
