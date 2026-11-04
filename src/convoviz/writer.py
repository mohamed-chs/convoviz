"""Write the Markdown archive, incrementally and without clobbering user edits.

Layout::

    <output>/
      conversations/2026/2026-03-14 Title.md
      attachments/…
      canvas/Title (1a2b3c4d).py   the final version of each Canvas document
      index.md
      Conversations.base
      Custom instructions.md   (when the export recorded any)
      insights.html
      .convoviz/state.json

``state.json`` remembers, per conversation id, where its note lives and the hash
of what convoviz last wrote there. On a re-run with a newer export:

* unchanged conversations are not touched;
* changed ones are rewritten (and moved if their title changed);
* notes the user edited since are kept unless ``force`` is set;
* notes the user moved or renamed are found again by the ``conversation_id``
  property and updated where they are, never moved back;
* notes the user deleted stay deleted unless ``force`` is set;
* a note carrying a conversation's id but missing from the state (an interrupted
  run, a deleted ``.convoviz`` folder) is adopted rather than duplicated;
* conversations missing from the new export are left alone (exports can be partial).

``index.md``, ``Conversations.base``, ``Custom instructions.md``, ``insights.html``
and the Canvas files are tracked the same way: a copy the user changed, or a file convoviz didn't
create, is left alone.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote

import orjson

from convoviz.assets import AssetStore
from convoviz.fsnames import safe_stem
from convoviz.markdown import escape_inline
from convoviz.render import (
    CANVAS_DIR,
    INSTRUCTIONS_PAGE,
    canvas_files,
    local_time,
    render_conversation,
    render_instructions,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from convoviz.chatgpt import LoadedExport
    from convoviz.diagnostics import Diagnostics
    from convoviz.model import Conversation
    from convoviz.render import RenderOptions

Layout = Literal["year", "flat"]
STATE_DIR = ".convoviz"
STATE_VERSION = 1
CONVERSATIONS_DIR = "conversations"


@dataclass(slots=True)
class Entry:
    """What convoviz remembers about one written note."""

    path: str  # relative to the output root, POSIX separators
    sha256: str
    title: str
    created: str
    updated: str
    messages: int = 0
    pinned: bool = False  # the user chose this location: never move the note
    deleted: bool = False  # the user deleted the note: don't recreate it

    def to_json(self) -> dict[str, object]:
        """Serializable form."""
        data: dict[str, object] = {
            "path": self.path,
            "sha256": self.sha256,
            "title": self.title,
            "created": self.created,
            "updated": self.updated,
            "messages": self.messages,
        }
        if self.pinned:
            data["pinned"] = True
        if self.deleted:
            data["deleted"] = True
        return data

    @classmethod
    def from_json(cls, data: object) -> Entry | None:
        """Parse a stored entry; None if it is malformed."""
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                path=str(data["path"]),
                sha256=str(data["sha256"]),
                title=str(data.get("title", "")),
                created=str(data.get("created", "")),
                updated=str(data.get("updated", "")),
                messages=int(data.get("messages", 0)),
                pinned=data.get("pinned") is True,
                deleted=data.get("deleted") is True,
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(slots=True)
class State:
    """The ``.convoviz/state.json`` manifest."""

    entries: dict[str, Entry] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)  # managed file -> sha256 written

    @classmethod
    def load(cls, output: Path, diagnostics: Diagnostics) -> State:
        """Read the manifest (an unreadable one is treated as empty)."""
        path = output / STATE_DIR / "state.json"
        if not path.exists():
            return cls()
        try:
            data = orjson.loads(path.read_bytes())
        except (OSError, orjson.JSONDecodeError) as exc:
            diagnostics.warn(f"ignoring unreadable {path}: {exc}")
            return cls()
        state = cls()
        raw = data.get("conversations") if isinstance(data, dict) else None
        for conv_id, value in (raw or {}).items():
            entry = Entry.from_json(value)
            if entry is not None:
                state.entries[str(conv_id)] = entry
        files = data.get("files") if isinstance(data, dict) else None
        if isinstance(files, dict):
            state.files = {str(k): v for k, v in files.items() if isinstance(v, str)}
        return state

    def save(self, output: Path) -> None:
        """Write the manifest atomically."""
        data = {
            "version": STATE_VERSION,
            "conversations": {k: v.to_json() for k, v in sorted(self.entries.items())},
            "files": dict(sorted(self.files.items())),
        }
        _atomic_write(
            output / STATE_DIR / "state.json",
            orjson.dumps(data, option=orjson.OPT_INDENT_2),
        )


@dataclass(slots=True)
class WriteReport:
    """What a run did to the output folder."""

    created: int = 0
    updated: int = 0
    moved: int = 0
    unchanged: int = 0
    # Edited notes whose conversation has newer content that was not written.
    kept_edits: list[str] = field(default_factory=list)
    # Edited notes with nothing newer in the export.
    edited_current: int = 0
    deleted_skipped: int = 0
    kept_files: list[str] = field(default_factory=list)
    attachments_copied: int = 0
    attachments_missing: int = 0
    total_notes: int = 0


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def note_path(conversation: Conversation, layout: Layout, options: RenderOptions) -> str:
    """Preferred relative path of a conversation's note."""
    created = local_time(conversation.created, options.timezone)
    name = f"{created:%Y-%m-%d} {safe_stem(conversation.title)}.md"
    if layout == "year":
        return f"{CONVERSATIONS_DIR}/{created:%Y}/{name}"
    return f"{CONVERSATIONS_DIR}/{name}"


def _with_suffix(path: str, conv_id: str) -> str:
    stem, ext = path.rsplit(".", 1)
    return f"{stem} ({conv_id.replace('-', '')[:8]}).{ext}"


class Writer:
    """Writes one export into an output folder."""

    def __init__(
        self,
        output: Path,
        options: RenderOptions,
        diagnostics: Diagnostics,
        *,
        layout: Layout = "year",
        force: bool = False,
    ) -> None:
        self.output = output
        self.options = options
        self.diagnostics = diagnostics
        self.layout = layout
        self.force = force
        self.state = State.load(output, diagnostics)
        self.report = WriteReport()
        self._adopted: dict[str, str] = {}

    def write(
        self,
        export: LoadedExport,
        progress: Callable[[Conversation], None] | None = None,
        *,
        extra_files: dict[str, bytes] | None = None,
    ) -> WriteReport:
        """Render every conversation, then refresh the index files and ``extra_files``."""
        self.output.mkdir(parents=True, exist_ok=True)
        assets = AssetStore(self.output, export.assets, self.diagnostics)
        try:
            self._reconcile(export.conversations)
            claimed = self._claimed_paths(export.conversations)
            for conversation in export.conversations:
                try:
                    self._write_one(conversation, claimed, assets)
                except OSError as exc:
                    self.diagnostics.error(f"could not write {conversation.title!r}: {exc}")
                if progress is not None:
                    progress(conversation)
            self._write_managed("index.md", self._index())
            self._write_managed("Conversations.base", _BASE.encode())
            instructions = render_instructions(
                export.conversations, self._link_from_root, self.options.timezone
            )
            if instructions is not None:
                self._write_managed(INSTRUCTIONS_PAGE, instructions.encode())
            for rel, data in (extra_files or {}).items():
                self._write_managed(rel, data)
        finally:
            # Saved even when interrupted, so a re-run knows which notes are its own.
            self.state.save(self.output)
        self.report.attachments_copied = assets.copied
        self.report.attachments_missing = assets.missing
        self.report.total_notes = sum(not e.deleted for e in self.state.entries.values())
        return self.report

    # ------------------------------------------------------------------ notes

    def _reconcile(self, conversations: list[Conversation]) -> None:
        """Bring the state up to date with what the user did to the notes since."""
        missing: list[tuple[str, Entry]] = []
        for conv_id, entry in self.state.entries.items():
            exists = (self.output / entry.path).exists()
            if entry.deleted and exists:
                entry.deleted = False  # restored by the user
            elif not entry.deleted and not exists:
                missing.append((conv_id, entry))
        unrecorded = [c.id for c in conversations if c.id not in self.state.entries]
        if not missing and not unrecorded:
            return
        found = _find_notes(self.output)
        for conv_id, entry in missing:
            paths = found.get(conv_id)
            if paths:
                entry.path, entry.pinned = paths[0], True  # moved or renamed by the user
            else:
                entry.deleted = True
        for conv_id in unrecorded:
            if conv_id in found:
                self._adopted[conv_id] = found[conv_id][0]

    def _claimed_paths(self, conversations: list[Conversation]) -> dict[str, str]:
        """Assign each conversation a unique path, keeping existing notes where they are."""
        taken: dict[str, str] = {
            e.path.casefold(): cid for cid, e in self.state.entries.items() if not e.deleted
        }
        claimed: dict[str, str] = {}
        for conversation in conversations:
            entry = self.state.entries.get(conversation.id)
            if entry is not None and entry.pinned and not entry.deleted:
                path = entry.path
            elif conversation.id in self._adopted:
                path = self._adopted[conversation.id]
            else:
                path = note_path(conversation, self.layout, self.options)
                owner = taken.get(path.casefold())
                existing_file = (self.output / path).exists()
                if (owner not in (None, conversation.id)) or (owner is None and existing_file):
                    path = _with_suffix(path, conversation.id)
            taken[path.casefold()] = conversation.id
            claimed[conversation.id] = path
        return claimed

    def _write_one(
        self, conversation: Conversation, claimed: dict[str, str], assets: AssetStore
    ) -> None:
        rel = claimed[conversation.id]
        target = self.output / rel
        previous = self.state.entries.get(conversation.id)
        adopted = conversation.id in self._adopted
        if previous is not None and previous.deleted and not self.force:
            self.report.deleted_skipped += 1
            return

        note = render_conversation(
            conversation,
            self.options,
            lambda a: assets.link_from(target, a),
            lambda path: Path(os.path.relpath(self.output / path, target.parent)).as_posix(),
        )
        for path, document in canvas_files(conversation).items():
            self._write_managed(path, (document.content.rstrip("\n") + "\n").encode())
        data = note.text.encode()
        digest = _sha(data)
        if previous is not None and not previous.deleted:
            edited = self._edited(previous)
        else:
            # An adopted note has no history: any difference may be the user's.
            edited = adopted and _sha(target.read_bytes()) != digest
        canonical = note_path(conversation, self.layout, self.options)
        entry = Entry(
            path=rel,
            sha256=digest,
            title=conversation.title,
            created=conversation.created.astimezone(UTC).isoformat(),
            updated=conversation.updated.astimezone(UTC).isoformat(),
            messages=note.messages,
            pinned=(previous is not None and previous.pinned) or (adopted and rel != canonical),
        )
        if edited and not self.force:
            kept_at = previous.path if previous is not None and not previous.deleted else rel
            if previous is not None and previous.sha256 == digest:
                self.report.edited_current += 1
            else:
                self.report.kept_edits.append(kept_at)
            if previous is None:
                self.state.entries[conversation.id] = entry  # adopt it, edits and all
            return

        # Record the note only once it is on disk: a failed write is retried next run
        # instead of looking like a note the user deleted.
        if previous is None or previous.deleted:
            if adopted and not edited:
                self.report.unchanged += 1
            else:
                _atomic_write(target, data)
                if adopted:
                    self.report.updated += 1
                else:
                    self.report.created += 1
        elif previous.path == rel and previous.sha256 == digest and target.exists() and not edited:
            self.report.unchanged += 1
        else:
            _atomic_write(target, data)
            if previous.path != rel:
                old = self.output / previous.path
                # A change of case only names the same file on macOS and Windows.
                if old.exists() and not old.samefile(target):
                    old.unlink()
                    _remove_empty_parents(old, self.output)
                self.report.moved += 1
            else:
                self.report.updated += 1
        self.state.entries[conversation.id] = entry

    def _edited(self, entry: Entry) -> bool:
        path = self.output / entry.path
        try:
            return _sha(path.read_bytes()) != entry.sha256
        except FileNotFoundError:
            return False

    # ------------------------------------------------------------------ other files

    def _write_managed(self, rel: str, data: bytes) -> None:
        """Write a generated file unless the user changed it or it isn't convoviz's."""
        path = self.output / rel
        digest = _sha(data)
        if path.exists():
            current = _sha(path.read_bytes())
            if current != digest and current != self.state.files.get(rel) and not self.force:
                self.report.kept_files.append(rel)
                return
            if current == digest:
                self.state.files[rel] = digest
                return
        _atomic_write(path, data)
        self.state.files[rel] = digest

    def _link_from_root(self, conversation: Conversation) -> str | None:
        entry = self.state.entries.get(conversation.id)
        if entry is None or entry.deleted:
            return None
        return quote(entry.path, safe="/._-~")

    def _index(self) -> bytes:
        entries = sorted(
            (e for e in self.state.entries.values() if not e.deleted),
            key=lambda e: e.created,
            reverse=True,
        )
        lines = [
            "# ChatGPT conversations",
            "",
            (
                f"{len(entries):,} conversation{'' if len(entries) == 1 else 's'}. "
                "Open `Conversations.base` in Obsidian "
                "for a sortable table, or `insights.html` for charts."
            ),
        ]
        year_month = ""
        for entry in entries:
            when = _parse_iso(entry.created)
            group = f"{local_time(when, self.options.timezone):%Y · %B}" if when else "Undated"
            if group != year_month:
                lines += ["", f"## {group}", ""]
                year_month = group
            link = quote(entry.path, safe="/._-~")
            day = f"{local_time(when, self.options.timezone):%d}" if when else ""
            prefix = f"{day} · " if day else ""
            lines.append(f"- {prefix}[{escape_inline(entry.title)}]({link})")
        return ("\n".join(lines) + "\n").encode()


# Notes are matched by property, not folder, so ones the user moves stay in the table.
_BASE = """\
filters:
  and:
    - note.source == "chatgpt"
    - file.ext == "md"
properties:
  file.name:
    displayName: Conversation
  note.created:
    displayName: Created
  note.updated:
    displayName: Updated
  note.messages:
    displayName: Messages
  note.models:
    displayName: Models
views:
  - type: table
    name: All conversations
    order:
      - file.name
      - created
      - updated
      - messages
      - models
    sort:
      - property: created
        direction: DESC
  - type: table
    name: Starred
    filters:
      and:
        - starred == true
    order:
      - file.name
      - created
      - messages
    sort:
      - property: created
        direction: DESC
"""

_ID_LINE_RE = re.compile(
    r"""^conversation_id:[ \t]*["']?([^"'\r\n]+?)["']?[ \t]*\r?$""", re.MULTILINE
)


def _find_notes(output: Path) -> dict[str, list[str]]:
    """Every Markdown note under ``output`` that carries a ``conversation_id`` property.

    Hidden folders (``.convoviz``, ``.obsidian``, Obsidian's ``.trash``),
    ``attachments`` and ``canvas`` are skipped. Only the first few KB of each file are read.
    """
    found: dict[str, list[str]] = defaultdict(list)
    for folder, subfolders, names in os.walk(output):
        subfolders[:] = sorted(
            d for d in subfolders if not d.startswith(".") and d not in ("attachments", CANVAS_DIR)
        )
        for name in sorted(names):
            if not name.endswith(".md"):
                continue
            path = Path(folder) / name
            try:
                with path.open("rb") as handle:
                    head = handle.read(8192)
            except OSError:
                continue
            if not head.startswith(b"---"):
                continue
            end = head.find(b"\n---", 3)
            properties = head[: end if end != -1 else len(head)].decode("utf-8", "replace")
            match = _ID_LINE_RE.search(properties)
            if match:
                found[match.group(1)].append(path.relative_to(output).as_posix())
    return found


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _remove_empty_parents(path: Path, root: Path) -> None:
    parent = path.parent
    while parent != root and root in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            return
        parent = parent.parent
