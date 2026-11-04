"""Write the Markdown archive, incrementally and without clobbering user edits.

Layout::

    <output>/
      conversations/2026/2026-03-14 Title.md
      attachments/…
      index.md
      Conversations.base
      insights.html
      .convoviz/state.json

``state.json`` remembers, per conversation id, where its note lives and the hash
of what convoviz last wrote there. On a re-run with a newer export:

* unchanged conversations are not touched;
* changed ones are rewritten (and moved if their title changed);
* notes the user edited since are kept unless ``force`` is set;
* conversations missing from the new export are left alone (exports can be partial).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote

import orjson

from convoviz.assets import AssetStore
from convoviz.fsnames import safe_stem
from convoviz.markdown import escape_inline
from convoviz.render import local_time, render_conversation

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

    def to_json(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "title": self.title,
            "created": self.created,
            "updated": self.updated,
            "messages": self.messages,
        }

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
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(slots=True)
class State:
    """The ``.convoviz/state.json`` manifest."""

    entries: dict[str, Entry] = field(default_factory=dict)

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
        return state

    def save(self, output: Path) -> None:
        """Write the manifest atomically."""
        data = {
            "version": STATE_VERSION,
            "conversations": {k: v.to_json() for k, v in sorted(self.entries.items())},
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
    kept_edits: list[str] = field(default_factory=list)
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

    def write(
        self,
        export: LoadedExport,
        progress: Callable[[Conversation], None] | None = None,
    ) -> WriteReport:
        """Render every conversation and refresh the index files."""
        self.output.mkdir(parents=True, exist_ok=True)
        assets = AssetStore(self.output, export.assets, self.diagnostics)
        claimed = self._claimed_paths(export.conversations)
        for conversation in export.conversations:
            try:
                self._write_one(conversation, claimed, assets)
            except OSError as exc:
                self.diagnostics.error(f"could not write {conversation.title!r}: {exc}")
            if progress is not None:
                progress(conversation)
        self.state.save(self.output)
        self._write_index()
        self._write_base()
        self.report.attachments_copied = assets.copied
        self.report.attachments_missing = assets.missing
        self.report.total_notes = len(self.state.entries)
        return self.report

    # ------------------------------------------------------------------ notes

    def _claimed_paths(self, conversations: list[Conversation]) -> dict[str, str]:
        """Assign each conversation a unique path, keeping existing notes where they are."""
        taken: dict[str, str] = {e.path.casefold(): cid for cid, e in self.state.entries.items()}
        claimed: dict[str, str] = {}
        for conversation in conversations:
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
        edited = previous is not None and self._edited(previous)
        if edited and not self.force:
            # The user changed the note since convoviz wrote it: leave it alone.
            self.report.kept_edits.append(previous.path)
            return

        note = render_conversation(
            conversation, self.options, lambda a: assets.link_from(target, a)
        )
        data = note.text.encode()
        digest = _sha(data)
        self.state.entries[conversation.id] = Entry(
            path=rel,
            sha256=digest,
            title=conversation.title,
            created=conversation.created.astimezone(UTC).isoformat(),
            updated=conversation.updated.astimezone(UTC).isoformat(),
            messages=note.messages,
        )
        if previous is None:
            _atomic_write(target, data)
            self.report.created += 1
        elif previous.path == rel and previous.sha256 == digest and target.exists() and not edited:
            self.report.unchanged += 1
        else:
            _atomic_write(target, data)
            if previous.path != rel:
                (self.output / previous.path).unlink(missing_ok=True)
                _remove_empty_parents(self.output / previous.path, self.output)
                self.report.moved += 1
            else:
                self.report.updated += 1

    def _edited(self, entry: Entry) -> bool:
        path = self.output / entry.path
        try:
            return _sha(path.read_bytes()) != entry.sha256
        except FileNotFoundError:
            return False

    # ------------------------------------------------------------------ indexes

    def _write_index(self) -> None:
        entries = sorted(self.state.entries.values(), key=lambda e: e.created, reverse=True)
        lines = [
            "# ChatGPT conversations",
            "",
            (
                f"{len(entries)} conversations. Open `Conversations.base` in Obsidian "
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
        _atomic_write(self.output / "index.md", ("\n".join(lines) + "\n").encode())

    def _write_base(self) -> None:
        folder = CONVERSATIONS_DIR
        base = f"""\
filters:
  and:
    - file.inFolder("{folder}")
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
        _atomic_write(self.output / "Conversations.base", base.encode())


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
