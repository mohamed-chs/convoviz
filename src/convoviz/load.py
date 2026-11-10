"""Read every conversation of one or more exports, whichever provider made them."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from convoviz import chatgpt, claude
from convoviz.chatgpt.files import AssetIndex
from convoviz.export import iter_raw_conversations

if TYPE_CHECKING:
    from collections.abc import Sequence

    from convoviz.diagnostics import Diagnostics
    from convoviz.model import Conversation, Memory, Project
    from convoviz.source import ExportSource

_PARSERS = {
    "chatgpt": chatgpt.parse_conversation,
    "claude": claude.parse_conversation,
}


@dataclass(slots=True)
class LoadedExport:
    """Everything read from one or more exports."""

    conversations: list[Conversation]
    assets: AssetIndex
    projects: list[Project] = field(default_factory=list)
    memory: Memory | None = None  # from the newest export that has one


def load_export(sources: Sequence[ExportSource], diagnostics: Diagnostics) -> LoadedExport:
    """Parse every conversation in ``sources`` (separate exports, newest first).

    Conversations that appear more than once (across shards, or across several
    exports) are de-duplicated: the most recently updated copy wins, ties go to
    the copy with more messages. Projects are de-duplicated the same way.
    Attachments are looked up in every export, preferring earlier ones.
    """
    by_id: dict[str, Conversation] = {}
    projects: dict[str, Project] = {}
    memory: Memory | None = None
    assets = AssetIndex()
    for source in sources:
        for project in claude.read_projects(source, diagnostics):
            known = projects.get(project.id)
            if known is None or (project.updated or _EPOCH) > (known.updated or _EPOCH):
                projects[project.id] = project
        memory = memory or claude.read_memory(source, diagnostics)
        for member, provider, raw in iter_raw_conversations(source, diagnostics):
            try:
                conversation = _PARSERS[provider](raw, diagnostics)
            except Exception as exc:  # noqa: BLE001 - isolate failures per conversation
                diagnostics.error(f"{member.path}: could not read a conversation ({exc!r})")
                continue
            if conversation is None:
                continue
            diagnostics.count("conversations_read")
            diagnostics.count(f"conversations_read:{provider}")
            existing = by_id.get(conversation.id)
            if existing is None:
                by_id[conversation.id] = conversation
                continue
            diagnostics.count("duplicates")
            if (conversation.updated, len(conversation.nodes)) > (
                existing.updated,
                len(existing.nodes),
            ):
                by_id[conversation.id] = conversation
        assets.merge(AssetIndex.build(source, diagnostics))
    if diagnostics.counters["duplicates"]:
        diagnostics.warn(
            f"{diagnostics.counters['duplicates']} conversations appeared more than "
            "once; kept the most recent copy of each."
        )
    chatgpt.attach_library_files(by_id, assets, diagnostics)
    conversations = sorted(by_id.values(), key=lambda c: (c.created, c.id))
    ordered = sorted(projects.values(), key=lambda p: (p.name.casefold(), p.id))
    return LoadedExport(conversations, assets, ordered, memory)


_EPOCH = datetime.min.replace(tzinfo=UTC)
