"""ChatGPT export adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from convoviz.chatgpt.files import AssetIndex, iter_raw_conversations
from convoviz.chatgpt.parse import kind_for, parse_conversation
from convoviz.model import Attachment, AttachmentBlock

if TYPE_CHECKING:
    from collections.abc import Sequence

    from convoviz.diagnostics import Diagnostics
    from convoviz.model import Conversation
    from convoviz.source import ExportSource


@dataclass(slots=True)
class LoadedExport:
    """Everything read from one or more exports."""

    conversations: list[Conversation]
    assets: AssetIndex


def load_export(sources: Sequence[ExportSource], diagnostics: Diagnostics) -> LoadedExport:
    """Parse every conversation in ``sources`` (separate exports, newest first).

    Conversations that appear more than once (across shards, or across several
    exports) are de-duplicated: the most recently updated copy wins, ties go to
    the copy with more messages. Attachments are looked up in every export,
    preferring earlier ones.
    """
    by_id: dict[str, Conversation] = {}
    assets = AssetIndex()
    for source in sources:
        for member, raw in iter_raw_conversations(source, diagnostics):
            try:
                conversation = parse_conversation(raw, diagnostics)
            except Exception as exc:  # noqa: BLE001 - isolate failures per conversation
                diagnostics.error(f"{member.path}: could not read a conversation ({exc!r})")
                continue
            if conversation is None:
                continue
            diagnostics.count("conversations_read")
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
    _attach_library_files(by_id, assets, diagnostics)
    conversations = sorted(by_id.values(), key=lambda c: (c.created, c.id))
    return LoadedExport(conversations, assets)


def _attach_library_files(
    by_id: dict[str, Conversation], assets: AssetIndex, diagnostics: Diagnostics
) -> None:
    """Give conversations the library files that came from them but appear in no message.

    Generated images and documents are sometimes listed only in ``library_files.json``
    (linked by ``origination_thread_id``), with no pointer in the conversation itself.
    """
    shown_by_conversation: dict[str, set[str]] = {}
    for library_file in assets.library:
        conversation = by_id.get(library_file.conversation_id)
        if conversation is None:
            continue
        if conversation.id not in shown_by_conversation:
            shown_by_conversation[conversation.id] = _shown_files(conversation, assets)
        shown = shown_by_conversation[conversation.id]
        info = assets.resolve(library_file.file_id)
        if library_file.file_id in shown or (info is not None and info.member.path in shown):
            continue
        conversation.files.append(
            Attachment(
                library_file.file_id,
                kind_for(library_file.mime, library_file.name),
                name=library_file.name,
                mime=library_file.mime,
            )
        )
        shown.add(library_file.file_id)
        diagnostics.count("library_files_added")


def _shown_files(conversation: Conversation, assets: AssetIndex) -> set[str]:
    """File ids and stored paths of every attachment in the conversation (all branches)."""
    shown: set[str] = set()
    for node in conversation.nodes.values():
        for block in node.message.blocks if node.message else ():
            if isinstance(block, AttachmentBlock):
                shown.add(block.attachment.file_id)
                info = assets.resolve(block.attachment.file_id)
                if info is not None:
                    shown.add(info.member.path)
    return shown
