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
    from convoviz.model import Conversation, Message
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


# Library entries that only repeat what a message already shows.
_REDUNDANT_ARTIFACTS = frozenset({"writing_block"})


def _attach_library_files(
    by_id: dict[str, Conversation], assets: AssetIndex, diagnostics: Diagnostics
) -> None:
    """Add the library files that came from a conversation but appear in none of its messages.

    Since mid-2026, generated images and documents are often listed only in
    ``library_files.json``. A file goes to the message named by
    ``origination_message_id`` when the export kept it, otherwise to the end of
    the conversation named by ``origination_thread_id``. Files whose payload is
    not in the export are skipped: there is nothing to show but a missing link.
    """
    wanted = {f.message_id for f in assets.library if f.message_id}
    hosts: dict[str, tuple[Conversation, Message]] = {}
    if wanted:
        for conversation in by_id.values():
            for node_id, node in conversation.nodes.items():
                if node_id in wanted and node.message is not None:
                    hosts[node_id] = (conversation, node.message)
    shown_by_conversation: dict[str, set[str]] = {}
    for library_file in sorted(assets.library, key=lambda f: (f.created, f.file_id)):
        host = hosts.get(library_file.message_id or "")
        if host is not None and library_file.conversation_id not in (None, host[0].id):
            host = None  # the ids disagree: trust the conversation
        conversation = host[0] if host else by_id.get(library_file.conversation_id or "")
        if conversation is None or library_file.artifact_type in _REDUNDANT_ARTIFACTS:
            continue
        info = assets.resolve(library_file.file_id)
        if info is None:
            diagnostics.count("library_files_without_payload")
            continue
        if conversation.id not in shown_by_conversation:
            shown_by_conversation[conversation.id] = _shown_files(conversation, assets)
        shown = shown_by_conversation[conversation.id]
        if library_file.file_id in shown or info.member.path in shown:
            continue
        attachment = Attachment(
            library_file.file_id,
            kind_for(library_file.mime, library_file.name),
            name=library_file.name,
            mime=library_file.mime,
        )
        if host is not None:
            host[1].blocks.append(AttachmentBlock(attachment))
        else:
            conversation.files.append(attachment)
        shown.update((library_file.file_id, info.member.path))
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
