from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from factory import BASE_TIME, PNG_1PX, ConversationBuilder, write_export

from convoviz.assets import sniff_extension
from convoviz.chatgpt import LoadedExport
from convoviz.diagnostics import Diagnostics
from convoviz.writer import Writer

if TYPE_CHECKING:
    from pathlib import Path

    from convoviz.render import RenderOptions
    from convoviz.writer import Layout


def _conv(
    title: str = "Chat", n: int = 1, text: str = "answer", created: float | None = None
) -> dict:
    b = ConversationBuilder(
        title=title,
        conversation_id=f"00000000-0000-4000-8000-{n:012d}",
        create_time=created or BASE_TIME,
    )
    b.user("question")
    b.add(
        "tool",
        {
            "content_type": "multimodal_text",
            "parts": [
                {
                    "content_type": "image_asset_pointer",
                    "asset_pointer": "sediment://file_00000000img1",
                }
            ],
        },
    )
    b.assistant(text)
    return b.build()


@pytest.fixture
def run(tmp_path: Path, load, options: RenderOptions):
    out = tmp_path / "out"

    def _run(
        conversations: list[dict],
        *,
        force: bool = False,
        layout: Layout = "year",
        files: dict | None = None,
        progress=None,
    ):
        path = write_export(
            tmp_path / "export.zip",
            conversations,
            files=files if files is not None else {"file_00000000img1-x.dat": PNG_1PX},
        )
        export, _ = load(path)
        diagnostics = Diagnostics()
        writer = Writer(out, options, diagnostics, layout=layout, force=force)
        report = writer.write(export, progress=progress)
        return report, out

    return _run


def _notes(out: Path) -> list[str]:
    return sorted(p.relative_to(out).as_posix() for p in (out / "conversations").rglob("*.md"))


def test_first_run_layout(run) -> None:
    report, out = run([_conv()])
    assert report.created == 1
    assert _notes(out) == ["conversations/2026/2026-01-01 Chat.md"]
    assert (out / "index.md").exists()
    assert (out / "Conversations.base").exists()
    assert [p.name for p in (out / "attachments").iterdir()] == ["image_0000img1.png"]
    note = (out / "conversations/2026/2026-01-01 Chat.md").read_text()
    assert "![](../../attachments/image_0000img1.png)" in note
    state = json.loads((out / ".convoviz/state.json").read_text())
    assert state["version"] == 1
    assert list(state["conversations"]) == ["00000000-0000-4000-8000-000000000001"]


def test_flat_layout(run) -> None:
    _, out = run([_conv()], layout="flat")
    assert _notes(out) == ["conversations/2026-01-01 Chat.md"]


def test_rerun_is_incremental(run) -> None:
    run([_conv()])
    report, _ = run([_conv()])
    assert (report.created, report.updated, report.unchanged) == (0, 0, 1)
    report, out = run([_conv(text="better answer")])
    assert report.updated == 1
    assert "better answer" in (out / _notes(out)[0]).read_text()


def test_title_change_moves_the_note(run) -> None:
    run([_conv("Old name")])
    report, out = run([_conv("New name")])
    assert report.moved == 1
    assert _notes(out) == ["conversations/2026/2026-01-01 New name.md"]


def test_user_edits_are_kept_unless_forced(run) -> None:
    _, out = run([_conv()])
    note = out / _notes(out)[0]
    note.write_text(note.read_text() + "\nMy notes\n")
    report, _ = run([_conv()])
    assert (report.kept_edits, report.edited_current) == ([], 1)  # nothing newer to lose
    report, _ = run([_conv(text="changed")])
    assert report.kept_edits == ["conversations/2026/2026-01-01 Chat.md"]
    assert "My notes" in note.read_text()
    report, _ = run([_conv("Renamed", text="changed")])
    assert report.kept_edits
    assert "My notes" in note.read_text()  # not moved either
    report, _ = run([_conv(text="changed")], force=True)
    assert report.updated == 1
    assert "My notes" not in note.read_text()


def test_conversations_absent_from_new_export_are_kept(run) -> None:
    run([_conv("First", 1), _conv("Second", 2)])
    report, out = run([_conv("Second", 2)])
    assert report.total_notes == 2
    assert len(_notes(out)) == 2
    assert "First" in (out / "index.md").read_text()


def test_same_title_same_day_gets_distinct_names(run) -> None:
    _, out = run([_conv("Same", 1), _conv("Same", 2)])
    assert _notes(out) == [
        "conversations/2026/2026-01-01 Same (00000000).md",
        "conversations/2026/2026-01-01 Same.md",
    ]


def test_unrelated_user_file_is_not_overwritten(run, tmp_path: Path) -> None:
    target = tmp_path / "out/conversations/2026/2026-01-01 Chat.md"
    target.parent.mkdir(parents=True)
    target.write_text("mine")
    _, out = run([_conv()])
    assert target.read_text() == "mine"
    assert len(_notes(out)) == 2


def test_corrupt_state_is_ignored(run, tmp_path: Path) -> None:
    state = tmp_path / "out/.convoviz/state.json"
    state.parent.mkdir(parents=True)
    state.write_text("{not json")
    report, _ = run([_conv()])
    assert report.created == 1


def test_missing_attachments_are_counted(run) -> None:
    report, out = run([_conv()], files={})
    assert report.attachments_missing == 1
    assert "not included in the export" in (out / _notes(out)[0]).read_text()


def test_empty_export(tmp_path: Path, options: RenderOptions) -> None:
    from convoviz.chatgpt.files import AssetIndex

    report = Writer(tmp_path / "o", options, Diagnostics()).write(LoadedExport([], AssetIndex()))
    assert report.total_notes == 0
    assert (tmp_path / "o/index.md").exists()


@pytest.mark.parametrize(
    ("head", "ext"),
    [
        (PNG_1PX, ".png"),
        (b"\xff\xd8\xff\xe0", ".jpg"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", ".webp"),
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", ".wav"),
        (b"%PDF-1.7", ".pdf"),
        (b"\x00\x00\x00\x18ftypmp42", ".mp4"),
        (b"plain text", ".txt"),
        (b"\x00\x01\x02\xff", None),
    ],
)
def test_sniff_extension(head: bytes, ext: str | None) -> None:
    assert sniff_extension(head) == ext


def test_dates_follow_the_configured_timezone(tmp_path: Path, load, options: RenderOptions) -> None:
    from zoneinfo import ZoneInfo

    evening_utc = 1_767_297_600.0  # 2026-01-01T20:00:00Z = 2026-01-02 05:00 in Tokyo
    export, _ = load(write_export(tmp_path / "e.zip", [_conv(created=evening_utc)]))
    tokyo = replace(options, timezone=ZoneInfo("Asia/Tokyo"))
    Writer(tmp_path / "out", tokyo, Diagnostics()).write(export)
    assert _notes(tmp_path / "out") == ["conversations/2026/2026-01-02 Chat.md"]


def test_interrupted_run_is_resumed_without_duplicates(run) -> None:
    conversations = [_conv(f"Chat {n}", n) for n in range(1, 5)]

    def interrupt_after_two(conversation) -> None:
        if conversation.title == "Chat 2":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(conversations, progress=interrupt_after_two)
    report, out = run(conversations)
    assert (report.created, report.unchanged) == (2, 2)
    assert len(_notes(out)) == 4


def test_lost_state_adopts_notes_instead_of_duplicating(run, tmp_path: Path) -> None:
    _, out = run([_conv("A", 1), _conv("B", 2)])
    edited = out / "conversations/2026/2026-01-01 B.md"
    edited.write_text(edited.read_text() + "\nmine\n")
    (out / ".convoviz/state.json").unlink()
    report, _ = run([_conv("A", 1), _conv("B", 2)])
    assert len(_notes(out)) == 2
    assert report.unchanged == 1
    assert report.kept_edits == ["conversations/2026/2026-01-01 B.md"]
    assert "mine" in edited.read_text()


def test_moved_note_is_updated_where_it_is(run) -> None:
    _, out = run([_conv()])
    moved = out / "Projects/Trip/My chat.md"
    moved.parent.mkdir(parents=True)
    (out / _notes(out)[0]).rename(moved)
    report, _ = run([_conv("Renamed in ChatGPT", text="better answer")])
    assert report.updated == 1
    assert _notes(out) == []
    assert "better answer" in moved.read_text()
    assert "Projects/Trip/My%20chat.md" in (out / "index.md").read_text()
    report, _ = run([_conv("Renamed in ChatGPT", text="better answer")])
    assert report.unchanged == 1


def test_deleted_note_stays_deleted_unless_forced(run) -> None:
    _, out = run([_conv("Keep", 1), _conv("Drop", 2)])
    (out / "conversations/2026/2026-01-01 Drop.md").unlink()
    report, _ = run([_conv("Keep", 1), _conv("Drop", 2)])
    assert report.deleted_skipped == 1
    assert _notes(out) == ["conversations/2026/2026-01-01 Keep.md"]
    assert "Drop" not in (out / "index.md").read_text()
    assert report.total_notes == 1
    report, _ = run([_conv("Keep", 1), _conv("Drop", 2)], force=True)
    assert report.created == 1
    assert len(_notes(out)) == 2


def test_note_in_obsidian_trash_counts_as_deleted(run) -> None:
    _, out = run([_conv()])
    trash = out / ".trash"
    trash.mkdir()
    (out / _notes(out)[0]).rename(trash / "Chat.md")
    report, _ = run([_conv()])
    assert report.deleted_skipped == 1


def test_generated_files_the_user_changed_are_kept(run, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.md").write_text("# My vault home\n")
    report, _ = run([_conv()])
    assert report.kept_files == ["index.md"]
    assert (out / "index.md").read_text() == "# My vault home\n"
    base = out / "Conversations.base"
    base.write_text(base.read_text() + "  - type: cards\n    name: Mine\n")
    report, _ = run([_conv("Other title")])
    assert report.kept_files == ["index.md", "Conversations.base"]
    assert "Mine" in base.read_text()
    (out / "index.md").unlink()
    report, _ = run([_conv()])
    assert report.kept_files == ["Conversations.base"]
    assert "Chat" in (out / "index.md").read_text()
