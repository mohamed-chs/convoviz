"""Reading conversation files: shards, odd entries, broken text, attachments index."""

from __future__ import annotations

import json
import zipfile
from typing import TYPE_CHECKING

from factory import PNG_1PX, ConversationBuilder, write_export

from convoviz.diagnostics import Diagnostics
from convoviz.load import load_export
from convoviz.model import AttachmentBlock
from convoviz.source import ExportSource

if TYPE_CHECKING:
    from pathlib import Path


def _conv(n: int, *, title: str | None = None, updated: float = 0) -> dict:
    builder = ConversationBuilder(
        title=title or f"Chat {n}", conversation_id=f"00000000-0000-4000-8000-{n:012d}"
    )
    builder.user(f"question {n}")
    builder.assistant(f"answer {n}")
    raw = builder.build()
    if updated:
        raw["update_time"] = updated
    return raw


def test_sharded_export_with_manifest(tmp_path: Path, load) -> None:
    path = write_export(tmp_path / "e.zip", [_conv(i) for i in range(5)], sharded=2)
    export, diagnostics = load(path)
    assert [c.title for c in export.conversations] == [f"Chat {i}" for i in range(5)]
    assert not diagnostics.errors


def test_shards_without_manifest_are_read_in_numeric_order(tmp_path: Path, load) -> None:
    path = tmp_path / "e.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for index in (10, 2, 1):
            archive.writestr(f"conversations-{index:03d}.json", json.dumps([_conv(index)]))
    export, _ = load(path)
    assert sorted(c.title for c in export.conversations) == ["Chat 1", "Chat 10", "Chat 2"]


def test_missing_shard_is_reported(tmp_path: Path, load) -> None:
    path = write_export(tmp_path / "e", [_conv(i) for i in range(4)], sharded=2, as_zip=False)
    (path / "conversations-001.json").unlink()
    export, diagnostics = load(path)
    assert len(export.conversations) == 2
    assert any("conversations-001.json" in e for e in diagnostics.errors)


def test_empty_and_junk_entries_are_skipped(tmp_path: Path, load) -> None:
    path = write_export(tmp_path / "e.zip", [{}, _conv(1), [], "x", {"title": "no mapping"}])
    export, diagnostics = load(path)
    assert [c.title for c in export.conversations] == ["Chat 1"]
    assert diagnostics.skipped
    assert not diagnostics.errors


def test_duplicates_keep_most_recent(tmp_path: Path, workdir: Path) -> None:
    older = write_export(tmp_path / "a.zip", [_conv(1, title="Old", updated=1_800_000_000)])
    newer = write_export(tmp_path / "b.zip", [_conv(1, title="New", updated=1_800_000_500)])
    diagnostics = Diagnostics()
    sources = [ExportSource.open(newer, workdir / "1"), ExportSource.open(older, workdir / "2")]
    export = load_export(sources, diagnostics)
    assert [c.title for c in export.conversations] == ["New"]
    assert diagnostics.warnings


def test_lone_surrogates_are_repaired(tmp_path: Path, load) -> None:
    raw = json.dumps([_conv(1, title="smile \\ud83d\\ude00 and lone \\ud83d")])
    # json.dumps escaped the backslashes; undo so the file has real \u escapes.
    path = tmp_path / "conversations.json"
    path.write_text(raw.replace("\\\\u", "\\u"), encoding="utf-8")
    export, diagnostics = load(path)
    assert export.conversations[0].title.startswith("smile 😀")
    assert not diagnostics.errors


def test_cesu8_bytes_are_repaired(tmp_path: Path, load) -> None:
    # Each UTF-16 surrogate half encoded as UTF-8 on its own (CESU-8), as seen in
    # real exports: 😂 as ED A0 BD ED B8 82 instead of F0 9F 98 82 (issue #69).
    data = json.dumps([_conv(1, title="laugh 😂")], ensure_ascii=False).encode()
    cesu = data.replace("😂".encode(), b"\xed\xa0\xbd\xed\xb8\x82")
    assert cesu != data
    path = tmp_path / "conversations.json"
    path.write_bytes(cesu)
    export, diagnostics = load(path)
    assert export.conversations[0].title == "laugh 😂"
    assert diagnostics.warnings
    assert not diagnostics.errors


def test_asset_index_resolves_dat_files_and_names(tmp_path: Path, load) -> None:
    files = {
        "file_00000000aaaa1111-0f3c.dat": PNG_1PX,
        "file-Abc123XYZ-report.dat": b"%PDF-1.7",
        "dalle-generations/file_00000000bbbb2222-1a2b.webp": b"RIFF0000WEBP",
    }
    names = {
        "file-Abc123XYZ-report.dat": "Quarterly report.pdf",
        "file_00000000aaaa1111-0f3c.dat": "6c3f0a52-1111-4222-8333-444455556666.png",
    }
    library = [
        {"file_id": "file-Abc123XYZ", "file_name": "ignored.pdf", "mime_type": "application/pdf"},
        {"file_id": "file_00000000aaaa1111", "file_name": "x.png", "trashed_at": "2026-01-01"},
    ]
    path = write_export(
        tmp_path / "e.zip",
        [_conv(1)],
        files=files,
        extra_json={"conversation_asset_file_names.json": names, "library_files.json": library},
    )
    export, _ = load(path)
    assets = export.assets
    image = assets.resolve("sediment://file_00000000aaaa1111")
    assert image is not None
    assert image.original_name is None  # a UUID name carries no meaning
    assert image.mime == "image/png"
    report = assets.resolve("file-service://file-Abc123XYZ")
    assert report is not None
    assert report.original_name == "Quarterly report.pdf"
    assert report.mime == "application/pdf"
    assert assets.resolve("sediment://file_00000000bbbb2222") is not None
    assert assets.resolve("sediment://file_doesnotexist") is None


def test_inventory_manifest_and_hyphenated_names(tmp_path: Path, load) -> None:
    """Some reports show ``export_files`` manifests and ``library-files.json``."""
    path = tmp_path / "e"
    path.mkdir()
    for index in range(3):
        (path / f"conversations-{index:03d}.json").write_text(json.dumps([_conv(index)]))
    inventory = [
        {"path": f"conversations-{index:03d}.json", "size_bytes": 10} for index in (2, 0, 1, 3)
    ]
    inventory.append({"path": "chat.html", "size_bytes": 1})
    inventory.append({"path": "file_0000lost-a.dat", "size_bytes": 2_500_000})
    (path / "Export-Manifest.json").write_text(json.dumps({"export_files": inventory}))
    (path / "file-Abc123XYZ-report.dat").write_bytes(b"%PDF-1.7")
    library = [{"file_id": "file-Abc123XYZ", "file_name": "Notes.pdf", "mime_type": "x/y"}]
    (path / "library-files.json").write_text(json.dumps(library))
    export, diagnostics = load(path)
    assert [c.title for c in export.conversations] == ["Chat 0", "Chat 1", "Chat 2"]
    assert any("conversations-003.json" in e for e in diagnostics.errors)
    assert any(
        "lists 2 files that are not in the export (2.5 MB)" in w for w in diagnostics.warnings
    )
    report = export.assets.resolve("file-service://file-Abc123XYZ")
    assert report is not None
    assert report.mime == "x/y"


def test_library_only_files_are_attached_to_their_conversation(tmp_path: Path, load) -> None:
    from convoviz.render import RenderOptions, render_conversation

    conv_id = "00000000-0000-4000-8000-000000000001"
    builder = ConversationBuilder(title="Pictures", conversation_id=conv_id)
    builder.user("draw a cat")
    builder.add(
        "tool",
        {
            "content_type": "multimodal_text",
            "parts": [
                {
                    "content_type": "image_asset_pointer",
                    "asset_pointer": "sediment://file_0000shown",
                }
            ],
        },
    )
    answer = builder.assistant("Here it is.")
    image = {"origination_thread_id": conv_id, "mime_type": "image/png"}
    library = [
        {"file_id": "file_0000shown", **image},
        {"file_id": "file_0000extra", **image, "created_at": "2026-08-02T00:00:00Z"},
        {"file_id": "file_0000early", **image, "created_at": "2026-08-01T00:00:00Z"},
        {"file_id": "file_0000msg", **image, "origination_message_id": answer},
        {"file_id": "file_0000gone", "origination_thread_id": conv_id, "file_name": "Plan.pdf"},
        {"file_id": "file_0000trash", "origination_thread_id": conv_id, "trashed_at": 1},
        {"file_id": "file_0000paste", **image, "library_artifact_type": "writing_block"},
        {"file_id": "file_0000other", "origination_thread_id": "someone-else"},
    ]
    stored = ("shown", "extra", "early", "msg", "paste", "other")
    path = write_export(
        tmp_path / "e.zip",
        [builder.build()],
        files={f"file_0000{name}-a.dat": PNG_1PX for name in stored},
        extra_json={"library_files.json": library},
    )
    export, diagnostics = load(path)
    conversation = export.conversations[0]
    # Oldest first; no payload, trashed, redundant and foreign entries are left out.
    assert [(a.file_id, a.kind) for a in conversation.files] == [
        ("file_0000early", "image"),
        ("file_0000extra", "image"),
    ]
    host = conversation.nodes[answer].message
    assert host is not None
    assert [b.attachment.file_id for b in host.blocks if isinstance(b, AttachmentBlock)] == [
        "file_0000msg"
    ]
    assert diagnostics.counters["library_files_added"] == 3
    assert diagnostics.counters["library_files_without_payload"] == 1

    def resolve(attachment):
        return "x.png" if export.assets.resolve(attachment.file_id) else None

    note = render_conversation(conversation, RenderOptions(), resolve).text
    files_section = note.split("## Files from this conversation", 1)[1]
    assert files_section.strip().splitlines()[0] == "![](x.png)"
    assert "![](x.png)" in note.split("Here it is.", 1)[1].split("## Files", 1)[0]
