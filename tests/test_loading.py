"""Reading conversation files: shards, odd entries, broken text, attachments index."""

from __future__ import annotations

import json
import zipfile
from typing import TYPE_CHECKING

from factory import PNG_1PX, ConversationBuilder, write_export

from convoviz.chatgpt import load_export
from convoviz.diagnostics import Diagnostics
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
    title = "emoji 😀"
    data = json.dumps([_conv(1, title=title)], ensure_ascii=False).encode("utf-16", "surrogatepass")
    cesu = data.decode("utf-16").encode("utf-8", "surrogatepass")  # same as utf-8 here
    path = tmp_path / "conversations.json"
    path.write_bytes(cesu.replace("😀".encode(), "😀".encode("utf-8", "surrogatepass")))
    export, _ = load(path)
    assert export.conversations[0].title == title


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
