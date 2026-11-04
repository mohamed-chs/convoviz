from __future__ import annotations

import io
import json
import os
import zipfile
from typing import TYPE_CHECKING

import pytest
from factory import ConversationBuilder, write_export

from convoviz.source import (
    ExportSource,
    SourceError,
    find_recent_exports,
    looks_like_export,
    with_sibling_parts,
)

if TYPE_CHECKING:
    from pathlib import Path


def _conv() -> dict:
    builder = ConversationBuilder()
    builder.user("hi")
    return builder.build()


def test_opens_zip_folder_and_json(tmp_path: Path, workdir: Path) -> None:
    zipped = write_export(tmp_path / "a.zip", [_conv()])
    folder = write_export(tmp_path / "folder", [_conv()], as_zip=False)
    for path in (zipped, folder, folder / "conversations.json"):
        source = ExportSource.open(path, workdir)
        assert source.has_conversation_data()
        assert source.get("conversations.json") is not None


def test_strips_single_wrapper_folder(tmp_path: Path, workdir: Path) -> None:
    path = tmp_path / "wrapped.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("export-2026/conversations.json", "[]")
        archive.writestr("export-2026/file_abc.dat", b"x")
    source = ExportSource.open(path, workdir)
    member = source.get("conversations.json")
    assert member is not None
    assert member.read_bytes() == b"[]"
    assert source.get("file_abc.dat") is not None


def test_member_names_cannot_escape(tmp_path: Path, workdir: Path) -> None:
    path = tmp_path / "evil.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("conversations.json", "[]")
        archive.writestr("../../etc/passwd", b"x")
        archive.writestr("C:/Windows/evil.dat", b"x")
        archive.writestr("__MACOSX/._conversations.json", b"x")
    paths = {m.path for m in ExportSource.open(path, workdir)}
    assert all(".." not in p and ":" not in p for p in paths)
    assert not any(p.startswith("__MACOSX") for p in paths)


def test_privacy_portal_nested_zip(tmp_path: Path, workdir: Path) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("conversations.json", json.dumps([_conv()]))
    files = io.BytesIO()
    with zipfile.ZipFile(files, "w") as archive:
        archive.writestr("file_00000000abcd-x.dat", b"\x89PNG")
    outer = tmp_path / "portal.zip"
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("Conversations__2026-10-01.zip", inner.getvalue())
        archive.writestr("Files__2026-10-01.zip", files.getvalue())
    source = ExportSource.open(outer, workdir)
    assert source.has_conversation_data()
    assert source.get("file_00000000abcd-x.dat") is not None


def test_uploaded_zip_attachments_are_not_unpacked(tmp_path: Path, workdir: Path) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("secret.txt", "x")
    path = write_export(
        tmp_path / "e.zip", [_conv()], files={"file_123-project.zip": inner.getvalue()}
    )
    paths = {m.path for m in ExportSource.open(path, workdir)}
    assert "secret.txt" not in paths


def test_missing_and_invalid_inputs(tmp_path: Path, workdir: Path) -> None:
    with pytest.raises(SourceError):
        ExportSource.open(tmp_path / "nope.zip", workdir)
    bad = tmp_path / "bad.txt"
    bad.write_text("hello")
    with pytest.raises(SourceError):
        ExportSource.open(bad, workdir)


def test_sibling_parts_are_collected_in_order(tmp_path: Path) -> None:
    for number in (2, 1, 10):
        (tmp_path / f"abc-2026-part-{number:04d}.zip").write_bytes(b"")
    (tmp_path / "other-part-0001.zip").write_bytes(b"")
    parts = with_sibling_parts(tmp_path / "abc-2026-part-0002.zip")
    assert [p.name for p in parts] == [
        "abc-2026-part-0001.zip",
        "abc-2026-part-0002.zip",
        "abc-2026-part-0010.zip",
    ]
    single = tmp_path / "plain.zip"
    assert with_sibling_parts(single) == [single]


def test_open_many_merges_parts(tmp_path: Path, workdir: Path) -> None:
    first = write_export(tmp_path / "x-part-0001.zip", [_conv()])
    second = tmp_path / "x-part-0002.zip"
    with zipfile.ZipFile(second, "w") as archive:
        archive.writestr("file_999-a.dat", b"data")
    source = ExportSource.open_many(with_sibling_parts(first), workdir)
    assert source.get("conversations.json") is not None
    assert source.get("file_999-a.dat") is not None


def test_find_recent_exports(tmp_path: Path) -> None:
    old = write_export(tmp_path / "old.zip", [_conv()])
    new = write_export(tmp_path / "new.zip", [_conv()])
    with zipfile.ZipFile(tmp_path / "photos.zip", "w") as archive:
        archive.writestr("a.jpg", b"x")
    os.utime(old, (1_000_000, 1_000_000))
    assert looks_like_export(new)
    assert find_recent_exports(tmp_path) == [new, old]
    assert find_recent_exports(tmp_path / "missing") == []
