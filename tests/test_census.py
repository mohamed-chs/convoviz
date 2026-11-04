from __future__ import annotations

import json
from typing import TYPE_CHECKING

from factory import PNG_1PX, ConversationBuilder, write_export

from convoviz.census import census, normalize_member_name
from convoviz.source import ExportSource

if TYPE_CHECKING:
    from pathlib import Path

SECRETS = [
    "Secret Title",
    "my private question",
    "https://private.example",
    "salary.pdf",
    "1f3a9c77",
]


def test_census_never_contains_content(tmp_path: Path, workdir: Path) -> None:
    b = ConversationBuilder(
        title="Secret Title", conversation_id="1f3a9c77-aaaa-4bbb-8ccc-1234567890ab"
    )
    b.user(
        "my private question", metadata={"attachments": [{"id": "file-xyz", "name": "salary.pdf"}]}
    )
    b.assistant(
        "see https://private.example",
        metadata={
            "content_references": [
                {"type": "webpage", "url": "https://private.example", "matched_text": "x"}
            ],
            "my private question": 1,
        },
    )
    path = write_export(
        tmp_path / "e.zip",
        [b.build()],
        sharded=1,
        files={"file-xyz-salary.pdf.dat": PNG_1PX, "user-1f3a9c77/salary.pdf": b"x"},
    )
    report = census([ExportSource.open(path, workdir)])
    dumped = json.dumps(report)
    for secret in SECRETS:
        assert secret not in dumped
    assert report["shape"]["conversations"] == 1
    assert report["content_reference_types"] == {"webpage": 1}
    assert report["parsed"]["conversations"] == 1
    assert "conversations-N.json" in report["files"]


def test_normalize_member_name() -> None:
    assert normalize_member_name("conversations-012.json") == "conversations-N.json"
    assert normalize_member_name("file_00000000abc-My Taxes.dat") == "file_*.dat"
    assert normalize_member_name("dalle-generations/6c3f.webp") == "dalle-generations/*.webp"
    assert normalize_member_name("user-abc/Holiday.JPG") == "*/*.jpg"
