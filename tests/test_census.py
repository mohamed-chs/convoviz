from __future__ import annotations

import json
from typing import TYPE_CHECKING

from factory import (
    PNG_1PX,
    ClaudeBuilder,
    ConversationBuilder,
    tool_use,
    write_claude_export,
    write_export,
)

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
                {"type": "webpage", "url": "https://private.example", "matched_text": "x"},
                {"type": "my private question", "matched_text": "see"},
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
    assert report["chatgpt"]["shape"]["conversations"] == 1
    assert report["chatgpt"]["content_reference_types"] == {"webpage": 1, "<other>": 1}
    assert report["parsed"]["unknown"] == {"content_reference:<other>": 1}
    assert report["parsed"]["conversations"] == 1
    assert "conversations-N.json" in report["files"]


def test_normalize_member_name() -> None:
    assert normalize_member_name("conversations-012.json") == "conversations-N.json"
    assert normalize_member_name("file_00000000abc-My Taxes.dat") == "file_*.dat"
    assert normalize_member_name("dalle-generations/6c3f.webp") == "dalle-generations/*.webp"
    assert normalize_member_name("user-abc/Holiday.JPG") == "*/*.jpg"


def test_claude_census_never_contains_content(tmp_path: Path, workdir: Path) -> None:
    b = ClaudeBuilder(name="Secret Title", uuid="1f3a9c77-aaaa-4bbb-8ccc-1234567890ab")
    b.human(
        "my private question",
        attachments=[{"file_name": "salary.pdf", "file_type": "pdf", "extracted_content": "x"}],
    )
    cite = {
        "end_index": 3,
        "details": {"type": "web_search_citation", "url": "https://private.example"},
    }
    b.assistant(
        [
            tool_use("AcmeInternal:query", {"my private question": 1}, "t1"),
            tool_use("Bash", {"command": "cat salary.pdf"}, "t2"),
            {"type": "text", "text": "see https://private.example", "citations": [cite]},
        ]
    )
    path = write_claude_export(tmp_path / "conversations-000.zip", [b.build()])
    report = census([ExportSource.open(path, workdir)])
    dumped = json.dumps(report)
    for secret in [*SECRETS, "AcmeInternal"]:
        assert secret not in dumped
    assert "chatgpt" not in report
    claude = report["claude"]
    assert claude["shape"] == {"conversations": 1, "messages": 2}
    assert claude["tool_names"] == {"<mcp>": 1, "Bash": 1}
    assert claude["tool_input_keys"] == {"Bash.command": 1}
    assert claude["attachment_file_types"] == {"pdf": 1}
    assert claude["citation_detail_types"] == {"web_search_citation": 1}
    assert report["parsed"]["counters"]["conversations_read:claude"] == 1
