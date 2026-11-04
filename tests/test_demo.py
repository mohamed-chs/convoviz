"""The made-up export behind the README's example note and screenshots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import demo

from convoviz.chatgpt.parse import parse_conversation
from convoviz.diagnostics import Diagnostics
from convoviz.render import render_conversation

if TYPE_CHECKING:
    from convoviz.render import RenderOptions

EXAMPLE = Path(__file__).parent.parent / "docs" / "example" / "Planning a trip to Kyoto.md"


def test_readme_example_note_is_current(options: RenderOptions) -> None:
    conversation = parse_conversation(demo._kyoto(), Diagnostics())
    assert conversation is not None
    text = render_conversation(conversation, options, lambda _a: None).text
    if os.environ.get("UPDATE_GOLDEN") or not EXAMPLE.exists():
        EXAMPLE.write_text(text, encoding="utf-8")
    assert text == EXAMPLE.read_text(encoding="utf-8")


def test_demo_export_loads_cleanly(tmp_path: Path, load) -> None:
    export, diagnostics = load(demo.build(tmp_path / "demo.zip"))
    assert len(export.conversations) > 1000
    assert not diagnostics.errors
