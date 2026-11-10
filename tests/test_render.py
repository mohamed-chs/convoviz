from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from factory import ConversationBuilder, thinking_turn

from convoviz.chatgpt.parse import parse_conversation
from convoviz.diagnostics import Diagnostics
from convoviz.render import RenderOptions, render_conversation

GOLDEN = Path(__file__).parent / "golden"
OPEN, SEP, CLOSE = "\ue200", "\ue202", "\ue201"


def rich_conversation() -> dict:
    """One conversation that exercises most of what the renderer handles."""
    b = ConversationBuilder(title="Trip: Paris / Rome?", extra={"is_starred": True})
    b.add("system", "hidden")
    b.user("Plan a trip for <b>$500</b>.\n\n# Not a heading")
    b.add(
        "assistant",
        {
            "content_type": "thoughts",
            "thoughts": [{"summary": "Budget", "content": "Cheap options."}],
        },
    )
    b.add("assistant", {"content_type": "reasoning_recap", "content": "Thought for 5s"})
    marker = f"{OPEN}cite{SEP}turn0search0{CLOSE}"
    text = (
        f"# Plan\nTrains are cheap{marker}. The cost is \\(n \\times 20\\).\n```python\nprint('$5')"
    )
    answer = b.assistant(
        text,
        metadata={
            "content_references": [
                {
                    "type": "grouped_webpages",
                    "matched_text": marker,
                    "start_idx": text.index(marker),
                    "end_idx": text.index(marker) + len(marker),
                    "items": [
                        {
                            "url": "https://rail.example/fares?utm_source=chatgpt.com",
                            "title": "Rail fares",
                            "attribution": "rail.example",
                        }
                    ],
                }
            ]
        },
    )
    b.user("Shorter please", parent=answer)
    b.assistant("Take the train.")
    b.user("Make a doc", parent=answer)
    b.add(
        "assistant",
        {
            "content_type": "code",
            "text": json.dumps(
                {"name": "Itinerary", "type": "document", "content": "Day 1: Paris"}
            ),
        },
        recipient="canmore.create_textdoc",
    )
    b.add("assistant", {"content_type": "code", "text": "sum([1, 2])"}, recipient="python")
    b.add("tool", {"content_type": "execution_output", "text": "3"}, name="python")
    b.add(
        "tool",
        {
            "content_type": "multimodal_text",
            "parts": [
                {"content_type": "image_asset_pointer", "asset_pointer": "sediment://file_missing"}
            ],
        },
    )
    later = f"See again{marker}."
    b.assistant(
        later,
        model="gpt-5-5-thinking",
        metadata={
            "content_references": [
                {
                    "type": "webpage",
                    "matched_text": marker,
                    "start_idx": len("See again"),
                    "end_idx": len(later) - 1,
                    "url": "https://rail.example/fares",
                    "title": "Rail fares",
                }
            ]
        },
    )
    b.user("Email the group and suggest a bag")
    b.assistant("Comparing bags first.", metadata={"is_thinking_preamble_message": True})
    products = f'{OPEN}products{SEP}{{"selections":[["turn1product1","Day pack"]]}}{CLOSE}'
    chart = "{{chart}}"
    final = (
        ':::writing{variant="email" subject="Trip"}\nHi all,\n\nTrains it is.\n:::\n'
        f"Bags:\n{products}\nPrices by city:{chart}"
    )
    b.assistant(
        final,
        metadata={
            "content_references": [
                {
                    "type": "products",
                    "matched_text": products,
                    "products": [
                        {
                            "cite": "turn1product1",
                            "title": "Day pack",
                            "price": "€40",
                            "merchants": "Bags.example - Store",
                        }
                    ],
                },
                {
                    "type": "client_defined_widget",
                    "matched_text": chart,
                    "data": {
                        "widget_type": "charts_widget_v2",
                        "content": {
                            "xKey": "city",
                            "series": [{"dataKey": "eur", "label": "Price", "valueSuffix": "€"}],
                            "data": [{"city": "Paris", "eur": 30}, {"city": "Rome", "eur": 25}],
                        },
                    },
                },
            ]
        },
    )
    return b.build()


def _render(raw: dict, options: RenderOptions, resolve=lambda _a: None) -> str:
    conv = parse_conversation(raw, Diagnostics())
    assert conv is not None
    return render_conversation(conv, options, resolve).text


def test_golden_rich_conversation(options: RenderOptions) -> None:
    text = _render(rich_conversation(), options)
    golden = GOLDEN / "rich.md"
    if os.environ.get("UPDATE_GOLDEN") or not golden.exists():
        golden.write_text(text, encoding="utf-8")
    assert text == golden.read_text(encoding="utf-8")


def test_structure(options: RenderOptions) -> None:
    text = _render(rich_conversation(), options)
    assert text.startswith('---\ntitle: "Trip: Paris / Rome?"\n')
    assert "starred: true" in text
    assert "\n# Trip: Paris / Rome?\n" in text
    assert "### Plan" in text  # message headings sit below turn headings
    assert "\n### Not a heading\n" in text
    assert "&lt;b>\\$500&lt;/b>" in text
    assert "$n \\times 20$" in text
    assert "print('$5')\n```" in text  # dangling fence closed, code untouched
    assert "<summary>Thought for 5s</summary>" in text
    assert "<summary>↳ 1 other version of the next part</summary>" in text
    assert "*[Image not included in the export]*" in text
    assert "<details open>\n<summary>🗎 Canvas: Itinerary</summary>" in text


def test_footnotes_are_numbered_once_per_conversation(options: RenderOptions) -> None:
    text = _render(rich_conversation(), options)
    assert text.count("[^1]:") == 1
    assert "[^2]" not in text
    assert "Trains are cheap[^1]." in text
    assert "See again[^1]." in text
    assert "[^1]: [Rail fares](https://rail.example/fares) — rail.example" in text


def test_active_branch_only(options: RenderOptions) -> None:
    text = _render(rich_conversation(), replace(options, branches="active"))
    assert "other version" not in text
    assert "Take the train." not in text  # an older sibling branch
    assert "Make a doc" in text


def test_options_hide_reasoning_and_tools(options: RenderOptions) -> None:
    text = _render(
        rich_conversation(),
        replace(
            options,
            reasoning=False,
            tool_calls=False,
            timestamps=False,
            user_label="Me",
            assistant_label="AI",
            tags=(),
        ),
    )
    assert "Thought for" not in text
    assert "sum([1, 2])" not in text
    assert "## Me\n" in text
    assert "## AI\n" in text
    assert "tags:" not in text


def test_attachments_link_through_resolver(options: RenderOptions) -> None:
    b = ConversationBuilder()
    b.add(
        "user",
        {
            "content_type": "multimodal_text",
            "parts": [
                {"content_type": "image_asset_pointer", "asset_pointer": "sediment://file_1"}
            ],
        },
        metadata={
            "attachments": [
                {"id": "file_1", "name": "my photo.png"},
                {"id": "file_2", "name": "a.pdf"},
            ]
        },
    )
    text = _render(b.build(), options, lambda a: f"../attachments/{a.name}")
    assert "![my photo.png](../attachments/my%20photo.png)" in text
    assert "[📎 a.pdf](../attachments/a.pdf)" in text


def test_message_count_counts_turns(options: RenderOptions) -> None:
    b = ConversationBuilder()
    b.user("q")
    b.add("assistant", {"content_type": "thoughts", "thoughts": [{"content": "hmm"}]})
    b.assistant("a")
    b.user("q2")
    b.assistant("a2")
    conv = parse_conversation(b.build(), Diagnostics())
    assert conv is not None
    assert render_conversation(conv, options, lambda _a: None).messages == 4


def test_folded_summaries_use_html_escaping_and_urls_stay_links(options: RenderOptions) -> None:
    b = ConversationBuilder()
    b.user("Draft it")
    b.add(
        "assistant",
        {
            "content_type": "code",
            "text": json.dumps({"name": "Notes_v2 & <draft>", "type": "document", "content": "x"}),
        },
        recipient="canmore.create_textdoc",
    )
    marker = f"{OPEN}cite{SEP}turn0search0{CLOSE}"
    text = f"Done{marker}."
    b.assistant(
        text,
        metadata={
            "content_references": [
                {
                    "type": "webpage",
                    "matched_text": marker,
                    "start_idx": 4,
                    "end_idx": 4 + len(marker),
                    "url": "https://ex.example/a b(c",
                    "title": "Page",
                }
            ]
        },
    )
    note = _render(b.build(), options)
    assert "<summary>🗎 Canvas: Notes_v2 &amp; &lt;draft&gt;</summary>" in note
    assert "[^1]: [Page](https://ex.example/a%20b%28c)" in note


def test_inline_drafts_are_labelled(options: RenderOptions) -> None:
    builder = ConversationBuilder()
    builder.user("write it")
    builder.assistant(':::writing{variant="email" subject="Q3 <plan>"}\nHi team,\n\n# Agenda\n:::')
    text = _render(builder.build(), options)
    assert (
        "<details open>\n<summary>🗎 Email: Q3 &lt;plan&gt;</summary>\n\nHi team,\n\n### Agenda"
        in text
    )


def test_a_thinking_turn_is_one_panel(options: RenderOptions) -> None:
    builder = ConversationBuilder()
    thinking_turn(builder)
    note = _render(builder.build(), options)
    assert note.count("<details>") == 1
    assert "<summary>Thought for 9s</summary>" in note
    assert note.index("Checking the docs.") < note.index("</details>") < note.index("The answer.")


def test_a_recap_closes_the_panel(options: RenderOptions) -> None:
    builder = ConversationBuilder()
    builder.user("q")
    for recap in ("Thought for 3s", "Thought for 5s"):
        builder.add("assistant", {"content_type": "thoughts", "thoughts": [{"content": "hm"}]})
        builder.add("assistant", {"content_type": "reasoning_recap", "content": recap})
    builder.assistant("a")
    note = _render(builder.build(), options)
    assert "<summary>Thought for 3s</summary>" in note
    assert "<summary>Thought for 5s</summary>" in note


def test_each_scheduled_run_is_its_own_reply(options: RenderOptions) -> None:
    builder = ConversationBuilder()
    builder.user("Every morning, the weather")
    builder.assistant("Scheduled.")
    builder.assistant("Sunny.", metadata={"automation_id": "auto_1"})
    builder.assistant("Rain.", metadata={"automation_id": "auto_1"})
    assert _render(builder.build(), options).count("## ChatGPT") == 3
