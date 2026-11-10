from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from factory import ConversationBuilder, thinking_turn

from convoviz.chatgpt.parse import parse_conversation
from convoviz.diagnostics import Diagnostics
from convoviz.model import (
    AttachmentBlock,
    CodeBlock,
    DocumentBlock,
    NoticeBlock,
    OutputBlock,
    QuoteBlock,
    ReasoningBlock,
    TextBlock,
)
from convoviz.tree import linearize


def _messages(builder: ConversationBuilder, diagnostics: Diagnostics | None = None):
    conv = parse_conversation(builder.build(), diagnostics or Diagnostics())
    assert conv is not None
    return linearize(conv, include_forks=False).messages()


def test_metadata() -> None:
    builder = ConversationBuilder(title="Hello", extra={"is_starred": True, "gizmo_id": "g-p-abc"})
    builder.user("hi")
    conv = parse_conversation(builder.build(), Diagnostics())
    assert conv is not None
    assert conv.title == "Hello"
    assert conv.created == datetime(2026, 1, 1, tzinfo=UTC)
    assert conv.url == f"https://chatgpt.com/c/{builder.conversation_id}"
    assert conv.project_id == "g-p-abc"


def test_unusable_conversations() -> None:
    diagnostics = Diagnostics()
    assert parse_conversation({"title": "x"}, diagnostics) is None
    assert parse_conversation({"id": "x", "mapping": []}, diagnostics) is None
    assert len(diagnostics.skipped) == 2


def test_hidden_and_system_messages_are_dropped() -> None:
    builder = ConversationBuilder()
    builder.add("system", "You are ChatGPT")
    builder.add(
        "user",
        {
            "content_type": "user_editable_context",
            "user_profile": "I am X",
            "user_instructions": "Be brief",
        },
    )
    builder.user("visible", metadata={"is_visually_hidden_from_conversation": False})
    builder.assistant("hidden", metadata={"is_visually_hidden_from_conversation": True})
    builder.assistant("answer")
    messages = _messages(builder)
    assert [m.text for m in messages] == ["visible", "answer"]


def test_text_parts_as_objects_and_missing_fields() -> None:
    builder = ConversationBuilder()
    builder.add(
        "user", {"content_type": "text", "parts": [{"content_type": "text", "text": "obj"}, " str"]}
    )
    builder.mapping["n1"]["message"].pop("create_time")
    builder.mapping["n1"]["message"]["metadata"] = None
    builder.mapping["n1"]["message"].pop("recipient")
    [message] = _messages(builder)
    assert message.text == "obj str"
    assert message.time is None


def test_reasoning_and_recap() -> None:
    builder = ConversationBuilder()
    builder.user("q")
    builder.add(
        "assistant",
        {"content_type": "thoughts", "thoughts": [{"summary": "Plan", "content": "Do it"}]},
    )
    builder.add("assistant", {"content_type": "reasoning_recap", "content": "Thought for 4s"})
    builder.assistant("a")
    messages = _messages(builder)
    assert messages[1].blocks == [ReasoningBlock("**Plan**\n\nDo it")]
    assert messages[2].blocks == [ReasoningBlock("", title="Thought for 4s")]


def test_thinking_preamble_folds_into_reasoning() -> None:
    builder = ConversationBuilder()
    builder.user("q")
    builder.assistant("Checking the docs first.", metadata={"is_thinking_preamble_message": True})
    builder.assistant("a")
    messages = _messages(builder)
    assert messages[1].blocks == [ReasoningBlock("Checking the docs first.")]
    assert messages[2].blocks == [TextBlock("a")]


def test_preamble_without_its_flag_is_recognized() -> None:
    builder = ConversationBuilder()
    thinking_turn(builder)
    diagnostics = Diagnostics()
    messages = _messages(builder, diagnostics)
    assert messages[2].blocks == [ReasoningBlock("Checking the docs.")]
    assert messages[5].blocks == [TextBlock("The answer.")]
    assert diagnostics.counters["inferred_preambles"] == 1


def test_scheduled_task_runs_are_marked() -> None:
    builder = ConversationBuilder()
    builder.user("Every morning, the weather")
    builder.assistant("Scheduled.")
    builder.assistant("Sunny.", metadata={"automation_id": "auto_1"})
    assert [m.scheduled for m in _messages(builder)] == [False, False, True]


def test_interactive_reply_uses_markdown_fallback() -> None:
    builder = ConversationBuilder()
    builder.user("q")
    builder.assistant(
        '<box>DIL.useState("x")</box>',
        metadata={"model_dil_v2": {"fallbackMarkdown": "Use <escape><</escape>`a`\n\n- one"}},
    )
    diagnostics = Diagnostics()
    messages = _messages(builder, diagnostics)
    assert messages[1].blocks == [TextBlock("Use <`a`\n\n- one")]
    assert diagnostics.counters["interactive_replies"] == 1


def test_writing_directive_becomes_a_document() -> None:
    builder = ConversationBuilder()
    builder.user("q")
    builder.assistant(':::writing{variant="social_post"}\nBig news!\n:::\nWant changes?')
    messages = _messages(builder)
    assert messages[1].blocks == [
        DocumentBlock("", "Big news!", kind="Social post"),
        TextBlock("Want changes?"),
    ]


def test_code_execution_and_output() -> None:
    builder = ConversationBuilder()
    builder.user("plot")
    builder.add(
        "assistant",
        {"content_type": "code", "language": "unknown", "text": "print(1)"},
        recipient="python",
    )
    builder.add(
        "tool",
        {"content_type": "execution_output", "text": "1"},
        name="python",
        metadata={
            "aggregate_result": {
                "messages": [{"message_type": "image", "image_url": "file-service://file-Chart1"}]
            }
        },
    )
    builder.assistant("done")
    messages = _messages(builder)
    assert isinstance(messages[1].blocks[0], CodeBlock)
    assert messages[1].role == "tool"
    assert messages[2].blocks[0] == OutputBlock("1", "python")
    assert isinstance(messages[2].blocks[1], AttachmentBlock)
    assert messages[2].blocks[1].attachment.file_id == "file-Chart1"


def test_plumbing_tool_calls_are_hidden_but_counted() -> None:
    builder = ConversationBuilder()
    diagnostics = Diagnostics()
    builder.user("search")
    builder.add("assistant", {"content_type": "code", "text": '{"q": "x"}'}, recipient="web.run")
    builder.add("tool", {"content_type": "text", "parts": ["results"]}, name="web.run")
    builder.add("tool", {"content_type": "text", "parts": ["ok"]}, name="bio")
    builder.assistant("answer")
    messages = _messages(builder, diagnostics)
    assert [type(b) for m in messages for b in m.blocks] == [TextBlock, NoticeBlock, TextBlock]
    assert diagnostics.counters["tool_call:web.run"] == 1


def test_quotes() -> None:
    builder = ConversationBuilder()
    builder.user("q")
    builder.add(
        "tool",
        {"content_type": "tether_quote", "text": "quoted", "title": "Page", "url": "https://e.com"},
    )
    [_, quote] = _messages(builder)
    assert quote.blocks == [QuoteBlock("quoted", "Page", "https://e.com")]


def test_canvas_create_and_regex_update_per_branch() -> None:
    builder = ConversationBuilder()
    builder.user("write")
    create = {"name": "Essay", "type": "document", "content": "Hello world. Bye world."}
    builder.add(
        "assistant",
        {"content_type": "code", "text": json.dumps(create)},
        recipient="canmore.create_textdoc",
    )
    fork = builder.assistant("created")
    update = {"updates": [{"pattern": "(w)orld", "replacement": "W$1rld!", "multiple": True}]}
    builder.add(
        "assistant",
        {"content_type": "code", "text": json.dumps(update)},
        recipient="canmore.update_textdoc",
    )
    builder.add(
        "assistant",
        {
            "content_type": "code",
            "text": json.dumps({"updates": [{"pattern": ".*", "replacement": "Other"}]}),
        },
        recipient="canmore.update_textdoc",
        parent=fork,
    )
    conv = parse_conversation(builder.build(), Diagnostics())
    assert conv is not None
    docs = {
        n.id: n.message.blocks[0]
        for n in conv.nodes.values()
        if n.message and isinstance(n.message.blocks[0], DocumentBlock)
    }
    assert docs["n2"].content == "Hello world. Bye world."
    assert docs["n4"].content == "Hello Wwrld!. Bye Wwrld!."
    assert docs["n4"].version == 2
    assert docs["n5"].content == "Other"  # sibling branch starts from the created version
    assert docs["n5"].version == 2


def test_attachments() -> None:
    builder = ConversationBuilder()
    image = {
        "content_type": "image_asset_pointer",
        "asset_pointer": "sediment://file_00000000aaaa",
        "width": 10,
        "height": 5,
    }
    builder.add(
        "user",
        {"content_type": "multimodal_text", "parts": [image, "what is this?"]},
        metadata={
            "attachments": [
                {"id": "file_00000000aaaa", "name": "photo.jpg"},
                {"id": "file-Doc1", "name": "notes.pdf", "mime_type": "application/pdf"},
            ]
        },
    )
    builder.add(
        "tool",
        {
            "content_type": "multimodal_text",
            "parts": [
                {
                    "content_type": "image_asset_pointer",
                    "asset_pointer": "sediment://file_gen1",
                    "metadata": {"dalle": {"prompt": "a cat"}},
                },
                "Generated image status text",
            ],
        },
        name="t2uay3k.sj1i4kz",
    )
    user, tool = _messages(builder)
    blocks = user.blocks
    assert isinstance(blocks[0], AttachmentBlock)
    assert blocks[0].attachment.name == "photo.jpg"
    assert blocks[0].attachment.width == 10
    assert blocks[1] == TextBlock("what is this?")
    assert isinstance(blocks[2], AttachmentBlock)
    assert blocks[2].attachment.kind == "file"
    assert len(tool.blocks) == 1
    assert isinstance(tool.blocks[0], AttachmentBlock)
    assert tool.blocks[0].caption == "a cat"


def test_image_prompt_comes_from_the_tool_call() -> None:
    builder = ConversationBuilder()
    builder.user("draw a fox")
    builder.add(
        "assistant",
        {
            "content_type": "text",
            "parts": ['{"prompt": "a red fox, watercolor", "size": "1024x1024"}'],
        },
        recipient="t2uay3k.sj1i4kz",
    )
    image = {"content_type": "image_asset_pointer", "asset_pointer": "sediment://file_fox"}
    builder.add(
        "tool", {"content_type": "multimodal_text", "parts": [image]}, name="t2uay3k.sj1i4kz"
    )
    _, tool = _messages(builder)
    assert isinstance(tool.blocks[0], AttachmentBlock)
    assert tool.blocks[0].caption == "a red fox, watercolor"


def test_voice_messages() -> None:
    builder = ConversationBuilder()
    builder.add(
        "user",
        {
            "content_type": "multimodal_text",
            "parts": [
                {"content_type": "audio_transcription", "text": "spoken words"},
                {
                    "content_type": "real_time_user_audio_video_asset_pointer",
                    "audio_asset_pointer": {
                        "content_type": "audio_asset_pointer",
                        "asset_pointer": "sediment://file_audio1",
                    },
                },
            ],
        },
    )
    [message] = _messages(builder)
    assert message.blocks[0] == TextBlock("spoken words")
    assert isinstance(message.blocks[1], AttachmentBlock)
    assert message.blocks[1].attachment.kind == "audio"


def test_unknown_content_is_kept_and_reported() -> None:
    builder = ConversationBuilder()
    diagnostics = Diagnostics()
    builder.add("assistant", {"content_type": "brand_new_thing", "text": "still readable"})
    builder.add(
        "user", {"content_type": "multimodal_text", "parts": [{"content_type": "hologram"}, "x"]}
    )
    messages = _messages(builder, diagnostics)
    assert messages[0].text == "still readable"
    assert diagnostics.unknown


def test_timestamps_in_various_shapes() -> None:
    builder = ConversationBuilder()
    builder.user("ms", time=1_767_225_600_000)
    builder.user("iso")
    builder.mapping["n2"]["message"]["create_time"] = "2026-01-02T00:00:00Z"
    first, second = _messages(builder)
    assert first.time == datetime(2026, 1, 1, tzinfo=UTC)
    assert second.time == datetime(2026, 1, 2, tzinfo=UTC)


@pytest.mark.parametrize(
    ("content", "pattern", "replacement", "multiple", "expected"),
    [
        ("a1 a2", r"a(\d)", r"b\1", True, "b1 b2"),  # Python references
        ("a1 a2", r"a(\d)", "b$1", False, "b1 a2"),  # ECMAScript references
        ("x", "x", "$$&[$&]", False, "$&[x]"),
        ("x", "(?P<n>x)", r"\g<n>$<n>", False, "xx"),
        ("x", "x", r"\frac{1}{2} \n $9", False, r"\frac{1}{2} \n $9"),  # literal otherwise
        ("Hello you\nKeep me", "Hello.*", "Hi", False, "Hi\nKeep me"),  # . stops at newlines
        ("a(b a(b", "a(b", "c", True, "c c"),  # invalid regex: literal text
        ("old\nlines", ".*", "new", False, "new"),
    ],
)
def test_canvas_update_dialects(
    content: str, pattern: str, replacement: str, multiple: bool, expected: str
) -> None:
    from convoviz.chatgpt.parse import _apply_canvas_update

    assert _apply_canvas_update(content, pattern, replacement, multiple=multiple) == expected
