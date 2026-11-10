"""Random, export-shaped JSON must never crash parsing, rendering or statistics.

The parser isolates exceptions per message, so a bug would surface only as a
diagnostic error. These tests treat any such error as a failure.
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from convoviz.chatgpt.parse import parse_conversation
from convoviz.claude.parse import parse_conversation as parse_claude
from convoviz.diagnostics import Diagnostics
from convoviz.render import RenderOptions, render_conversation
from convoviz.stats import compute

_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**40), max_value=2**53),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=30),
)
_JSON = st.recursive(
    _SCALARS,
    lambda inner: (
        st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=8), inner, max_size=4)
    ),
    max_leaves=12,
)


def _maybe(strategy: st.SearchStrategy[Any]) -> st.SearchStrategy[Any]:
    return st.one_of(strategy, _JSON)


_CONTENT_TYPES = st.sampled_from(
    [
        "text",
        "multimodal_text",
        "code",
        "execution_output",
        "thoughts",
        "reasoning_recap",
        "tether_quote",
        "tether_browsing_display",
        "user_editable_context",
        "computer_output",
        "something_new",
    ]
)
_PART = st.one_of(
    st.text(max_size=40),
    st.fixed_dictionaries(
        {
            "content_type": _maybe(
                st.sampled_from(
                    [
                        "image_asset_pointer",
                        "audio_transcription",
                        "text",
                        "real_time_user_audio_video_asset_pointer",
                        "x",
                    ]
                )
            ),
            "asset_pointer": _maybe(
                st.sampled_from(["sediment://file_00aa", "file-service://file-X"])
            ),
            "text": _maybe(st.text(max_size=20)),
        }
    ),
    _JSON,
)
_TEXTISH = st.text(
    alphabet=st.sampled_from(
        [*list("ab #$`~\\()[]\n<>"), "\ue200", "\ue201", "\ue202", "【", "†", "】"]
    ),
    max_size=60,
)
_MESSAGE = st.fixed_dictionaries(
    {
        "author": _maybe(
            st.fixed_dictionaries(
                {
                    "role": _maybe(st.sampled_from(["user", "assistant", "tool", "system"])),
                    "name": _maybe(st.sampled_from(["python", "bio", "web"])),
                }
            )
        ),
        "content": _maybe(
            st.fixed_dictionaries(
                {
                    "content_type": _maybe(_CONTENT_TYPES),
                    "parts": _maybe(st.lists(st.one_of(_TEXTISH, _PART), max_size=3)),
                    "text": _maybe(_TEXTISH),
                    "thoughts": _maybe(
                        st.lists(
                            _maybe(
                                st.fixed_dictionaries({"summary": _TEXTISH, "content": _TEXTISH})
                            ),
                            max_size=2,
                        )
                    ),
                }
            )
        ),
        "recipient": _maybe(
            st.sampled_from(
                ["all", "python", "canmore.create_textdoc", "canmore.update_textdoc", "web.run"]
            )
        ),
        "create_time": _maybe(st.floats(min_value=1.6e9, max_value=1.8e9)),
        "metadata": _maybe(
            st.fixed_dictionaries(
                {
                    "content_references": _maybe(st.lists(_JSON, max_size=2)),
                    "citations": _maybe(st.lists(_JSON, max_size=2)),
                    "attachments": _maybe(st.lists(_JSON, max_size=2)),
                }
            )
        ),
    }
)


@st.composite
def conversations(draw: st.DrawFn) -> dict[str, Any]:
    count = draw(st.integers(min_value=0, max_value=6))
    ids = [f"n{i}" for i in range(count)]
    mapping: dict[str, Any] = {}
    for i, node_id in enumerate(ids):
        parent = draw(st.one_of(st.none(), st.sampled_from(ids))) if i else None
        mapping[node_id] = {
            "id": node_id,
            "parent": parent,
            "message": draw(_maybe(_MESSAGE)),
        }
    return {
        "id": "c1",
        "title": draw(_maybe(st.text(max_size=20))),
        "create_time": draw(_maybe(st.floats(min_value=1.6e9, max_value=1.8e9))),
        "update_time": draw(_JSON),
        "mapping": mapping,
        "current_node": draw(_maybe(st.sampled_from([*ids, "missing"]) if ids else st.none())),
    }


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(conversations())
def test_random_conversations_never_crash(raw: dict[str, Any]) -> None:
    diagnostics = Diagnostics()
    conversation = parse_conversation(raw, diagnostics)
    assert not diagnostics.errors, diagnostics.errors
    if conversation is None:
        return
    for branches in ("all", "active"):
        options = RenderOptions(timezone=UTC, branches=branches)
        render_conversation(conversation, options, lambda _a: "x.png")
    compute([conversation], UTC)


_CLAUDE_TOOLS = st.sampled_from(
    [
        "artifacts",
        "create_file",
        "str_replace",
        "Write",
        "Edit",
        "Bash",
        "repl",
        "present_files",
        "visualize:show_widget",
        "memory_user_edits",
        "GitHub:create_issue",
        "mcp__x__y",
        "web_search",
        "",
    ]
)
_CLAUDE_INPUT = st.fixed_dictionaries(
    {
        "command": _maybe(st.sampled_from(["create", "update", "rewrite", "view", "add"])),
        "id": _maybe(st.sampled_from(["a", "b"])),
        "type": _maybe(st.sampled_from(["text/markdown", "application/vnd.ant.code", "x"])),
        "content": _maybe(_TEXTISH),
        "old_str": _maybe(_TEXTISH),
        "new_str": _maybe(_TEXTISH),
        "path": _maybe(st.sampled_from(["/a/x.py", "/a/y.md", ""])),
        "file_text": _maybe(_TEXTISH),
        "widget_code": _maybe(_TEXTISH),
    }
)
_CLAUDE_BLOCK = st.one_of(
    st.fixed_dictionaries(
        {
            "type": _maybe(st.sampled_from(["text", "thinking", "voice_note", "x", "image"])),
            "text": _maybe(_TEXTISH),
            "thinking": _maybe(_TEXTISH),
            "summaries": _maybe(st.lists(_JSON, max_size=2)),
            "citations": _maybe(
                st.lists(
                    st.fixed_dictionaries(
                        {
                            "end_index": _maybe(st.integers(min_value=-5, max_value=80)),
                            "details": _maybe(st.fixed_dictionaries({"url": _maybe(_TEXTISH)})),
                        }
                    ),
                    max_size=3,
                )
            ),
        }
    ),
    st.fixed_dictionaries(
        {
            "type": st.just("tool_use"),
            "id": _maybe(st.sampled_from(["t1", "t2"])),
            "name": _maybe(_CLAUDE_TOOLS),
            "input": _maybe(_CLAUDE_INPUT),
        }
    ),
    st.fixed_dictionaries(
        {
            "type": st.just("tool_result"),
            "tool_use_id": _maybe(st.sampled_from(["t1", "t2"])),
            "name": _maybe(_CLAUDE_TOOLS),
            "content": _maybe(st.lists(_JSON, max_size=3)),
            "is_error": _maybe(st.booleans()),
        }
    ),
    _JSON,
)


@st.composite
def claude_conversations(draw: st.DrawFn) -> dict[str, Any]:
    count = draw(st.integers(min_value=0, max_value=6))
    ids = [f"m{i}" for i in range(count)]
    messages = [
        {
            "uuid": draw(_maybe(st.just(message_id))),
            "sender": draw(_maybe(st.sampled_from(["human", "assistant"]))),
            "text": draw(_maybe(_TEXTISH)),
            "content": draw(_maybe(st.lists(_CLAUDE_BLOCK, max_size=4))),
            "created_at": draw(_maybe(st.sampled_from(["2026-01-01T00:00:00Z", "bad"]))),
            "parent_message_uuid": draw(
                _maybe(st.sampled_from([*ids, "00000000-0000-4000-8000-000000000000", "gone"]))
            ),
            "attachments": draw(_maybe(st.lists(_JSON, max_size=2))),
            "files": draw(_maybe(st.lists(_JSON, max_size=2))),
        }
        for message_id in ids
    ]
    return {
        "uuid": "c1",
        "name": draw(_maybe(_TEXTISH)),
        "created_at": draw(_JSON),
        "chat_messages": messages,
    }


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(claude_conversations())
def test_random_claude_conversations_never_crash(raw: dict[str, Any]) -> None:
    diagnostics = Diagnostics()
    conversation = parse_claude(raw, diagnostics)
    assert not diagnostics.errors, diagnostics.errors
    if conversation is None:
        return
    for branches in ("all", "active"):
        options = RenderOptions(timezone=UTC, branches=branches)
        render_conversation(conversation, options, lambda _a: "x.png", lambda p: p)
    compute([conversation], UTC)


@st.composite
def cited_texts(draw: st.DrawFn) -> tuple[str, dict[str, Any]]:
    text = draw(_TEXTISH)
    refs = []
    for _ in range(draw(st.integers(0, 3))):
        start = draw(st.integers(-2, len(text) + 2))
        end = draw(st.integers(-2, len(text) + 2))
        matched = text[max(0, start) : max(0, end)] if draw(st.booleans()) else draw(_TEXTISH)
        refs.append(
            {
                "type": draw(
                    _maybe(st.sampled_from(["grouped_webpages", "webpage", "image_v2", "x"]))
                ),
                "matched_text": draw(_maybe(st.just(matched))),
                "start_idx": draw(_maybe(st.just(start))),
                "end_idx": draw(_maybe(st.just(end))),
                "items": draw(
                    _maybe(
                        st.lists(
                            _maybe(st.fixed_dictionaries({"url": _TEXTISH, "title": _TEXTISH})),
                            max_size=2,
                        )
                    )
                ),
                "alt": draw(_maybe(_TEXTISH)),
            }
        )
    meta = {
        "content_references": refs,
        "citations": draw(
            _maybe(
                st.lists(
                    st.fixed_dictionaries(
                        {
                            "start_ix": st.integers(-1, 70),
                            "end_ix": st.integers(-1, 70),
                            "metadata": _JSON,
                        }
                    ),
                    max_size=2,
                )
            )
        ),
        "search_result_groups": draw(_JSON),
    }
    return text, meta


@settings(deadline=None)
@given(cited_texts())
def test_random_citations_never_crash(case: tuple[str, dict[str, Any]]) -> None:
    from convoviz.chatgpt.citations import render_text
    from convoviz.model import strip_citations

    text, meta = case
    diagnostics = Diagnostics()
    block = render_text(text, meta, diagnostics)
    assert not diagnostics.errors
    assert "\ue200" not in strip_citations(block.text)
