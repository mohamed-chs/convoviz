from __future__ import annotations

from datetime import UTC, date

from factory import ConversationBuilder

from convoviz.chatgpt.parse import parse_conversation
from convoviz.diagnostics import Diagnostics
from convoviz.insights import render_insights
from convoviz.stats import compute, distinctive_words, tokenize

DAY = 86_400


def _conversations():
    out = []
    for n in range(6):
        b = ConversationBuilder(
            title=f"c{n}", conversation_id=f"id-{n}", create_time=1_767_225_600 + n * 40 * DAY
        )
        b.user(
            "python pandas dataframe question " * 3 if n < 3 else "garden tomatoes watering " * 3
        )
        b.add("assistant", {"content_type": "thoughts", "thoughts": [{"content": "x"}]})
        b.assistant("answer", model="gpt-5-2" if n % 2 else "gpt-6")
        out.append(parse_conversation(b.build(), Diagnostics()))
    return [c for c in out if c is not None]


def test_stats() -> None:
    stats = compute(_conversations(), UTC)
    assert stats.conversations == 6
    assert stats.prompts == 6
    assert stats.replies == 12
    assert stats.first == date(2026, 1, 1)
    assert stats.features["reasoning"] == 6
    assert stats.longest_streak == 1
    assert sum(stats.lengths.values()) == 6
    assert stats.months()[0] == "2026-01"
    assert sum(stats.per_month.values()) == 6


def test_tokenize_drops_code_urls_and_stopwords() -> None:
    words = tokenize(
        "Please explain `foo_bar` and https://x.com/a in Python, déjà vu ```\ncode\n```"
    )
    assert words == ["explain", "python", "déjà"]


def test_distinctive_words() -> None:
    stats = compute(_conversations(), UTC)
    words = distinctive_words(stats)
    assert {w for w, _ in words[2026]} >= {"python", "garden"}


def test_report_is_self_contained_and_escaped() -> None:
    conversations = _conversations()
    b = ConversationBuilder(conversation_id="evil")
    b.user("<script>alert(1)</script> " * 3)
    evil = parse_conversation(b.build(), Diagnostics())
    assert evil is not None
    html = render_insights(compute([*conversations, evil], UTC))
    assert html.startswith("<!doctype html>")
    assert "<script>alert" not in html
    for section in (
        "Activity",
        "When you ask",
        "Prompts per month",
        "Models",
        "Features",
        "Conversation length",
    ):
        assert f"<h2>{section}</h2>" in html
    assert "http://" not in html.replace("http://www.w3.org", "")
    assert "https://" not in html


def test_report_with_no_data() -> None:
    html = render_insights(compute([], UTC))
    assert "No messages found." in html
