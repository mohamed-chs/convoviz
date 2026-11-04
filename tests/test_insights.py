from __future__ import annotations

from datetime import UTC, date

from factory import ConversationBuilder

from convoviz.chatgpt.parse import parse_conversation
from convoviz.diagnostics import Diagnostics
from convoviz.insights import render_insights
from convoviz.stats import compute, distinctive_words, frequent_words, tokenize

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
    assert words == ["python", "déjà"]


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


def test_word_clouds_have_all_years_and_one_tab_per_year() -> None:
    conversations = _conversations()
    late = ConversationBuilder(conversation_id="late", create_time=1_798_761_600)  # 2027
    late.user("sourdough starter hydration " * 3)
    parsed = parse_conversation(late.build(), Diagnostics())
    assert parsed is not None
    stats = compute([*conversations, parsed], UTC)
    assert frequent_words(stats)[0] == ("python", 9)
    html = render_insights(stats)
    assert "<h2>What you asked about</h2>" in html
    assert html.count('role="tab"') == 3  # All years, 2027, 2026
    assert html.count('class="cloud"') == 3
    assert 'data-name="All years"' in html
    assert '<span data-w="1.000" data-n="9">python</span>' in html
    assert "Save image" in html


def test_single_year_has_one_cloud_without_tabs() -> None:
    html = render_insights(compute(_conversations(), UTC))
    assert html.count('class="cloud"') == 1
    assert 'role="tab"' not in html
