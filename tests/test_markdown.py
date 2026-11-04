from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from convoviz.chatgpt.textfix import clean_assistant_text
from convoviz.markdown import (
    close_fences,
    convert_math,
    escape_currency,
    escape_html,
    fence_for,
    map_prose,
    split_fenced,
)


def test_math_is_converted_outside_code_only() -> None:
    text = "Inline \\(a+b\\) and\n\\[\nx^2\n\\]\n```\n\\(keep\\)\n```\n`\\(also\\)`"
    assert convert_math(text) == "Inline $a+b$ and\n$$\nx^2\n$$\n```\n\\(keep\\)\n```\n`\\(also\\)`"


def test_currency_is_escaped() -> None:
    assert escape_currency("It costs $5 or $10.") == "It costs \\$5 or \\$10."
    assert escape_currency("Math $x$ stays, `$5` in code too") == "Math $x$ stays, `$5` in code too"


def test_html_is_neutralized_in_prose() -> None:
    assert escape_html("<script>x</script> 3 < 4") == "&lt;script>x&lt;/script> 3 < 4"
    assert escape_html("```html\n<b>\n```") == "```html\n<b>\n```"


def test_dangling_fence_is_closed() -> None:
    assert close_fences("text\n```python\ncode") == "text\n```python\ncode\n```"
    closed = "```\ncode\n```"
    assert close_fences(closed) == closed
    assert close_fences("````\n```\n") == "````\n```\n````"


def test_fence_for_nested_backticks() -> None:
    assert fence_for("no fences") == "```"
    assert fence_for("has ```` inside") == "`````"


def test_split_fenced_handles_tildes() -> None:
    chunks = split_fenced("a\n~~~\n```\n~~~\nb")
    assert chunks == [(False, "a\n"), (True, "~~~\n```\n~~~\n"), (False, "b")]


def test_canvas_directives_and_sandbox_links() -> None:
    text = ':::writing{variant="email" id="1"}\nDear X\n:::\nSee [report](sandbox:/mnt/data/r.pdf).'
    assert clean_assistant_text(text) == "\nDear X\n\nSee `report`."


@given(st.text())
def test_map_prose_identity_round_trips(text: str) -> None:
    assert map_prose(text, lambda s: s) == text


@given(st.text(alphabet=st.sampled_from(["`", "~", "\n", "a", " "]), max_size=60))
def test_close_fences_is_idempotent(text: str) -> None:
    once = close_fences(text)
    assert close_fences(once) == once
    assert (
        not split_fenced(once)
        or not split_fenced(once)[-1][0]
        or once.rstrip().endswith(("```", "~~~"))
    )
