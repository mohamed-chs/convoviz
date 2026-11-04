from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from convoviz.chatgpt.textfix import Directive, clean_assistant_text, split_directives
from convoviz.markdown import (
    close_fences,
    convert_math,
    currency_escaped,
    escape_currency,
    escape_html,
    fence_for,
    html_escaped,
    map_prose,
    math_to_dollars,
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


def test_sandbox_links() -> None:
    assert clean_assistant_text("See [report](sandbox:/mnt/data/r.pdf).") == "See `report`."


def test_writing_directives_split_out() -> None:
    text = (
        'Here:\n:::writing{variant="email" subject="Hi" id="1"}\nDear X\n'
        "```\n:::\n```\n:::\nBye\n```\n:::other{}\n```\n:::writing{}\nopen"
    )
    assert split_directives(text) == [
        "Here:\n",
        Directive("Email", "Hi", "Dear X\n```\n:::\n```"),
        "Bye\n```\n:::other{}\n```\n",
        Directive("Canvas", "", "open"),
    ]
    assert split_directives("a ::: b") == ["a ::: b"]


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


_MARKDOWNISH = st.text(
    alphabet=st.sampled_from(
        ["`", "~", "\n", "a", " ", "$", "5", "\\", "(", ")", "[", "]", "<", "b"]
    ),
    max_size=80,
)


@given(_MARKDOWNISH)
def test_one_pass_equals_the_chain_of_passes(text: str) -> None:
    chained = convert_math(escape_currency(escape_html(text)))
    assert map_prose(text, html_escaped, currency_escaped, math_to_dollars) == chained


def _split_fenced_reference(text: str) -> list[tuple[bool, str]]:
    """The straightforward line-by-line definition the fast version must match."""
    import re

    fence_re = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
    chunks: list[tuple[bool, str]] = []
    buffer: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        line_nl = line + "\n"
        match = fence_re.match(line)
        if fence is None:
            if match and not (match.group(2)[0] == "`" and "`" in match.group(3)):
                if buffer:
                    chunks.append((False, "".join(buffer)))
                buffer, fence = [line_nl], match.group(2)
            else:
                buffer.append(line_nl)
        else:
            buffer.append(line_nl)
            if (
                match
                and match.group(2)[0] == fence[0]
                and len(match.group(2)) >= len(fence)
                and not match.group(3).strip()
            ):
                chunks.append((True, "".join(buffer)))
                buffer, fence = [], None
    if buffer:
        chunks.append((fence is not None, "".join(buffer)))
    # Undo the newline added to the last line.
    if chunks:
        is_code, last = chunks[-1]
        chunks[-1] = (is_code, last[:-1])
        if not chunks[-1][1]:
            chunks.pop()
    return chunks


@given(st.text(alphabet=st.sampled_from(["`", "~", "\n", "a", " "]), max_size=80))
def test_split_fenced_matches_reference(text: str) -> None:
    assert split_fenced(text) == _split_fenced_reference(text)
