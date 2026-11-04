from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from convoviz.fsnames import MAX_STEM_BYTES, safe_stem, slug

UNSAFE = set('<>:"/\\|?*[]#^')


def test_unicode_titles_are_kept() -> None:
    assert safe_stem("Привет, мир") == "Привет, мир"
    assert safe_stem("数据分析 🚀") == "数据分析 🚀"
    assert safe_stem("Café") == "Café"


def test_unsafe_characters_are_removed() -> None:
    assert safe_stem('a/b\\c:d*e?f"g<h>i|j[k]l#m^n') == "a b c d e f g h i j k l m n"
    assert safe_stem("bidi\u202etext\u200d ok") == "biditext ok"
    assert safe_stem("  trailing dots... ") == "trailing dots"


def test_reserved_and_empty_names() -> None:
    assert safe_stem("CON") == "CON_"
    assert safe_stem("nul.txt") == "nul.txt_"
    assert safe_stem("???") == "Untitled"
    assert safe_stem("") == "Untitled"


def test_long_titles_are_cut_at_words_within_bytes() -> None:
    title = "word " * 100
    stem = safe_stem(title)
    assert len(stem.encode()) <= MAX_STEM_BYTES + len("…".encode())
    assert stem.endswith("word…")


def test_slug() -> None:
    assert slug("My Report (final)") == "My-Report-final"
    assert " " not in slug("a b c")
    assert slug("???") == ""


@given(st.text(max_size=300))
def test_safe_stem_is_always_safe(title: str) -> None:
    stem = safe_stem(title)
    assert stem
    assert not UNSAFE & set(stem)
    assert len(stem.encode()) <= MAX_STEM_BYTES + 3
    assert stem == stem.strip(" .") or stem.endswith("_")
    assert all(ord(c) >= 32 for c in stem)
