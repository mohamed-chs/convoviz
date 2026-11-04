from __future__ import annotations

from convoviz.chatgpt.citations import render_text
from convoviz.diagnostics import Diagnostics
from convoviz.model import CITE_CLOSE, CITE_OPEN, Source

OPEN, SEP, CLOSE = "\ue200", "\ue202", "\ue201"


def _render(text: str, meta: dict | None = None):
    return render_text(text, meta or {}, Diagnostics())


def _cite(*ids: int) -> str:
    return CITE_OPEN + ",".join(map(str, ids)) + CITE_CLOSE


def test_content_references_become_sources() -> None:
    marker = f"{OPEN}cite{SEP}turn0search0{SEP}turn0search1{CLOSE}"
    text = f"Fact{marker}. More{marker}."
    ref = {
        "type": "grouped_webpages",
        "matched_text": marker,
        "start_idx": 4,
        "end_idx": 4 + len(marker),
        "items": [
            {
                "url": "https://a.org/x?utm_source=chatgpt.com",
                "title": "A",
                "supporting_websites": [{"url": "https://b.org", "title": "B"}],
            },
        ],
    }
    second = {**ref, "start_idx": text.rindex(marker), "end_idx": text.rindex(marker) + len(marker)}
    block = _render(text, {"content_references": [ref, second]})
    assert block.text == f"Fact{_cite(0, 1)}. More{_cite(0, 1)}."
    assert block.sources == (Source("https://a.org/x", "A"), Source("https://b.org", "B"))


def test_utf16_indices_are_understood() -> None:
    marker = f"{OPEN}cite{SEP}turn0search0{CLOSE}"
    text = f"😀 emoji first{marker}"
    utf16_start = len("😀 emoji first") + 1  # the emoji counts twice in UTF-16
    ref = {
        "type": "webpage",
        "matched_text": marker,
        "start_idx": utf16_start,
        "end_idx": utf16_start + len(marker),
        "url": "https://e.com",
        "title": "E",
    }
    block = _render(text, {"content_references": [ref]})
    assert block.text == f"😀 emoji first{_cite(0)}"


def test_wrong_indices_fall_back_to_matched_text() -> None:
    marker = f"{OPEN}cite{SEP}turn0search0{CLOSE}"
    ref = {
        "type": "webpage",
        "matched_text": marker,
        "start_idx": 999,
        "end_idx": 1010,
        "url": "https://e.com",
    }
    assert _render(f"x{marker}", {"content_references": [ref]}).text == f"x{_cite(0)}"


def test_markers_without_references_use_search_results() -> None:
    meta = {
        "search_result_groups": [
            {
                "entries": [
                    {
                        "url": "https://n.com",
                        "title": "N",
                        "ref_id": {"turn_index": 0, "ref_type": "news", "ref_index": 2},
                    },
                ]
            }
        ]
    }
    block = _render(
        f"Hi{OPEN}cite{SEP}turn0news2{CLOSE} and{OPEN}cite{SEP}turn9search9{CLOSE}.", meta
    )
    assert block.text == f"Hi{_cite(0)} and."
    assert block.sources == (Source("https://n.com", "N"),)


def test_entities_products_and_hidden_spans() -> None:
    entity = f'{OPEN}entity{SEP}["city","Paris","capital of France"]{CLOSE}'
    text = f"Visit {entity}.\ue203hidden-ish\ue204 {OPEN}navlist{SEP}x{SEP}y{CLOSE}"
    assert _render(text).text == "Visit Paris.hidden-ish "
    ref = {"type": "products", "matched_text": "PRODUCTS", "alt": "[Kettle]() and [Pot]()"}
    assert (
        _render("Buy PRODUCTS now", {"content_references": [ref]}).text == "Buy Kettle and Pot now"
    )


def test_file_citations() -> None:
    marker = f"{OPEN}filecite{SEP}turn0file0{CLOSE}"
    ref = {"type": "file", "matched_text": marker, "name": "notes.pdf"}
    block = _render(f"As noted{marker}.", {"content_references": [ref]})
    assert block.text == f"As noted{_cite(0)}."
    assert block.sources == (Source(None, "notes.pdf"),)


def test_legacy_citations_and_placeholders() -> None:
    text = "Old style【11†source】 claim."
    meta = {
        "citations": [
            {"start_ix": 9, "end_ix": 20, "metadata": {"url": "https://old.com", "title": "Old"}}
        ]
    }
    block = _render(text, meta)
    assert block.text == f"Old style{_cite(0)} claim."
    assert (
        _render("Leftover【3†source】 and citeturn0search1turn0news2 text").text
        == "Leftover and  text"
    )


def test_word_cite_is_never_stripped() -> None:
    assert _render("Please cite your sources.").text == "Please cite your sources."


def test_unknown_reference_types_are_reported() -> None:
    diagnostics = Diagnostics()
    ref = {"type": "hologram", "matched_text": "XX", "alt": "alt text"}
    block = render_text("see XX", {"content_references": [ref]}, diagnostics)
    assert block.text == "see alt text"
    assert diagnostics.unknown["content_reference:hologram"] == 1


def test_chart_widget_becomes_a_table() -> None:
    marker = "chartx"
    chart = {
        "meta": {"description": "Share by year"},
        "xKey": "year",
        "series": [{"dataKey": "v", "label": "Share", "valueSuffix": "%"}],
        "data": [{"year": 2025, "v": 10}, {"year": 2026, "v": 12}],
        "formattedData": [{"v": "10.0%"}],
    }
    ref = {
        "type": "client_defined_widget",
        "matched_text": marker,
        "data": {"widget_type": "charts_widget_v2", "content": chart},
    }
    block = render_text(f"Look:{marker}Done.", {"content_references": [ref]}, Diagnostics())
    assert block.text == (
        "Look:\n\nShare by year\n\n| year | Share |\n| --- | --- |\n"
        "| 2025 | 10.0% |\n| 2026 | 12% |\n\nDone."
    )


def test_image_groups_become_page_links() -> None:
    marker = "iturn0image1"
    ref = {
        "type": "image_group",
        "matched_text": marker,
        "images": [
            {"image_result": {"title": "A [cat]", "url": "https://p.example/cat"}},
            {"title": "no link"},
        ],
    }
    block = render_text(f"Cats{marker}", {"content_references": [ref]}, Diagnostics())
    assert block.text == "Cats\n\n- [A \\[cat\\]](<https://p.example/cat>)"


def test_safe_urls_and_sources_footnote() -> None:
    marker = "productsx"
    refs = [
        {"type": "product_link", "matched_text": marker, "safe_urls": ["https://shop.example/a"]},
        {
            "type": "sources_footnote",
            "sources": [
                {"title": "Shop", "url": "https://shop.example/a"},
                {"title": "Review", "url": "https://review.example/b"},
            ],
        },
    ]
    block = render_text(f"Buy{marker}.", {"content_references": refs}, Diagnostics())
    assert block.sources == (
        Source("https://shop.example/a", None),
        Source("https://review.example/b", "Review"),
    )
    assert block.text == f"Buy{CITE_OPEN}0{CITE_CLOSE}.\n\nSources: {CITE_OPEN}1{CITE_CLOSE}"


def test_product_carousels_list_their_products() -> None:
    payload = '{"selections":[["turn9product1","Fallback A"],["turn9product2","Only name"]]}'
    marker = f"{OPEN}products{SEP}{payload}{CLOSE}"
    refs = [
        {
            "type": "products",
            "matched_text": marker,
            "products": [
                {
                    "cite": "turn0product1",
                    "title": "Headphones A",
                    "price": "$99",
                    "merchants": "Shop.example - Seller",
                }
            ],
        }
    ]
    block = _render(f"Options:\n{marker}", {"content_references": refs})
    assert block.text == ("Options:\n\n- **Headphones A** — $99 · Shop.example\n- **Only name**")


def test_blank_lines_in_code_survive_tidying() -> None:
    block = render_text("Before\n\n\n\nAfter\n```\na\n\n\n\nb\n```", {}, Diagnostics())
    assert block.text == "Before\n\nAfter\n```\na\n\n\n\nb\n```"
