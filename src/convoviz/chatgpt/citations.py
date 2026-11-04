r"""Resolve ChatGPT's inline citation markers into neutral source references.

Assistant text carries private-use "widget" markers::

    \ue200 KIND \ue202 PAYLOAD (\ue202 PAYLOAD)* \ue201

``cite`` markers point at search results (``turn0search3``), ``entity`` markers
wrap a JSON ``["type", "name", ...]``, and so on. ``metadata.content_references``
describes most of them positionally (``start_idx``/``end_idx`` + ``matched_text``).
Older messages use ``metadata.citations`` (``start_ix``/``end_ix``) with
``【11†source】`` placeholders.

The output is a :class:`~convoviz.model.TextBlock` whose text contains encoded
references (``CITE_OPEN`` "0,2" ``CITE_CLOSE``) into its ``sources`` tuple. The
renderer turns those into footnotes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import orjson

from convoviz.markdown import split_fenced
from convoviz.model import CITE_CLOSE, CITE_OPEN, Source, TextBlock

if TYPE_CHECKING:
    from convoviz.diagnostics import Diagnostics

_OPEN, _SEP, _CLOSE = "\ue200", "\ue202", "\ue201"
_MARKER_RE = re.compile(r"\ue200([A-Za-z_]+)((?:\ue202[^\ue200\ue201]*)*)\ue201")
_PUA_RE = re.compile(r"[\ue200-\ue2ff]")
_BARE_CITE_RE = re.compile(r"cite(?:turn\d+[a-z_]+\d+(?:L\d+-L\d+)?)+")
_LEGACY_PLACEHOLDER_RE = re.compile(r"【\d+(?:†[^】]{0,80})?】")
_TOKEN_RE = re.compile(r"turn\d+[a-z_]+\d+")

# content_references types whose span renders as nothing.
_DROP_TYPES = frozenset(
    {
        "hidden",
        "sources_footnote",
        "nav_list",
        "navigation_list",
        "tldr",
        "attribution",
    }
)
_LINK_TYPES = frozenset(
    {
        "grouped_webpages",
        "webpage",
        "webpage_extended",
        "grouped_webpages_model_predicted_fallback",
    }
)


class _Sources:
    def __init__(self) -> None:
        self.items: list[Source] = []
        self._index: dict[tuple[str | None, str | None], int] = {}

    def add(self, source: Source) -> int:
        key = (source.url, None if source.url else source.title)
        if key not in self._index:
            self._index[key] = len(self.items)
            self.items.append(source)
        return self._index[key]

    def known(self, source: Source) -> bool:
        return (source.url, None if source.url else source.title) in self._index

    def marker(self, sources: list[Source]) -> str:
        ids = sorted({self.add(s) for s in sources if s.url or s.title})
        if not ids:
            return ""
        return CITE_OPEN + ",".join(map(str, ids)) + CITE_CLOSE


def render_text(text: str, meta: dict[str, Any], diagnostics: Diagnostics) -> TextBlock:
    """Replace citation markers in assistant ``text`` using message ``meta``."""
    if not text:
        return TextBlock(text)
    sources = _Sources()
    text = text.replace("\ue203", "").replace("\ue204", "")

    refs = meta.get("content_references")
    if isinstance(refs, list) and refs:
        text = _apply_spans(text, _content_reference_spans(text, refs, sources, diagnostics))
        footnote = _footnote_sources(refs)
    else:
        footnote = []

    legacy = meta.get("citations")
    if isinstance(legacy, list) and legacy:
        text = _apply_spans(text, _legacy_spans(text, legacy, sources))

    if _OPEN in text:
        lookup = _search_result_lookup(meta)
        products = _product_lookup(refs if isinstance(refs, list) else [])
        text = _MARKER_RE.sub(lambda m: _render_marker(m, lookup, products, sources), text)

    text = _LEGACY_PLACEHOLDER_RE.sub("", text)
    text = _BARE_CITE_RE.sub("", text)
    text = _PUA_RE.sub("", text)
    text = _tidy_blank_lines(_collapse_repeats(text))
    if uncited := [f for f in footnote if not sources.known(f)]:
        text = text.rstrip() + "\n\nSources: " + sources.marker(uncited)
    return TextBlock(text, tuple(sources.items))


# --------------------------------------------------------------------------- spans


def _apply_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Apply non-overlapping (start, end, replacement) spans right to left."""
    last_start = len(text) + 1
    for start, end, replacement in sorted(spans, key=lambda s: s[0], reverse=True):
        if end > last_start:
            continue  # overlaps an already-applied span
        text = text[:start] + replacement + text[end:]
        last_start = start
    return text


def _locate(
    text: str, start: Any, end: Any, matched: str | None, taken: set[int]
) -> tuple[int, int] | None:
    """Find a reference's span, tolerating code-point vs UTF-16 index units."""
    if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end:
        if (matched is None or text[start:end] == matched) and end <= len(text):
            return start, end
        cp_start, cp_end = (
            _utf16_to_codepoints(text, start),
            _utf16_to_codepoints(text, end),
        )
        if (
            cp_start is not None
            and cp_end is not None
            and (matched is None or text[cp_start:cp_end] == matched)
        ):
            return cp_start, cp_end
    if matched:
        position = text.find(matched)
        while position >= 0 and position in taken:
            position = text.find(matched, position + 1)
        if position >= 0:
            return position, position + len(matched)
    return None


def _utf16_to_codepoints(text: str, index: int) -> int | None:
    units = 0
    for cp_index, char in enumerate(text):
        if units >= index:
            return cp_index if units == index else None
        units += 2 if ord(char) > 0xFFFF else 1
    return len(text) if units == index else None


def _content_reference_spans(
    text: str, refs: list[Any], sources: _Sources, diagnostics: Diagnostics
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    taken: set[int] = set()
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        matched = ref.get("matched_text") if isinstance(ref.get("matched_text"), str) else None
        if not matched and not isinstance(ref.get("start_idx"), int):
            continue
        location = _locate(text, ref.get("start_idx"), ref.get("end_idx"), matched, taken)
        if location is None:
            continue
        replacement = _content_reference_replacement(
            ref, text[location[0] : location[1]], sources, diagnostics
        )
        if replacement is None:
            continue  # leave the span for the marker pass
        taken.add(location[0])
        spans.append((location[0], location[1], replacement))
    return spans


def _content_reference_replacement(
    ref: dict[str, Any], span: str, sources: _Sources, diagnostics: Diagnostics
) -> str | None:
    raw_kind = ref.get("type")
    kind = raw_kind if isinstance(raw_kind, str) else ""
    if kind in _DROP_TYPES:
        return ""
    if kind in ("image_group", "image_v2"):
        return _image_links(ref)
    if kind == "client_defined_widget":
        data = ref["data"] if isinstance(ref.get("data"), dict) else {}
        if data.get("widget_type") == "charts_widget_v2":
            return _chart_table(data.get("content"))
        # Code-block widgets: the text already holds their Markdown.
        return None if data.get("widget_type") == "code_block" else (_alt(ref) or "")
    if kind in _LINK_TYPES:
        return sources.marker(_sources_from_items(ref))
    if kind == "file":
        name = ref.get("name")
        return sources.marker([Source(None, name)]) if isinstance(name, str) else ""
    if kind == "entity" or (kind.endswith("entity") and _OPEN in span):
        return _entity_name(span) or _alt(ref) or ""
    if kind in ("products", "product_entity", "alt_text"):
        if alt := _alt(ref):
            return alt
        # A product carousel marker: rendered from its own payload later.
        return None if f"{_OPEN}products{_SEP}" in span else ""
    if kind:
        diagnostics.unknown_kind(f"content_reference:{kind}", "")
    alt = _alt(ref)
    if alt is not None:
        return alt
    if _sources_from_items(ref):
        return sources.marker(_sources_from_items(ref))
    return None


def _sources_from_items(ref: dict[str, Any]) -> list[Source]:
    found: list[Source] = []
    for key in ("items", "sources", "fallback_items"):
        items = ref.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            found.append(_source(item))
            supporting = item.get("supporting_websites")
            if isinstance(supporting, list):
                found.extend(_source(s) for s in supporting if isinstance(s, dict))
        if found:
            break
    if not found and isinstance(ref.get("url"), str):
        found.append(_source(ref))
    found = [s for s in found if s.url or s.title]
    if not found and isinstance(ref.get("safe_urls"), list):
        # Product references keep their links here only.
        found = [Source(url, None) for u in ref["safe_urls"] if (url := _clean_url(_text(u)))]
    return found


def _text(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _footnote_sources(refs: list[Any]) -> list[Source]:
    """The "Sources" list ChatGPT shows under a reply."""
    found: list[Source] = []
    for ref in refs:
        if isinstance(ref, dict) and ref.get("type") == "sources_footnote":
            found.extend(_sources_from_items(ref))
    return found


def _image_links(ref: dict[str, Any]) -> str:
    """Image search results become links to the pages they come from."""
    images = ref.get("images")
    links: list[str] = []
    for entry in images if isinstance(images, list) else ():
        if not isinstance(entry, dict):
            continue
        image = entry.get("image_result") if isinstance(entry.get("image_result"), dict) else entry
        url = _clean_url(_text(image.get("url")) or _text(image.get("content_url")))
        if url:
            title = _text(image.get("title")) or "Image"
            links.append(f"- [{_link_text(title)}](<{url}>)")
    return "\n\n" + "\n".join(links) + "\n\n" if links else ""


def _chart_table(chart: Any) -> str:
    """A chart widget becomes its description and a table of its data."""
    if not isinstance(chart, dict):
        return ""
    x_key, series, data = chart.get("xKey"), chart.get("series"), chart.get("data")
    if not (isinstance(x_key, str) and isinstance(series, list) and isinstance(data, list)):
        return ""
    columns = [s for s in series if isinstance(s, dict) and isinstance(s.get("dataKey"), str)]
    rows = [d for d in data if isinstance(d, dict)]
    if not columns or not rows:
        return ""
    formatted = chart.get("formattedData")
    formatted = formatted if isinstance(formatted, list) else []

    def cell(value: Any) -> str:
        return "" if value is None else " ".join(str(value).split()).replace("|", "\\|")

    def value(index: int, column: dict[str, Any]) -> Any:
        shown = formatted[index] if index < len(formatted) else None
        if isinstance(shown, dict) and shown.get(column["dataKey"]) is not None:
            return shown[column["dataKey"]]
        raw = rows[index].get(column["dataKey"])
        suffix = column.get("valueSuffix") if isinstance(column.get("valueSuffix"), str) else ""
        return None if raw is None else f"{raw}{suffix}"

    header = [x_key] + [_text(c.get("label")) or c["dataKey"] for c in columns]
    lines = [
        "| " + " | ".join(cell(h) for h in header) + " |",
        "|" + " --- |" * len(header),
    ]
    lines += [
        "| " + " | ".join([cell(row.get(x_key))] + [cell(value(i, c)) for c in columns]) + " |"
        for i, row in enumerate(rows)
    ]
    meta = chart.get("meta") if isinstance(chart.get("meta"), dict) else {}
    description = _text(meta.get("description"))
    table = "\n".join(lines)
    return f"\n\n{description}\n\n{table}\n\n" if description else f"\n\n{table}\n\n"


def _link_text(text: str) -> str:
    return " ".join(text.split()).replace("[", "\\[").replace("]", "\\]")


def _source(item: dict[str, Any]) -> Source:
    def text(key: str) -> str | None:
        value = item.get(key)
        return value.strip() or None if isinstance(value, str) else None

    return Source(
        url=_clean_url(text("url")),
        title=text("title"),
        attribution=text("attribution"),
    )


def _clean_url(url: str | None) -> str | None:
    if not url or not url.startswith(("http://", "https://")):
        return None
    # Drop the tracking parameter ChatGPT appends to every cited link.
    url = re.sub(
        r"([?&])utm_source=chatgpt\.com(&|$)",
        lambda m: m.group(1) if m.group(2) else "",
        url,
    )
    return url.rstrip("?&")


def _alt(ref: dict[str, Any]) -> str | None:
    alt = ref.get("alt")
    if not isinstance(alt, str):
        return None
    alt = alt.strip()
    # "[Name]()" (links without targets) become plain names.
    return re.sub(r"\[([^\]]+)\]\(\)", r"\1", alt)


def _legacy_spans(text: str, citations: list[Any], sources: _Sources) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for cit in citations:
        if not isinstance(cit, dict):
            continue
        start, end = cit.get("start_ix"), cit.get("end_ix")
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
            continue
        meta = cit.get("metadata") if isinstance(cit.get("metadata"), dict) else {}
        spans.append((start, end, sources.marker([_source(meta)])))
    return spans


# --------------------------------------------------------------------------- markers


def _search_result_lookup(meta: dict[str, Any]) -> dict[str, Source]:
    """Map ``turn0search3``-style tokens to sources via ``search_result_groups``."""
    lookup: dict[str, Source] = {}
    groups = meta.get("search_result_groups")
    if not isinstance(groups, list):
        return lookup
    for group in groups:
        entries = group.get("entries") if isinstance(group, dict) else None
        if not isinstance(entries, list):
            continue
        for entry in entries:
            ref_id = entry.get("ref_id") if isinstance(entry, dict) else None
            if not isinstance(ref_id, dict):
                continue
            turn, rtype, index = (
                ref_id.get("turn_index"),
                ref_id.get("ref_type"),
                ref_id.get("ref_index"),
            )
            if isinstance(turn, int) and isinstance(rtype, str) and isinstance(index, int):
                lookup[f"turn{turn}{rtype}{index}"] = _source(entry)
    return lookup


def _render_marker(
    match: re.Match[str],
    lookup: dict[str, Source],
    products: dict[str, dict[str, Any]],
    sources: _Sources,
) -> str:
    kind = match.group(1)
    if kind == "products":
        return _product_list(match.group(2).lstrip(_SEP), products)
    payload = [p for p in match.group(2).split(_SEP) if p]
    if kind == "cite":
        tokens = [t for p in payload for t in _TOKEN_RE.findall(p)]
        return sources.marker([lookup[t] for t in tokens if t in lookup])
    if kind == "entity":
        return _entity_name(match.group(0)) or ""
    if kind in ("url", "link") and payload:
        label = payload[0].strip()
        target = payload[-1].strip() if len(payload) > 1 else ""
        if target.startswith(("http://", "https://")):
            return f"[{label}]({target})"
        if target in lookup and lookup[target].url:
            return f"[{label}]({lookup[target].url})"
        return label
    return ""  # navlist, image_group, genui, map...


_PRODUCT_ID_RE = re.compile(r"product\d+$")


def _product_key(cite: str) -> str:
    # The marker says "turn461031product2", the reference "turn1product2".
    match = _PRODUCT_ID_RE.search(cite)
    return match.group(0) if match else cite


def _product_lookup(refs: list[Any]) -> dict[str, dict[str, Any]]:
    """Product details (``title``, ``price``, ``merchants``) by product id."""
    lookup: dict[str, dict[str, Any]] = {}
    for ref in refs:
        products = ref.get("products") if isinstance(ref, dict) else None
        for product in products if isinstance(products, list) else ():
            if isinstance(product, dict) and isinstance(product.get("cite"), str):
                lookup.setdefault(_product_key(product["cite"]), product)
    return lookup


def _product_list(payload: str, products: dict[str, dict[str, Any]]) -> str:
    """A shopping carousel becomes a list of its products' names, prices and shops.

    The payload is ``{"selections": [["turn0product1", "Name"], ...]}``. Images and
    purchase links are not in the export.
    """
    try:
        data = orjson.loads(payload)
    except orjson.JSONDecodeError:
        return ""
    selections = data.get("selections") if isinstance(data, dict) else None
    lines: list[str] = []
    for selection in selections if isinstance(selections, list) else ():
        if not (isinstance(selection, list) and selection and isinstance(selection[0], str)):
            continue
        product = products.get(_product_key(selection[0]), {})
        fallback = selection[1] if len(selection) > 1 and isinstance(selection[1], str) else None
        name = _text(product.get("title")) or _text(fallback)
        if not name:
            continue
        details = [_text(product.get("price")), _text(product.get("featured_tag"))]
        merchants = _text(product.get("merchants"))
        if merchants:
            details.append(merchants.split(" - ")[0])
        extra = " · ".join(d for d in details if d)
        lines.append(f"- **{_link_text(name)}**" + (f" — {extra}" if extra else ""))
    return "\n\n" + "\n".join(lines) + "\n\n" if lines else ""


def _entity_name(span: str) -> str | None:
    match = _MARKER_RE.search(span)
    if not match or match.group(1) != "entity":
        return None
    payload = match.group(2).lstrip(_SEP)
    try:
        data = orjson.loads(payload)
    except orjson.JSONDecodeError:
        return payload.split(_SEP)[0] or None
    if isinstance(data, list) and len(data) > 1 and isinstance(data[1], str):
        return data[1]
    if isinstance(data, list) and data and isinstance(data[0], str):
        return data[0]
    return None


_REPEAT_RE = re.compile(
    re.escape(CITE_OPEN)
    + r"([\d,]+)"
    + re.escape(CITE_CLOSE)
    + r"(?:\s*"
    + re.escape(CITE_OPEN)
    + r"\1"
    + re.escape(CITE_CLOSE)
    + r")+"
)


_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _tidy_blank_lines(text: str) -> str:
    """Drop the extra blank lines that block replacements (lists, tables) leave in prose."""
    if "\n\n\n" in text:
        text = "".join(
            chunk if code else _BLANK_RUN_RE.sub("\n\n", chunk)
            for code, chunk in split_fenced(text)
        )
    return text.strip("\n")


def _collapse_repeats(text: str) -> str:
    return _REPEAT_RE.sub(lambda m: CITE_OPEN + m.group(1) + CITE_CLOSE, text)
