"""A self-contained HTML report with charts of how ChatGPT was used.

Pure Python: charts are hand-built inline SVG (no plotting library, no network).
Colors follow a validated, colorblind-safe palette with separate light and dark
steps; every chart also has a data table.
"""

from __future__ import annotations

import calendar
import html
from datetime import date, timedelta
from typing import TYPE_CHECKING

from convoviz.stats import (
    FEATURES,
    LENGTH_BUCKETS,
    Stats,
    bucket_label,
    distinctive_words,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_SEQ_LEVELS = 4
_MAX_MODELS = 7  # + "Other" = 8 categorical slots
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _n(value: float) -> str:
    return f"{value:,.0f}"


def _f(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------- scales


def _levels(values: Sequence[int]) -> list[int]:
    """Upper bounds of the sequential color steps, from quantiles of non-zero values."""
    nonzero = sorted(v for v in values if v > 0)
    if not nonzero:
        return [1] * _SEQ_LEVELS
    last = len(nonzero) - 1
    bounds = [nonzero[min(last, len(nonzero) * i // _SEQ_LEVELS)] for i in range(1, _SEQ_LEVELS)]
    return [*bounds, nonzero[-1]]


def _level(value: int, bounds: list[int]) -> int:
    if value <= 0:
        return 0
    for i, bound in enumerate(bounds, 1):
        if value <= bound:
            return i
    return len(bounds)


def _nice_max(value: float) -> tuple[float, float]:
    """A rounded axis maximum and tick step."""
    if value <= 0:
        return 1, 1
    magnitude = 10 ** (len(str(int(value))) - 1)
    for step in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10):
        tick = step * magnitude
        if tick > 0 and value / tick <= 5:
            top = tick * -(-value // tick)
            return top, tick
    return value, value


def _column_path(x: float, y: float, w: float, h: float, r: float = 4) -> str:
    """A column with a rounded data end (top) and a square baseline."""
    r = min(r, w / 2, h)
    return (
        f"M{x:.1f},{y + h:.1f}V{y + r:.1f}Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f}"
        f"H{x + w - r:.1f}Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h:.1f}Z"
    )


def _bar_path(x: float, y: float, w: float, h: float, r: float = 4) -> str:
    """A horizontal bar with a rounded data end (right) and a square baseline."""
    r = min(r, h / 2, w)
    return (
        f"M{x:.1f},{y:.1f}H{x + w - r:.1f}Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}"
        f"V{y + h - r:.1f}Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f}H{x:.1f}Z"
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in row) + "</tr>" for row in rows)
    return (
        '<details class="data"><summary>Show data</summary>'
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div></details>"
    )


def _section(title: str, subtitle: str, chart: str, table: str = "") -> str:
    return (
        f'<section class="card"><h2>{_e(title)}</h2><p class="sub">{_e(subtitle)}</p>'
        f'<div class="chart">{chart}</div>{table}</section>'
    )


def _legend(items: Sequence[tuple[str, str]]) -> str:
    return (
        '<ul class="legend">'
        + "".join(f'<li><span class="sw {cls}"></span>{_e(label)}</li>' for label, cls in items)
        + "</ul>"
    )


# --------------------------------------------------------------------------- charts


def _tiles(stats: Stats) -> str:
    span = f"{stats.first:%b %Y} – {stats.last:%b %Y}" if stats.first and stats.last else "—"
    tiles = [
        ("Conversations", _n(stats.conversations)),
        ("Prompts sent", _n(stats.prompts)),
        ("Active days", _n(stats.active_days)),
        ("Longest streak", f"{_n(stats.longest_streak)} days"),
        ("Period", span),
    ]
    return (
        '<div class="tiles">'
        + "".join(
            f'<div class="tile"><div class="label">{_e(k)}</div><div class="value">{_e(v)}</div></div>'
            for k, v in tiles
        )
        + "</div>"
    )


def _calendar(stats: Stats) -> str:
    """One GitHub-style activity calendar per year, newest first."""
    if not stats.per_day or stats.first is None or stats.last is None:
        return ""
    bounds = _levels(list(stats.per_day.values()))
    cell, gap, left, top = 11, 2, 30, 16
    blocks = []
    for year in range(stats.last.year, stats.first.year - 1, -1):
        start = date(year, 1, 1)
        offset = start.weekday()
        days = 366 if calendar.isleap(year) else 365
        weeks = (offset + days + 6) // 7
        width = left + weeks * (cell + gap)
        height = top + 7 * (cell + gap)
        parts = [
            (
                f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
                f'role="img" aria-label="Prompts per day in {year}">'
            )
        ]
        for row, name in ((0, "Mon"), (2, "Wed"), (4, "Fri")):
            y = top + row * (cell + gap) + cell - 2
            parts.append(f'<text class="axis" x="0" y="{y}">{name}</text>')
        for month in range(1, 13):
            first = date(year, month, 1)
            week = (offset + (first - start).days) // 7
            x = left + week * (cell + gap)
            parts.append(f'<text class="axis" x="{x}" y="10">{first:%b}</text>')
        for index in range(days):
            day = start + timedelta(days=index)
            week, weekday = divmod(offset + index, 7)
            count = stats.per_day.get(day, 0)
            level = _level(count, bounds)
            noun = "prompt" if count == 1 else "prompts"
            tip = f"{day:%a %d %b %Y}: {count} {noun}"
            parts.append(
                f'<rect class="seq{level}" x="{left + week * (cell + gap)}" '
                f'y="{top + weekday * (cell + gap)}" width="{cell}" height="{cell}" rx="2" '
                f'data-tip="{_e(tip)}" aria-label="{_e(tip)}"/>'
            )
        parts.append("</svg>")
        total = sum(v for d, v in stats.per_day.items() if d.year == year)
        blocks.append(
            f'<div class="year"><h3>{year} <span class="muted">· {_n(total)} prompts</span></h3>'
            f'<div class="scroll">{"".join(parts)}</div></div>'
        )
    scale = _legend(
        [("None", "seq0")]
        + [
            (f"≤ {_n(b)}" if i < _SEQ_LEVELS else f"≤ {_n(b)} (max)", f"seq{i}")
            for i, b in enumerate(bounds, 1)
        ]
    )
    table = _table(["Month", "Prompts"], [(m, stats.per_month.get(m, 0)) for m in stats.months()])
    return _section(
        "Activity",
        "Prompts sent per day. Darker means busier.",
        "".join(blocks) + scale,
        table,
    )


def _week_hour(stats: Stats) -> str:
    values = [v for row in stats.week_hour for v in row]
    if not any(values):
        return ""
    bounds = _levels(values)
    cell, gap, left, top = 22, 2, 34, 18
    width = left + 24 * (cell + gap)
    height = top + 7 * (cell + gap)
    parts = [
        (
            f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
            'aria-label="Prompts by weekday and hour">'
        )
    ]
    for hour in range(0, 24, 3):
        x = left + hour * (cell + gap)
        parts.append(f'<text class="axis" x="{x}" y="12">{hour:02d}h</text>')
    for row, name in enumerate(_WEEKDAYS):
        y = top + row * (cell + gap)
        parts.append(f'<text class="axis" x="0" y="{y + cell - 7}">{name}</text>')
        for hour in range(24):
            count = stats.week_hour[row][hour]
            tip = f"{name} {hour:02d}:00–{hour:02d}:59: {count} prompts"
            parts.append(
                f'<rect class="seq{_level(count, bounds)}" x="{left + hour * (cell + gap)}" '
                f'y="{y}" width="{cell}" height="{cell}" rx="3" data-tip="{_e(tip)}" '
                f'aria-label="{_e(tip)}"/>'
            )
    parts.append("</svg>")
    table = _table(
        ["Day", *[f"{h:02d}" for h in range(24)]],
        [(name, *stats.week_hour[i]) for i, name in enumerate(_WEEKDAYS)],
    )
    return _section(
        "When you ask",
        "Prompts by day of week and hour, in your local time.",
        f'<div class="scroll">{"".join(parts)}</div>',
        table,
    )


def _columns(
    months: list[str],
    series: list[tuple[str, str, list[int]]],
    label: str,
) -> str:
    """Stacked (or single) columns over months. ``series`` = (name, css class, values)."""
    totals = [sum(s[2][i] for s in series) for i in range(len(months))]
    top_value, tick = _nice_max(max(totals, default=0))
    slot = max(6.0, min(36.0, 720 / max(1, len(months))))
    bar = min(24.0, slot * 0.7)
    left, chart_h, bottom = 44, 200, 22
    width = int(left + slot * len(months) + 8)
    height = chart_h + bottom + 8
    parts = [
        (
            f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
            f'aria-label="{_e(label)}">'
        )
    ]
    steps = round(top_value / tick) if tick else 0
    for i in range(steps + 1):
        value = tick * i
        y = 8 + chart_h - chart_h * value / top_value
        cls = "base" if i == 0 else "grid"
        parts.append(f'<line class="{cls}" x1="{left}" x2="{width}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="axis num" x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{_n(value)}</text>'
        )
    label_every = max(1, round(len(months) / 12))
    for index, month in enumerate(months):
        x = left + slot * index + (slot - bar) / 2
        y_cursor = 8 + chart_h
        nonzero = [(n, c, v[index]) for n, c, v in series if v[index] > 0]
        for position, (name, cls, value) in enumerate(nonzero):
            h = chart_h * value / top_value
            is_top = position == len(nonzero) - 1
            gap = 0 if position == 0 else 2
            seg_h = max(0.5, h - gap)
            y = y_cursor - h
            tip = f"{_month_label(month)} · {name}: {_n(value)}"
            shape = (
                f'<path class="{cls}" d="{_column_path(x, y, bar, seg_h)}"'
                if is_top
                else f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{bar:.1f}" height="{seg_h:.1f}"'
            )
            parts.append(f'{shape} data-tip="{_e(tip)}" aria-label="{_e(tip)}"/>')
            y_cursor = y
        if index % label_every == 0:
            short = _month_label(month, short=True)
            parts.append(
                f'<text class="axis" x="{x + bar / 2:.1f}" y="{height - 6}" '
                f'text-anchor="middle">{_e(short)}</text>'
            )
    parts.append("</svg>")
    return f'<div class="scroll">{"".join(parts)}</div>'


def _month_label(month: str, *, short: bool = False) -> str:
    year, number = month.split("-")
    name = calendar.month_abbr[int(number)]
    if short:
        return f"{name} ’{year[2:]}" if number == "01" else name
    return f"{name} {year}"


def _monthly(stats: Stats) -> str:
    months = stats.months()
    if not months:
        return ""
    values = [stats.per_month.get(m, 0) for m in months]
    chart = _columns(months, [("Prompts", "cat1", values)], "Prompts per month")
    table = _table(
        ["Month", "Prompts"],
        [(_month_label(m), v) for m, v in zip(months, values, strict=True)],
    )
    peak_index = max(range(len(values)), key=values.__getitem__)
    subtitle = (
        f"Busiest month: {_month_label(months[peak_index])} ({_n(values[peak_index])} prompts)."
    )
    return _section("Prompts per month", subtitle, chart, table)


def _models(stats: Stats) -> str:
    months = stats.months()
    totals: dict[str, int] = {}
    for counts in stats.models_per_month.values():
        for model, count in counts.items():
            totals[model] = totals.get(model, 0) + count
    if not totals or not months:
        return ""
    ranked = sorted(totals, key=lambda m: (-totals[m], m))
    # Eight slots: all models if they fit, else the top seven plus "Other".
    top = ranked if len(ranked) <= _MAX_MODELS + 1 else ranked[:_MAX_MODELS]
    rest = [m for m in ranked if m not in top]
    series: list[tuple[str, str, list[int]]] = []
    for slot, model in enumerate(top, 1):
        series.append(
            (
                model,
                f"cat{slot}",
                [stats.models_per_month.get(m, {}).get(model, 0) for m in months],
            )
        )
    if rest:
        series.append(
            (
                f"Other ({len(rest)})",
                "other",
                [sum(stats.models_per_month.get(m, {}).get(r, 0) for r in rest) for m in months],
            )
        )
    chart = _legend([(name, cls) for name, cls, _ in series]) + _columns(
        months, series, "Replies per month by model"
    )
    table = _table(["Model", "Replies"], [(m, totals[m]) for m in ranked])
    return _section("Models", "Replies per month, by the model that wrote them.", chart, table)


def _hbars(rows: list[tuple[str, int, str]], label: str) -> str:
    """Horizontal bars with the value at the tip. ``rows`` = (label, value, tip suffix)."""
    peak = max((v for _, v, _ in rows), default=0) or 1
    left, bar_h, row_h, right = 190, 16, 28, 90
    width = 640
    plot = width - left - right
    height = row_h * len(rows) + 4
    parts = [
        (
            f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
            f'aria-label="{_e(label)}">'
        )
    ]
    for index, (name, value, extra) in enumerate(rows):
        y = index * row_h + (row_h - bar_h) / 2
        parts.append(
            f'<text class="label" x="{left - 10}" y="{y + bar_h - 3:.1f}" text-anchor="end">{_e(name)}</text>'
        )
        parts.append(
            f'<line class="base" x1="{left}" x2="{left}" y1="{y - 4:.1f}" y2="{y + bar_h + 4:.1f}"/>'
        )
        w = plot * value / peak
        tip = f"{name}: {_n(value)}{extra}"
        if value > 0:
            parts.append(
                f'<path class="cat1" d="{_bar_path(left, y, max(w, 2), bar_h)}" '
                f'data-tip="{_e(tip)}" aria-label="{_e(tip)}"/>'
            )
        parts.append(
            f'<text class="value num" x="{left + w + 8:.1f}" y="{y + bar_h - 3:.1f}">{_n(value)}{_e(extra)}</text>'
        )
    parts.append("</svg>")
    return f'<div class="scroll">{"".join(parts)}</div>'


def _lengths(stats: Stats) -> str:
    if not stats.conversations:
        return ""
    rows = [
        (
            f"{bucket_label(lo, hi)} messages",
            stats.lengths.get(bucket_label(lo, hi), 0),
            "",
        )
        for lo, hi in LENGTH_BUCKETS
    ]
    chart = _hbars(rows, "Conversations by length")
    table = _table(["Length", "Conversations"], [(r[0], r[1]) for r in rows])
    return _section(
        "Conversation length",
        "How many messages (prompts and replies) conversations have.",
        chart,
        table,
    )


def _features(stats: Stats) -> str:
    if not stats.conversations or not stats.features:
        return ""
    rows = []
    for key, label in FEATURES:
        count = stats.features.get(key, 0)
        share = 100 * count / stats.conversations
        rows.append((label, count, f" · {_f(share)}%"))
    rows.sort(key=lambda r: -r[1])
    chart = _hbars(rows, "Conversations using each feature")
    table = _table(
        ["Feature", "Conversations", "Share"],
        [(r[0], r[1], r[2].strip(" ·")) for r in rows],
    )
    return _section("Features", "Conversations in which each feature was used.", chart, table)


def _words(stats: Stats) -> str:
    by_year = distinctive_words(stats)
    by_year = {year: words for year, words in by_year.items() if words}
    if not by_year:
        return ""
    blocks = []
    for year in sorted(by_year, reverse=True):
        words = by_year[year]
        top_score = words[0][1]
        spans = []
        for word, score in sorted(words, key=lambda w: w[0]):
            size = 0.85 + 1.15 * (score / top_score) ** 0.6
            weight = 600 if score / top_score > 0.5 else 400
            spans.append(
                f'<span style="font-size:{size:.2f}rem;font-weight:{weight}">{_e(word)}</span>'
            )
        blocks.append(
            f'<div class="year"><h3>{year}</h3><p class="words">{" ".join(spans)}</p></div>'
        )
    subtitle = (
        "Words in your prompts that set each year apart from the others."
        if len(by_year) > 1
        else "The words you used most in your prompts."
    )
    rows = [
        (year, ", ".join(w for w, _ in by_year[year])) for year in sorted(by_year, reverse=True)
    ]
    return _section(
        "What you asked about",
        subtitle,
        "".join(blocks),
        _table(["Year", "Words"], rows),
    )


# --------------------------------------------------------------------------- page


def render_insights(stats: Stats, title: str = "Your ChatGPT, in charts") -> str:
    """The complete HTML document."""
    sections = [
        _calendar(stats),
        _week_hour(stats),
        _monthly(stats),
        _models(stats),
        _features(stats),
        _lengths(stats),
        _words(stats),
    ]
    body = "".join(s for s in sections if s) or '<p class="sub">No messages found.</p>'
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>"
        f'<main><header><h1>{_e(title)}</h1><p class="sub">Generated by convoviz from your '
        "ChatGPT export. Everything stays on your computer.</p></header>"
        f"{_tiles(stats)}{body}</main>"
        f'<div id="tip" role="tooltip" hidden></div><script>{_JS}</script></body></html>\n'
    )


_CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--grid:#e1e0d9;--base:#c3c2b7;--ring:rgba(11,11,11,.10);
--seq0:#eeede8;--seq1:#b7d3f6;--seq2:#6da7ec;--seq3:#2a78d6;--seq4:#184f95;
--cat1:#2a78d6;--cat2:#eb6834;--cat3:#1baf7a;--cat4:#eda100;--cat5:#e87ba4;
--cat6:#008300;--cat7:#6250d6;--cat8:#e34948;--other:#a8a7a0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--base:#383835;--ring:rgba(255,255,255,.10);--seq0:#242423;--seq1:#104281;--seq2:#1c5cab;
--seq3:#3987e5;--seq4:#86b6ef;--cat1:#3987e5;--cat2:#d95926;--cat3:#199e70;--cat4:#c98500;
--cat5:#d55181;--cat6:#008300;--cat7:#9085e9;--cat8:#e66767;--other:#6b6a65}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--base:#383835;--ring:rgba(255,255,255,.10);
--seq0:#242423;--seq1:#104281;--seq2:#1c5cab;--seq3:#3987e5;--seq4:#86b6ef;--cat1:#3987e5;
--cat2:#d95926;--cat3:#199e70;--cat4:#c98500;--cat5:#d55181;--cat6:#008300;--cat7:#9085e9;
--cat8:#e66767;--other:#6b6a65}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:1.75rem;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:1.1rem;margin:0}
h3{font-size:.95rem;margin:16px 0 6px}
.sub{color:var(--ink2);margin:2px 0 16px}
.muted{color:var(--muted);font-weight:400}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}
.tile,.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px}
.tile{padding:14px 16px}
.tile .label{color:var(--ink2);font-size:.85rem}
.tile .value{font-size:1.5rem;font-weight:600;margin-top:2px}
.card{padding:20px;margin:16px 0}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
svg{display:block;max-width:none;font-size:11px}
svg .axis{fill:var(--muted)}
svg .label{fill:var(--ink2);font-size:12px}
svg .value{fill:var(--ink);font-size:12px}
.num{font-variant-numeric:tabular-nums}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .base{stroke:var(--base);stroke-width:1}
svg [data-tip]{cursor:default}
svg [data-tip]:hover{opacity:.8}
.seq0{fill:var(--seq0)}.seq1{fill:var(--seq1)}.seq2{fill:var(--seq2)}.seq3{fill:var(--seq3)}
.seq4{fill:var(--seq4)}.cat1{fill:var(--cat1)}.cat2{fill:var(--cat2)}.cat3{fill:var(--cat3)}
.cat4{fill:var(--cat4)}.cat5{fill:var(--cat5)}.cat6{fill:var(--cat6)}.cat7{fill:var(--cat7)}
.cat8{fill:var(--cat8)}.other{fill:var(--other)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;list-style:none;padding:0;margin:8px 0 12px;
color:var(--ink2);font-size:.85rem}
.legend li{display:flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
.sw.seq0{background:var(--seq0)}.sw.seq1{background:var(--seq1)}.sw.seq2{background:var(--seq2)}
.sw.seq3{background:var(--seq3)}.sw.seq4{background:var(--seq4)}.sw.cat1{background:var(--cat1)}
.sw.cat2{background:var(--cat2)}.sw.cat3{background:var(--cat3)}.sw.cat4{background:var(--cat4)}
.sw.cat5{background:var(--cat5)}.sw.cat6{background:var(--cat6)}.sw.cat7{background:var(--cat7)}
.sw.cat8{background:var(--cat8)}.sw.other{background:var(--other)}
.words{line-height:1.9;margin:0;color:var(--ink)}
.words span{margin-right:.6em;white-space:nowrap}
details.data{margin-top:12px;color:var(--ink2);font-size:.85rem}
details.data summary{cursor:pointer}
.table-wrap{overflow-x:auto;max-height:360px;margin-top:8px}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:4px 12px 4px 0;border-bottom:1px solid var(--grid);white-space:nowrap}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);
padding:4px 8px;border-radius:6px;font-size:12px;z-index:10;max-width:280px}
"""

_JS = """
(()=>{const t=document.getElementById('tip');
const show=(e,el)=>{t.textContent=el.dataset.tip;t.hidden=false;
const x=Math.min(e.clientX+12,innerWidth-t.offsetWidth-8);
t.style.left=x+'px';t.style.top=(e.clientY-t.offsetHeight-12)+'px'};
document.addEventListener('pointermove',e=>{const el=e.target.closest('[data-tip]');
if(el)show(e,el);else t.hidden=true});
document.addEventListener('pointerleave',()=>{t.hidden=true});})();
"""
