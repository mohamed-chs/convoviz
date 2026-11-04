# Architecture

convoviz is a small pipeline. Each stage knows as little as possible about the
stages around it.

```
input paths ─► source.py ─► chatgpt/ (adapter) ─► model.py ─┬─► render.py ─► writer.py ─► notes, index, .base
               ExportSource   files.py  parse.py   Conversation│    tree.py      assets.py
                              citations.py textfix.py          └─► stats.py ─► insights.py ─► insights.html
```

| Module | Responsibility |
|---|---|
| `source.py` | Read-only, flat view of an export, whatever its packaging: ZIP, folder, single JSON, ZIPs nested inside a ZIP (Privacy Portal), split `-part-NNNN.zip` sets. Member names are only ever dictionary keys, so ZIP path traversal has nothing to exploit. It also finds exports in Downloads. |
| `chatgpt/files.py` | Which members hold conversations (`export_manifest.json`, then numbered shards, then `conversations.json`), tolerant JSON loading (including surrogate repair), and `AssetIndex`, which resolves `sediment://file_…` pointers to `.dat` members, original names and MIME types. |
| `chatgpt/parse.py` | Raw conversation dict → `Conversation`. Every judgement about ChatGPT's message zoo lives here: what is hidden, what a tool call becomes, and how Canvas edits are replayed per branch. |
| `chatgpt/citations.py` | Private-use citation markers and `content_references` → neutral `Source`s, with encoded markers in the text. |
| `chatgpt/textfix.py` | ChatGPT-only prose cleanups (`:::writing{}` directives, `sandbox:` links). |
| `model.py` | The provider-neutral model: `Conversation` → `Node` tree → `Message` → typed blocks. |
| `tree.py` | Rebuilds the tree from `parent` links, picks the active branch, and linearizes it into a `Thread` with `Fork`s for the alternatives. |
| `diagnostics.py` | Collects errors, warnings, skips and unknown kinds for the end-of-run summary and for `inspect`. |
| `render.py` | `Conversation` → one Markdown note (`Note`). Owns the Markdown conventions. |
| `markdown.py` | Code-aware text utilities: transform prose without touching code, close fences, math delimiters, escaping. |
| `fsnames.py` | Safe file names that keep any script (Cyrillic, CJK, emoji…). |
| `assets.py` | Copies attachments under readable names with real extensions (magic-byte sniffing for `.dat`). |
| `writer.py` | Output layout and the incremental `state.json` logic. Writes `index.md` and `Conversations.base`. |
| `stats.py` / `insights.py` | Usage aggregates → a self-contained HTML page of hand-built SVG charts. |
| `census.py` | The anonymous structure report behind `convoviz inspect`. |
| `config.py` / `cli.py` | TOML settings (validated, with clear errors) and the cyclopts CLI. |

## Principles

- **Never fail silently, never fail loudly.** One bad record must not sink an
  export. Parsing is defensive (`_str`, `_dict` and friends; no schema
  validation), and per-conversation and per-message exceptions are caught and
  recorded in `Diagnostics`. The user sees a summary. Maintainers get
  `inspect --json`.
- **Provider-neutral core.** Only `chatgpt/` knows ChatGPT. Renderers, stats
  and the writer see `model.py` types. A Claude or Gemini adapter would be a
  sibling package producing the same model.
- **Interpret once.** The model holds what a reader cares about, already
  interpreted (a `DocumentBlock` carries the full Canvas text after its edits
  are replayed, and citations are already `Source`s). Renderers stay simple.
- **Text is Markdown, code is sacred.** Every prose transformation goes
  through `markdown.map_prose`, so fenced and inline code are never altered.
- **The user's files win.** The writer never overwrites a note (or `index.md`,
  `Conversations.base`, `insights.html`) whose hash differs from what convoviz
  last wrote, never overwrites a file it didn't create, follows notes the user
  moved (by their `conversation_id` property), doesn't recreate notes the user
  deleted, and never deletes notes for conversations missing from a newer
  export. State is saved even when a run is interrupted.
- **Few dependencies.** cyclopts (CLI), rich (terminal output), orjson (fast
  JSON), platformdirs (Downloads/Documents/config locations). The charts are
  plain SVG strings, with no plotting library.

## Markdown conventions

- Properties: JSON-quoted YAML strings, `created`/`updated` as
  `YYYY-MM-DDTHH:MM:SS` local time (Obsidian's Date & time type), and lists for
  `models`, `tags` and `aliases`. Property names and types stay the same in
  every note, because Obsidian fixes a property's type vault-wide.
- `# Title` once, then `## You` / `## ChatGPT` per turn. Several stored
  messages (reasoning, tool calls, answer) are grouped into one assistant
  turn. Headings inside messages are pushed down two levels so they can't break
  the outline.
- Folded parts use `<details>` with a blank line after `<summary>`, so the
  Markdown inside renders (CommonMark HTML-block rule). Canvas documents are
  `<details open>`.
- Citations become footnotes numbered once per conversation, de-duplicated by
  URL.
- `\( \)` / `\[ \]` become `$` / `$$`. `$5` is escaped. Raw HTML in user text is
  neutralized. Dangling code fences are closed.
- Branches: at each fork point, alternatives are rendered (recursively) in a
  folded block before the active branch continues.

## Testing

`tests/factory.py` builds synthetic exports (2026 parent-only trees, legacy
`children`, shards with a manifest, `.dat` assets, nested ZIPs). **No real
export data is ever committed.** `tests/golden/rich.md` pins the rendered
output of a conversation that exercises most features. Regenerate it with
`UPDATE_GOLDEN=1 uv run pytest tests/test_render.py` and review the diff.
Hypothesis covers the text utilities and the file-name sanitizer. A privacy
test asserts that `inspect` output contains no content.
