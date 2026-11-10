# Handoff

Current state of the project: what exists, what's known to be missing, and
where to go next. Keep this describing the present, not the history (that is
`CHANGELOG.md` and git).

## State (v1.0.1, November 2026)

A ground-up rewrite (branch `overhaul`) for the 2026 export format, with a
second adapter for Claude exports. The full pipeline is in place and tested on synthetic exports: reading (all known
packagings), parsing, Markdown rendering, incremental writing, attachments,
the index/Bases files, insights (charts and word clouds), `inspect`, config
and CLI. The quality gate (ruff, ty, pytest) is green on Python 3.12 and 3.15,
and the fuzz tests pass the `long` Hypothesis profile.

Performance: a synthetic 6,000-conversation, 344 MB export takes about 27 s and
600 MB of memory (parsing, rendering and stats in roughly equal parts). Re-runs
take about as long, since every conversation is rendered to compare hashes.
Skipping unchanged conversations by fingerprinting the raw JSON would make
re-runs much faster if that ever matters.

## Biggest risk

**No real 2026 export has been run through this code.** Everything is built
from public evidence: other projects' PRs and issues, community reports, and
convoviz's own issue history (see `export-format.md`). Expect surprises in the
first reports from users. The design aims to make them visible
(Diagnostics, the summary, `inspect --json`) rather than fatal. Triage
`inspect` reports first, then make the fix and add a synthetic test.

## Known gaps / next steps

- **Unconfirmed format details.** One source reports `library-files.json`
  (hyphen) and an `export_manifest.json` with an `export_files: [{path,
  size_bytes}]` inventory instead of `logical_files`. Both are accepted. Another
  claims mid-2026 exports no longer contain `tool`/`system` messages, which
  would only mean less to hide.
- **Library-only files.** Files no message shows go to the message named by
  `origination_message_id`, or else to the end of the conversation named by
  `origination_thread_id`. Nexus AI Chat Importer, which tests against real
  exports, confirms these fields. Entries without a payload and `writing_block`
  entries are skipped. Unlike Nexus, convoviz doesn't invent a message when the
  originating one is missing. Check `library_files_added` and
  `library_files_without_payload` in `inspect --json`.
- **Reference implementations.** The best-maintained parsers are
  [pionxzh/chatgpt-exporter](https://github.com/pionxzh/chatgpt-exporter) (live
  API, citations, widgets) and
  [Superkikim/nexus-ai-chat-importer](https://github.com/Superkikim/nexus-ai-chat-importer)
  (2026 ZIP layout, library, directives, multipart). When the format drifts,
  diff their recent commits first: they see real exports.
- **Deep research / agent mode.** `research_kickoff_tool` and other agent
  plumbing is hidden. Whether the final report always arrives as a normal
  assistant message is unverified.
- **Message `channel`.** 2026 reports mention `channel: "commentary"` vs
  `"final"` on assistant messages, while chatgpt-exporter says official
  exports drop `channel` (and `recipient`) entirely. It isn't used. `inspect`
  counts `role:channel` pairs if it ever shows up.
- **Inferred preambles.** A text reply followed by a reasoning step is folded
  into the reasoning even without `is_thinking_preamble_message`. If a report
  shows real answers disappearing into the reasoning panel, look at
  `inferred_preambles` first.
- **Insights cover the current export only.** Conversations kept in the
  archive from older exports but missing from the new one aren't counted.
- **Canvas.** Every edit renders the whole document again (in an open
  `<details>`). For heavily edited documents, showing only the final version
  per turn might read better. The final version on the active branch is also
  saved under `canvas/` (`render.document_files`, named by document title
  plus the conversation id so names are stable; Claude's artifacts and files
  go to `artifacts/` the same way). `state.json` records each generated
  file's owner (`conversation:<id>`, `project:<id>`), and files an owner no
  longer produces are removed if unchanged (`Writer._remove_outdated`).
- **Custom instructions** go to `Custom instructions.md` (one section per
  distinct version, newest first), written by the Writer as a managed file.
  Memory snapshots (`model_editable_context`, reportedly `model_set_context`
  text) are hidden and not captured; a "Memories" page could follow the same
  pattern once a real export confirms the shape.
- **Project names.** Only `project_id` is known. If an export ever carries
  project names, group notes by project.
- **Word clouds** drop filler words in English, French, Spanish, Portuguese,
  German, Italian and Arabic (`stopwords.py`, plus the user's `ignored_words`).
  Other languages work, but their filler words may show up. The layout runs in the browser (`_CLOUD_JS` in
  `insights.py`, spiral placement on measured text boxes) and isn't covered by
  the Python tests. Check it in a browser, at phone and desktop widths, after
  changing it.
- **Changing `--timezone`** can change dates in file names, so notes move.
  That's handled (as moves), but it isn't obvious to users.
- **Claude.** `convoviz/claude/` reads claude.ai exports. It was built from
  public evidence only (`claude-export-format.md` lists the sources and how
  sure each fact is); no real Claude export has been through it. Watch for:
  - tool names drifting again (`_ALIASES`, `_tool_use`); unknown tools are
    hidden and counted as `tool_call:<name>`, and `inspect --json` lists
    `claude.tool_names` (connectors only as `<mcp>`);
  - `image`/`document` content blocks, whose shape is unknown;
  - citation offsets (`_word_end` nudges markers to word ends in case they
    count UTF-16 units);
  - `md_citations` on artifacts and `citations_grouping_mode`, not used yet.
  Projects (`projects/<name>.md` plus knowledge docs as text) and memory
  (`Claude memory.md`) are read from their ZIPs (`claude/extras.py`).
  Conversations carry no project link, so project notes can't list their
  conversations. Published artifacts (`frames-000.zip`),
  `design_chats` and Claude Code cloud routines aren't read.

## Conventions worth knowing

- `Writer` decides "edited by the user" by comparing a note's current hash
  with the hash recorded in `.convoviz/state.json`. A recorded note missing
  from its path is searched for by its `conversation_id` property (moved →
  `pinned`, not found → `deleted` tombstone). Managed files (`index.md`,
  `Conversations.base`, `insights.html`) have hashes under `files`.
- `tests/golden/rich.md` is the reference rendering. Any output change shows
  up there.
- `tests/demo.py` builds a seeded, made-up export of about 1,500
  conversations. It is the source of the README's example note
  (`docs/example/`, checked by `tests/test_demo.py`) and its images
  (`docs/assets/`, regenerated by `scripts/readme_images.py`).
- Citation markers inside `TextBlock.text` are encoded as `\x1e0,2\x1f`
  (indices into `TextBlock.sources`). Use `model.strip_citations` for plain
  text.
