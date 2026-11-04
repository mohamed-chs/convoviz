# Handoff

Current state of the project: what exists, what's known to be missing, and
where to go next. Keep this describing the present, not the history (that is
`CHANGELOG.md` and git).

## State (v1.0.0, November 2026)

A ground-up rewrite (branch `overhaul`) for the 2026 export format. The full
pipeline is in place and tested on synthetic exports: reading (all known
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
- **Library-only files.** Library entries whose `origination_thread_id` names a
  conversation, and which no message shows, are added to that note under "Files
  from this conversation" (`Conversation.files`). That the field is the
  conversation id is an assumption: no 2026 source confirms it. If it's wrong,
  nothing is added (check `library_files_added` in `inspect --json`).
- **Deep research / agent mode.** `research_kickoff_tool` and other agent
  plumbing is hidden. Whether the final report always arrives as a normal
  assistant message is unverified.
- **Message `channel`.** 2026 reports mention `channel: "commentary"` vs
  `"final"` on assistant messages. Both are rendered as normal text for now.
  `inspect` counts `role:channel` pairs, so a report will show whether
  commentary should be folded like reasoning.
- **Insights cover the current export only.** Conversations kept in the
  archive from older exports but missing from the new one aren't counted.
- **Canvas.** Every edit renders the whole document again (in an open
  `<details>`). For heavily edited documents, showing only the final version
  per turn might read better.
- **Custom instructions** are captured in `Conversation.extra` but not
  rendered anywhere. They could become an optional note or `index.md` section.
- **Project names.** Only `project_id` is known. If an export ever carries
  project names, group notes by project.
- **Word clouds** use an English stop list. Other languages work, but filler
  words in them may show up. The layout runs in the browser (`_CLOUD_JS` in
  `insights.py`, spiral placement on measured text boxes) and isn't covered by
  the Python tests. Check it in a browser, at phone and desktop widths, after
  changing it.
- **Changing `--timezone`** can change dates in file names, so notes move.
  That's handled (as moves), but it isn't obvious to users.
- **Other providers.** `model.py` is provider-neutral. A Claude export adapter
  (`convoviz/claude/`) would be a self-contained addition. Research notes on
  other providers' formats are not in the repo yet; ask the maintainer.

## Conventions worth knowing

- `Writer` decides "edited by the user" by comparing a note's current hash
  with the hash recorded in `.convoviz/state.json`. A recorded note missing
  from its path is searched for by its `conversation_id` property (moved →
  `pinned`, not found → `deleted` tombstone). Managed files (`index.md`,
  `Conversations.base`, `insights.html`) have hashes under `files`.
- `tests/golden/rich.md` is the reference rendering. Any output change shows
  up there.
- Citation markers inside `TextBlock.text` are encoded as `\x1e0,2\x1f`
  (indices into `TextBlock.sources`). Use `model.strip_citations` for plain
  text.
