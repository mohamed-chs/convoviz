# Handoff

Current state of the project: what exists, what's known to be missing, and
where to go next. Keep this describing the present, not the history (that is
`CHANGELOG.md` and git).

## State (v1.0.0, November 2026)

A ground-up rewrite (branch `overhaul`) for the 2026 export format. The full
pipeline is in place and tested on synthetic exports: reading (all known
packagings), parsing, Markdown rendering, incremental writing, attachments,
the index/Bases files, insights, `inspect`, config and CLI. The quality gate
(ruff, ty, pytest) is green on Python 3.12 and 3.15.

## Biggest risk

**No real 2026 export has been run through this code.** Everything is built
from public evidence: other projects' PRs and issues, community reports, and
convoviz's own issue history (see `export-format.md`). Expect surprises in the
first reports from users. The design aims to make them visible
(Diagnostics, the summary, `inspect --json`) rather than fatal. Triage
`inspect` reports first, then make the fix and add a synthetic test.

## Known gaps / next steps

- **Library-only files.** `library_files.json` is used only for names and MIME
  types. Generated images and documents that appear *only* in the library
  (linked by `origination_thread_id`, with no pointer in any message) are
  not added to their conversation. Adding them as a "Files from this
  conversation" section is the natural next step once real data confirms the
  linkage.
- **Deep research / agent mode.** `research_kickoff_tool` and other agent
  plumbing is hidden. Whether the final report always arrives as a normal
  assistant message is unverified.
- **Canvas.** Every edit renders the whole document again (in an open
  `<details>`). For heavily edited documents, showing only the final version
  per turn might read better.
- **Custom instructions** are captured in `Conversation.extra` but not
  rendered anywhere. They could become an optional note or `index.md` section.
- **Project names.** Only `project_id` is known. If an export ever carries
  project names, group notes by project.
- **Words chart** uses an English stop list. Other languages work, but filler
  words in them may show up.
- **Changing `--timezone`** can change dates in file names, so notes move.
  That's handled (as moves), but it isn't obvious to users.
- **Other providers.** `model.py` is provider-neutral. A Claude export adapter
  (`convoviz/claude/`) would be a self-contained addition. Research notes on
  other providers' formats are not in the repo yet; ask the maintainer.

## Conventions worth knowing

- `Writer` decides "edited by the user" by comparing a note's current hash
  with the hash recorded in `.convoviz/state.json`.
- `tests/golden/rich.md` is the reference rendering. Any output change shows
  up there.
- Citation markers inside `TextBlock.text` are encoded as `\x1e0,2\x1f`
  (indices into `TextBlock.sources`). Use `model.strip_citations` for plain
  text.
