# AGENTS.md

> `CLAUDE.md` and `GEMINI.md` are symlinks to this file.

**convoviz** turns ChatGPT data exports (ZIP/folder/JSON) into an Obsidian-friendly
Markdown archive, an index, an Obsidian Bases view, and a self-contained
`insights.html`. Python ≥ 3.12, `src/` layout, managed with `uv`.

## First, orient yourself

Read [`docs/dev/HANDOFF.md`](docs/dev/HANDOFF.md) (current state, open work) and
[`docs/dev/architecture.md`](docs/dev/architecture.md) (how the pipeline fits)
before changing anything. Skim the modules you'll touch and their tests. For
anything about the export's shape, read
[`docs/dev/export-format.md`](docs/dev/export-format.md).

## Commands

Always go through `uv` (`uv add` for dependencies, never hand-edit versions).

| | |
|---|---|
| Test | `uv run pytest` (`-x` to stop at the first failure) |
| Lint | `uv run ruff check src tests` (`--fix` for autofixes) |
| Format | `uv run ruff format` |
| Types | `uv run ty check src tests` |
| Golden output | `UPDATE_GOLDEN=1 uv run pytest tests/test_render.py tests/test_claude.py`, then review the diff |

Run all four checks and fix everything before calling work done.

## How to work here

- **Be critical.** Don't assume existing code is right. The export format
  drifts, and so do assumptions. Verify against tests and the format doc.
- **No hacks.** Fix root causes. Backward compatibility is not a goal unless
  explicitly requested. Prefer a clean rewrite to a patch on a bad design.
- **Privacy is non-negotiable.** Never commit or download real user exports.
  Tests use `tests/factory.py`. `inspect` output must never contain content,
  and `tests/test_census.py` guards that.
- **Tolerate, then report.** Parsing never raises on odd data. It records the
  problem in `Diagnostics` so the summary and `inspect` can surface it.
- **User experience first.** Think about the person opening the notes in
  Obsidian, and about the person whose re-run must not clobber their edits.
- **Push back.** If a request is flawed or contradicts these patterns, say so
  and propose something better. Ask the maintainer when you need information
  only they have.
- **Cohesion pass.** After a change, sweep config, CLI flags, README, docs,
  tests and the golden file so the behavior is wired end to end.
- **Docs.** Add user-visible changes to `CHANGELOG.md` under "Unreleased", and
  only append there. When you leave work unfinished or change the system's
  shape, update `docs/dev/HANDOFF.md`. It describes the current state, not a
  history.
- **Commits.** Small and logically scoped, with tests for behavior changes.
