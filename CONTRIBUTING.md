# Contributing

Thanks for helping! Bug reports with `convoviz inspect --json` output are the
single most useful contribution: the export format changes constantly, and that
report shows what changed without exposing anyone's data.

## Setup

```sh
git clone https://github.com/mohamed-chs/convoviz && cd convoviz
uv sync
uv run convoviz --help
```

## Before you push

```sh
uv run ruff format
uv run ruff check src tests
uv run ty check src tests
uv run pytest
```

CI runs the same checks, with tests on Python 3.12–3.15 (Linux) plus Windows and
macOS.

## Ground rules

- **Never commit real export data**, including "anonymized" excerpts. Build
  what you need with `tests/factory.py` (`ConversationBuilder`, `write_export`).
- A new export quirk means a test reproducing it (from a synthetic export), a
  change in `chatgpt/`, and a line in `docs/dev/export-format.md` with a source.
- Changes to the Markdown output show up in `tests/golden/rich.md`. Regenerate
  it with `UPDATE_GOLDEN=1 uv run pytest tests/test_render.py` and check the
  diff reads well.
- Record user-visible changes under "Unreleased" in `CHANGELOG.md`.
- Keep dependencies few. The charts are plain SVG on purpose.

How the pieces fit is in [docs/dev/architecture.md](docs/dev/architecture.md).

## Releasing

1. Move the "Unreleased" changelog entries under a new version heading.
2. `uv version --bump minor` (or `patch`/`major`), then commit.
3. Tag `vX.Y.Z` and push the tag. The release workflow tests, builds, publishes
   to PyPI through trusted publishing, and creates the GitHub release from the
   changelog section.
