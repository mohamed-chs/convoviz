# Handoff Document for Convoviz

This document provides context for continuing work on the convoviz project.

## System Architecture

### Module Structure

```
convoviz/
├── __init__.py          # Public API exports
├── __main__.py          # python -m convoviz entry
├── cli.py               # Typer CLI
├── config.py            # Pydantic configuration models
├── exceptions.py        # Custom exception hierarchy
├── interactive.py       # Questionary prompts
├── message_logic.py     # Message extraction/visibility utilities
├── pipeline.py          # Main processing pipeline
├── utils.py             # Utility functions
├── models/              # Pure data models (Pydantic)
│   ├── message.py       # Message, MessageAuthor, Content
│   ├── node.py          # Node (tree structure)
│   ├── conversation.py  # Conversation logic
│   └── collection.py    # ConversationCollection
├── renderers/           # Markdown & YAML templates
│   ├── markdown.py      # Markdown generation
│   └── yaml.py          # YAML frontmatter
├── io/                  # File I/O
│   ├── __init__.py      # I/O package entry
│   ├── assets.py        # Attachment handling
│   ├── loaders.py       # ZIP/JSON loaders
│   ├── writers.py       # Markdown/JSON writers
│   └── canvas.py        # Canvas document extraction
├── analysis/            # Visualizations
    ├── graphs/          # Graph generation (modular)
    │   ├── common.py    # Shared utilities
    │   ├── timeseries.py # Activity charts
    │   ├── distributions.py # Histograms
    │   ├── heatmaps.py  # Heatmap charts
    │   └── dashboard.py # Summary dashboard
    └── wordcloud.py     # Word clouds
└── utils.py             # Shared helpers
```

### Important Patterns

#### Configuration Flow
```
get_default_config() → ConvovizConfig
    └── Loaded from bundled TOML defaults
    └── User config (OS-native path) merges over defaults
    └── Runtime defaults applied via apply_runtime_defaults()
        (auto input fallback + default wordcloud font)
    └── CLI flags override merged config
    └── Can be modified directly or via interactive prompts
    └── Passed to run_pipeline(config)
```

#### Data Flow
```
Input (ZIP/Dir/JSON) → loaders.py → ConversationCollection
    └── Accepts single conversations.json or split conversations-NNN.json files
    └── Split files are loaded and merged via ConversationCollection.update()
    └── Contains list of Conversation objects
    └── Sets source_path for asset resolution
```

#### Rendering Flow
```
Conversation + Config → render_conversation() → Markdown string
    └── Uses render_yaml_header() for frontmatter
    └── Uses render_node() for each message
    └── Citation placeholders become per-message footnote refs + footnote block
    └── Uses asset_resolver callback to copy/link images
```

#### Robustness Guardrails
```
Message extraction:
    └── image-only messages return empty text (not exceptions)
    └── Canvas payloads require string name/content; malformed type falls back safely

Conversation traversal:
    └── active-branch traversal is cycle-safe (guards malformed parent graphs)

Archive extraction:
    └── ZIP extraction rejects traversal paths and symlink members
    └── files are extracted through explicit checked paths (no blind extractall)
```

## Key Files to Know

| File | Purpose |
|------|---------|
| `convoviz/config.py` | Configuration models + default loaders (`get_default_graph_config`, `apply_runtime_defaults`) |
| `convoviz/assets/default_config.toml` | Bundled TOML defaults (source of truth for defaults) |
| `convoviz/pipeline.py` | Main processing flow - start here to understand the app |
| `convoviz/io/assets.py`| Logic for finding and copying image assets |
| `convoviz/message_logic.py` | Message extraction/visibility logic (kept out of models) |
| `AGENTS.md` | Context and operational guidelines for AI agents |

## Running the Project

```bash
# Install dependencies
uv sync

# Run CLI
uv run convoviz --help
uv run convoviz --input export.zip --output ./output
uv run convoviz --input ./extracted_export_dir --output ./output

# Full quality check (Lint + Type check + Format + Tests)
uv run ruff check convoviz tests && uv run ty check convoviz && uv run ruff format && uv run pytest
```

## Meta-Notes
- This is a working document for brainstorming and info-sharing; it is not a directive.
- It's the entry point for working/continuing work on the project.
- Try to keep it updated; complete rewrites are within the realm of possibility.
- **A GOOD HABIT TO HAVE (I'm talking to you, AI coding agent) is to write other such .md files to persist plans/context in this session and the next.** Write them in the root project directory, or in other places if you prefer (like ./dev/), depending on the context. Take the liberty to do it anytime, it's very much encouraged and recommended; you don't even have to ask before you do it, or notify the user at all for that matter.
- **KEEP `HANDOFF.md` FOCUSED ON CURRENT STATE.** Use `CHANGELOG.md` for recording recent updates. This file should always reflect the *now*.
- **Changelog Hygiene**: Do not read `docs/dev/CHANGELOG.md` during active work; open it only at the end to append a new entry if needed.

Codebase updates are recorded in [**`docs/dev/CHANGELOG.md`**](CHANGELOG.md).
