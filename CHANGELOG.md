# Changelog

User-visible changes. Format: [Keep a Changelog](https://keepachangelog.com),
versions follow [SemVer](https://semver.org).

## Unreleased

## 1.0.0 — 2026-11

A complete rewrite for ChatGPT's 2026 export format.

### Added
- Reads every current export layout: split `conversations-NNN.json` with
  `export_manifest.json`, `.dat` attachments, nested Privacy Portal ZIPs,
  multi-part downloads, extracted folders. Several exports can be passed at once.
- `convoviz` with no arguments finds the newest export in Downloads.
- Incremental updates: re-running only rewrites changed conversations, moves
  renamed ones, and never overwrites notes you edited (`--force` to override).
- Edited and regenerated replies are kept, folded where they branch
  (`--branches active` for the visible branch only).
- Sources as footnotes, numbered once per conversation; Canvas documents with
  their edits applied; code, output and reasoning folded; generated and
  uploaded images copied with real file extensions.
- `index.md`, an Obsidian `Conversations.base`, and `insights.html` with
  activity, hour-of-week, monthly, model, feature, length and topic charts
  (works offline, light and dark).
- `convoviz inspect [--json]`: an anonymous structure report for bug reports.
- `convoviz config init|path` and a TOML config file.

### Fixed
- Conversations rendered only their last message (exports no longer list
  `children`) (#62, #70).
- One malformed or empty entry aborted the whole run (#60, #70).
- Non-Latin titles produced empty or garbled file names (#64).
- Broken emoji (unpaired surrogates) failed to load (#69).
- Footnote numbers restarted, and collided, in every message.

### Removed
- Word cloud images, bundled fonts and the matplotlib/NLTK dependencies
  (replaced by the insights page).
- The browser script that downloaded conversations from chatgpt.com directly.
- Python APIs of 0.x (`ConversationCollection`, `run_pipeline`, …); convoviz is
  a command-line tool.
