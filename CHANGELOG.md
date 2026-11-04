# Changelog

User-visible changes. Format: [Keep a Changelog](https://keepachangelog.com),
versions follow [SemVer](https://semver.org).

## Unreleased

### Added

- Interactive replies are written from their Markdown version instead of
  widget code.
- Charts in replies become tables, and image search results become links to
  their pages.
- Shopping carousels list their products, with prices and shops.
- A reply's "Sources" list keeps the sources it didn't cite inline.
- Generated images and documents that are only in the export's file library
  appear in the message that produced them, when the export kept it.

### Changed

- Progress notes written while the model thinks are folded into its
  reasoning.
- Library files with no payload in the export are no longer listed as missing.

### Fixed

- AVIF images and QuickTime videos stored as `.dat` get the right extension.

## 1.0.0 — 2026-11

A complete rewrite for ChatGPT's 2026 export format.

### Added
- Reads every current export layout: split `conversations-NNN.json` with
  `export_manifest.json`, `.dat` attachments, nested Privacy Portal ZIPs,
  multi-part downloads, extracted folders. Several exports can be passed at once.
- `convoviz` with no arguments finds the newest export in Downloads.
- Incremental updates: re-running only rewrites changed conversations and
  renames notes of renamed ones. Notes you edited are kept, notes you moved are
  updated where they are, notes you deleted stay deleted, and files convoviz
  didn't create are never overwritten (`--force` overrides edits and deletions).
- Edited and regenerated replies are kept, folded where they branch
  (`--branches active` for the visible branch only).
- Generated images and files that the export lists only in its file library
  are added to their conversation's note ("Files from this conversation").
- Sources as footnotes, numbered once per conversation; Canvas documents with
  their edits applied; code, output and reasoning folded; generated and
  uploaded images copied with real file extensions.
- `index.md`, an Obsidian `Conversations.base`, and `insights.html` with
  activity, hour-of-week, monthly, model, feature and length charts, and word
  clouds of your prompts: all years, plus one per year showing what set it apart
  (works offline, light and dark; each cloud can be saved as a PNG).
- The default output folder is `~/Documents/ChatGPT Archive`.
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
- Bundled fonts and the matplotlib, wordcloud and NLTK dependencies. Word
  clouds are now drawn by the insights page itself.
- The browser script that downloaded conversations from chatgpt.com directly.
- Python APIs of 0.x (`ConversationCollection`, `run_pipeline`, …); convoviz is
  a command-line tool.
