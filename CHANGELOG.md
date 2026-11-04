# Changelog

User-visible changes. Format: [Keep a Changelog](https://keepachangelog.com),
versions follow [SemVer](https://semver.org).

## Unreleased

### Changed

- The end-of-run summary is easier to read: counts get thousands separators,
  the output folder is shown as `~/…`, `index.md` and `insights.html` are
  clickable in terminals that support links, and the first run says how to
  open the archive in Obsidian.
- `convoviz inspect` with no path uses the newest export in Downloads, like
  `convoviz` does.
- Config errors say where a misplaced setting belongs (for example,
  `timezone` goes under `[markdown]`).
- `--help` no longer shows `[default: False]` next to every flag.
- The "When you ask" heatmap in `insights.html` is larger and fills its card.

### Fixed

- The models chart counted reasoning steps and tool calls as separate
  replies, which made reasoning models look busier than they were. Each reply
  now counts once.
- A note that failed to write (a full disk, say) is written on the next run.
  Before, convoviz took it for a note you had deleted and skipped it.
- On macOS and Windows, a title that changed only in case (`chat` to `Chat`)
  no longer deletes the note.
- A damaged file inside the export ZIP is listed as an error and the rest is
  converted. Before, the run stopped with a traceback.
- A cut-short download is reported as such, instead of "Not a ZIP archive".
- Titles with square brackets no longer garble or crash the summary.
- When there are more warnings or errors than the summary lists, it says how
  many were left out.

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
- Interactive replies are written from their Markdown version instead of
  widget code.
- Charts in replies become tables, and image search results become links to
  their pages.
- Shopping carousels list their products, with prices and shops.
- A reply's "Sources" list keeps the sources it didn't cite inline.
- Generated images and documents that are only in the export's file library
  appear in the message that produced them, when the export kept it.
- A warning when the export's manifest lists files that aren't there, which
  usually means a part of a split download is missing.
- Word clouds leave out filler words in French, Spanish, Portuguese, German,
  Italian and Arabic too, not only English, and an `ignored_words` setting
  leaves out words of your choice.
- `Custom instructions.md` lists every version of your custom instructions,
  with the dates it was in use and the last conversation that used it.
- The final version of each Canvas document is also saved as its own file in
  `canvas/` (`.md`, `.py`, `.html`, …), linked from the note.
- A config file left over from convoviz 0.x gets a message saying how to
  replace it, and `convoviz config init --force` keeps the replaced file as
  `config.toml.bak`.
- Progress notes written while the model thinks are folded into its
  reasoning.
- Drafts ChatGPT shows as cards (emails, posts, documents) appear as labelled
  sections, such as "Email: subject", instead of plain text.

### Fixed
- Conversations rendered only their last message (exports no longer list
  `children`) (#62, #70).
- One malformed or empty entry aborted the whole run (#60, #70).
- Non-Latin titles produced empty or garbled file names (#64).
- Broken emoji (unpaired surrogates) failed to load (#69).
- Footnote numbers restarted, and collided, in every message.
- Generated images get their prompt as a caption when the image itself
  doesn't carry it.
- AVIF images and QuickTime videos stored as `.dat` get the right extension.

### Removed
- Bundled fonts and the matplotlib, wordcloud and NLTK dependencies. Word
  clouds are now drawn by the insights page itself.
- The browser script that downloaded conversations from chatgpt.com directly.
- Python APIs of 0.x (`ConversationCollection`, `run_pipeline`, …); convoviz is
  a command-line tool.
