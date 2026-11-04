# convoviz

[![PyPI](https://img.shields.io/pypi/v/convoviz)](https://pypi.org/project/convoviz/)
[![CI](https://img.shields.io/github/actions/workflow/status/mohamed-chs/convoviz/ci.yml?label=CI)](https://github.com/mohamed-chs/convoviz/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/pepy/dt/convoviz)](https://pepy.tech/projects/convoviz)
[![License](https://img.shields.io/pypi/l/convoviz)](LICENSE)

Turn your ChatGPT data export into a folder of Markdown notes you own, plus a page of charts about how you've used ChatGPT.

- **One readable note per conversation.** Each note has its properties (dates, models, message count), your prompts and the answers, and its sources as footnotes. Reasoning, code and tool output are folded away. Charts become tables, shopping picks become lists, and drafted emails and posts become labelled sections. Edited and regenerated replies are kept too, folded at the point where they branch.
- **Works with today's exports.** That includes split exports (`conversations-000.json` …), attachments stored as `.dat` files, ZIPs nested inside ZIPs, and multi-part downloads. Unknown content is kept as plain text and reported, never dropped silently.
- **Safe to re-run.** Run it again on a newer export and only changed conversations are rewritten. Notes you've edited, moved or deleted are respected.
- **Made for [Obsidian](https://obsidian.md)**, and readable anywhere. The notes are plain CommonMark/GFM, so they also render on GitHub, in VS Code, or as text. The archive comes with a `Conversations.base` table view.
- **Private.** Everything runs on your computer, and nothing is sent anywhere.

## Quick start

**1. Get your export.** In ChatGPT, go to **Settings → Data controls → Export data**. OpenAI emails you a download link, and the ZIP lands in your Downloads folder.

**2. Install convoviz.** It needs [uv](https://docs.astral.sh/uv/), which also takes care of Python:

```sh
uv tool install convoviz
```

<details>
<summary>No uv yet? One-line installers</summary>

macOS / Linux:

```sh
curl -LsSf https://raw.githubusercontent.com/mohamed-chs/convoviz/main/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/mohamed-chs/convoviz/main/install.ps1 | iex"
```

Both install uv if it's missing, then convoviz.
</details>

**3. Run it.**

```sh
convoviz
```

convoviz finds the newest export in your Downloads folder, asks you to confirm, and writes the archive to `~/Documents/ChatGPT Archive`. To choose the input or output yourself:

```sh
convoviz ~/Downloads/chatgpt-export.zip -o ~/Notes/ChatGPT
```

You can also pass an extracted folder or a bare `conversations.json`. If OpenAI split your export into `…-part-0001.zip`, `…-part-0002.zip`, …, pass any one of them and the other parts are picked up automatically.

## What you get

```
ChatGPT Archive/
├── conversations/
│   └── 2026/
│       └── 2026-03-14 Planning a trip to Kyoto.md
├── attachments/            your uploads and generated images
├── canvas/                 the final version of each Canvas document (.md, .py, …)
├── index.md                every conversation, newest first
├── Custom instructions.md  each version of your custom instructions
├── Conversations.base      a sortable table in Obsidian
└── insights.html           charts and word clouds of how you've used ChatGPT
```

A note looks like this:

```markdown
---
title: "Planning a trip to Kyoto"
created: 2026-03-14T09:30:12
updated: 2026-03-14T10:02:45
source: "chatgpt"
conversation_id: "67d3…"
url: "https://chatgpt.com/c/67d3…"
models:
  - "gpt-5-5-thinking"
messages: 6
tags:
  - "chatgpt"
---

# Planning a trip to Kyoto

## You · 2026-03-14 09:30

Three days in Kyoto in April — what should I not miss?

## ChatGPT · 2026-03-14 09:31

<details>
<summary>Thought for 8s</summary>
…
</details>

Fushimi Inari early in the morning, before the crowds[^1]. …

[^1]: [Fushimi Inari Taisha](https://inari.jp/en/) — inari.jp
```

`insights.html` is a single self-contained file: your activity calendar, when in the week you ask, models and features over time, and word clouds of what you asked about, for all years and for each year on its own. Open it in any browser; it works offline, in light and dark mode, and each word cloud can be saved as an image.

## Updating your archive

Request a new export whenever you like and run convoviz again on the same output folder:

- new and changed conversations are written, and unchanged ones are left alone;
- a conversation renamed in ChatGPT has its note renamed too;
- **notes you've edited are kept as they are**. If the conversation has newer messages, the summary tells you (`--force` replaces your version);
- **notes you moved or renamed** (into your own folders, say) are found again and updated where they are;
- **notes you deleted stay deleted** (`--force` brings them back);
- `index.md`, `Conversations.base` and `insights.html` are refreshed, unless you changed them;
- conversations missing from the new export are kept, since exports are sometimes incomplete.

convoviz recognizes its notes by their `conversation_id` property, and keeps track of what it wrote in `.convoviz/state.json` inside the output folder. You can point it at an existing Obsidian vault: it never overwrites files it didn't create.

## Options

| Flag | |
|---|---|
| `-o, --output PATH` | Output folder (default `~/Documents/ChatGPT Archive`). |
| `--layout year\|flat` | One folder per year (default) or a single folder. |
| `--branches all\|active` | Keep edited/regenerated alternatives folded (default), or only the branch you last viewed. |
| `--timezone ZONE` | Time zone for dates, e.g. `Europe/Paris` (default: your computer's). |
| `--no-insights` | Skip `insights.html`. |
| `--force` | Overwrite notes and files you edited, and recreate notes you deleted. |
| `-y, --yes` | Don't ask before using the export found in Downloads. |
| `-q, --quiet` / `-v, --verbose` | Less or more output. |

To make settings permanent, run `convoviz config init` and edit the file it creates. `convoviz config path` shows where it is. In that file you can also rename the turn labels ("You", "ChatGPT"), hide timestamps, reasoning or tool output, set the tags added to every note, and list words to leave out of the word clouds.

## PDFs

To turn notes into PDFs, try [convpdf](https://github.com/mohamed-chs/convpdf):

```sh
npm install -g convpdf
cd ~/Documents/"ChatGPT Archive"
convpdf "conversations/**/*.md" -o PDF/
```

## Something looks wrong?

ChatGPT's export format is undocumented and changes often. When convoviz meets something new, the end-of-run summary says so. To report it:

```sh
convoviz inspect --json ~/Downloads/chatgpt-export.zip
```

This prints a description of the export's **structure only**: field names, content types and counts. It never includes titles, message text, links, file names or ids. Paste it into a [bug report](https://github.com/mohamed-chs/convoviz/issues/new?template=bug_report.yml). Please don't attach your export itself.

## Notes

- **Business, Enterprise and Edu workspaces** can't use ChatGPT's export feature. That's an OpenAI limitation, and convoviz needs the export.
- **Images missing?** Some 2026 exports leave out generated images or library files on OpenAI's side. convoviz puts a placeholder in the note and counts what's missing in the summary.
- **Custom instructions** are collected in `Custom instructions.md`, one section per version, with the dates each was in use. Saved memories are not written anywhere.

## Contributing

Ideas, quirks, or just a note that it worked for you: [open an issue](https://github.com/mohamed-chs/convoviz/issues). If convoviz is useful to you, a ⭐ helps others find it.

See [CONTRIBUTING.md](CONTRIBUTING.md). For how the code fits together, see [docs/dev/architecture.md](docs/dev/architecture.md). What's known about the export format is in [docs/dev/export-format.md](docs/dev/export-format.md).

## License

MIT
