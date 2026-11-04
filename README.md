<h1 align="center">convoviz</h1>

<p align="center">
  <strong>Your ChatGPT history, as Markdown notes you own.</strong><br>
  Turn your ChatGPT data export into a clean, searchable archive for Obsidian or any Markdown app,<br>
  plus a private page of charts and word clouds about how you use ChatGPT.
</p>

<p align="center">
  <a href="https://pypi.org/project/convoviz/"><img src="https://img.shields.io/pypi/v/convoviz?color=2a78d6" alt="PyPI version"></a>
  <a href="https://pepy.tech/projects/convoviz"><img src="https://img.shields.io/pepy/dt/convoviz?color=2a78d6" alt="Downloads"></a>
  <a href="https://github.com/mohamed-chs/convoviz/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/mohamed-chs/convoviz/ci.yml?label=tests" alt="Tests"></a>
  <a href="https://github.com/mohamed-chs/convoviz/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/convoviz?color=555" alt="MIT license"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#what-you-get">What you get</a> ·
  <a href="#your-chatgpt-in-charts">Charts</a> ·
  <a href="#keeping-your-archive-up-to-date">Updating</a> ·
  <a href="#questions">Questions</a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mohamed-chs/convoviz/main/docs/assets/insights-dark.png">
    <img src="https://raw.githubusercontent.com/mohamed-chs/convoviz/main/docs/assets/insights-light.png" alt="The insights page convoviz creates: totals for conversations, prompts and active days, and a heatmap of when in the week the prompts were sent" width="860">
  </picture>
  <br>
  <sub>Made from an invented export. Yours stays on your computer.</sub>
</p>

## Why convoviz

ChatGPT lets you download all your conversations, but what arrives is a ZIP of raw JSON that's hard to read or search. convoviz turns it into two things:

- a folder of Markdown notes, one per conversation, with your prompts, the answers, and properties for the dates, models and message count. Sources become footnotes, reasoning and tool output are folded away, and edited or regenerated replies are kept. The notes are made for [Obsidian](https://obsidian.md), where a table view lists every conversation, and they also open in VS Code, on GitHub or in any other Markdown app;
- a page of charts: when you ask, how often, which models answered, and word clouds of what you asked about, year by year.

Run it again on next month's export and only what changed is rewritten. Notes you edited, moved or deleted stay as you left them. Everything runs on your computer, with no account or API key, and nothing is uploaded. It reads ChatGPT's 2026 export format, including split exports, `.dat` attachments and multi-part downloads.

## Quick start

**1. Request your export.** In ChatGPT, click your name in the corner, then **Settings → Data controls → Export data**. OpenAI emails you a download link, which can take from a few minutes to a few hours. Download the ZIP; there's no need to unzip it.

**2. Install convoviz.** Open a terminal and paste the line for your system.

macOS / Linux:

```sh
curl -LsSf https://raw.githubusercontent.com/mohamed-chs/convoviz/main/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/mohamed-chs/convoviz/main/install.ps1 | iex"
```

This installs [uv](https://docs.astral.sh/uv/) if you don't have it, which takes care of Python too, then convoviz. If you already use uv: `uv tool install convoviz`.

<details>
<summary>New to the terminal?</summary>

- On macOS, press <kbd>⌘ Space</kbd>, type *Terminal*, press <kbd>Enter</kbd>.
- On Windows, press the <kbd>Windows</kbd> key, type *PowerShell*, press <kbd>Enter</kbd>.
- On Linux, press <kbd>Ctrl Alt T</kbd>, or find *Terminal* in your apps.

Paste the line above for your system and press <kbd>Enter</kbd>. When it's done, close the window and open a new one, so it finds the `convoviz` command.
</details>

**3. Run it.**

```sh
convoviz
```

convoviz finds the newest export in your Downloads folder, checks with you, and writes your archive to `Documents/ChatGPT Archive`:

```text
Found export: chatgpt-export.zip (just now)

✓ 1,488 conversations in ~/Documents/ChatGPT Archive (1.0s)
  1,488 new
  Open insights.html for your charts, or browse index.md.
```

To choose the input or output yourself:

```sh
convoviz ~/Downloads/chatgpt-export.zip -o ~/Notes/ChatGPT
```

The input can also be an extracted folder or a `conversations.json`. If OpenAI split your export into `…-part-0001.zip`, `…-part-0002.zip` and so on, pass any one of them and the other parts are found automatically.

## What you get

```text
ChatGPT Archive/
├── conversations/
│   └── 2026/
│       └── 2026-03-14 Planning a trip to Kyoto.md
├── attachments/            your uploads and generated images
├── canvas/                 the final version of each Canvas document (.md, .py, …)
├── index.md                every conversation, newest first
├── Custom instructions.md  each version of your custom instructions
├── Conversations.base      a sortable, filterable table in Obsidian
└── insights.html           your charts and word clouds
```

To read it in Obsidian, choose **Open folder as vault** and pick `ChatGPT Archive`, or point convoviz at a folder inside a vault you already have.

A note looks like this:

```markdown
---
title: "Planning a trip to Kyoto"
created: 2026-03-14T09:30:12
models:
  - "gpt-5-5-thinking"
messages: 4
tags:
  - "chatgpt"
---

# Planning a trip to Kyoto

## You · 2026-03-14 09:30

Three days in Kyoto in April — what should I not miss?

## ChatGPT · 2026-03-14 09:30

<details>
<summary>Thought for 8s</summary>
…
</details>

April is a lovely time to go: it's cherry blossom season, so plan around the crowds.

#### Day 1 · Southern Higashiyama

- **Fushimi Inari** early in the morning, before the tour groups[^1].
…

[^1]: [Fushimi Inari Taisha](https://inari.example/en/) — inari.example
```

[See the whole note as GitHub renders it.](https://github.com/mohamed-chs/convoviz/blob/main/docs/example/Planning%20a%20trip%20to%20Kyoto.md)

Images are copied with their real file type. Canvas documents keep their edits. Charts become tables and shopping picks become lists. Drafted emails and posts become labelled sections. Anything convoviz doesn't recognize yet is kept as plain text and listed in the summary.

## Your ChatGPT, in charts

`insights.html` is a single file that opens in any browser, works offline, and follows your light or dark theme. It shows:

- your activity, day by day, for every year;
- when in the week you ask, hour by hour;
- prompts per month, and which models answered;
- the features you used (search, reasoning, code, images, Canvas, voice) and how long conversations run;
- word clouds of what you asked about: all years, and each year on its own, showing what set it apart. Each one can be saved as an image.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mohamed-chs/convoviz/main/docs/assets/wordcloud-dark.png">
    <img src="https://raw.githubusercontent.com/mohamed-chs/convoviz/main/docs/assets/wordcloud-light.png" alt="A word cloud of the topics asked about, with tabs for each year" width="860">
  </picture>
</p>

<details>
<summary>More charts: your activity calendar, and which models answered</summary>
<br>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mohamed-chs/convoviz/main/docs/assets/charts-dark.png">
    <img src="https://raw.githubusercontent.com/mohamed-chs/convoviz/main/docs/assets/charts-light.png" alt="A calendar of prompts per day for each year, and a stacked chart of replies per month by model" width="860">
  </picture>
</p>
</details>

Every chart has a **Show data** table, and the colors are chosen to stay readable with color blindness.

## Keeping your archive up to date

Request a new export whenever you like, and run convoviz again on the same folder:

- new and changed conversations are written, and the rest are left alone;
- a conversation renamed in ChatGPT has its note renamed too;
- notes you've edited are kept as they are. If the conversation has newer messages, the summary tells you, and `--force` replaces your version;
- notes you've moved or renamed, into your own folders for example, are found and updated where they are;
- notes you've deleted stay deleted, unless you pass `--force`;
- conversations missing from the new export are kept, since exports are sometimes incomplete.

convoviz recognizes its notes by their `conversation_id` property and remembers what it wrote in `.convoviz/state.json`. It never overwrites a file it didn't create, so it's safe to point at an existing vault.

## Options

| Flag | |
|---|---|
| `-o, --output PATH` | Output folder (default `~/Documents/ChatGPT Archive`). |
| `--layout year\|flat` | One folder per year (default), or a single folder. |
| `--branches all\|active` | Keep edited and regenerated replies, folded (default), or only the branch you last viewed. |
| `--timezone ZONE` | Time zone for dates, such as `Europe/Paris` (default: your computer's). |
| `--no-insights` | Skip `insights.html`. |
| `--force` | Overwrite notes and files you edited, and recreate notes you deleted. |
| `-y, --yes` | Don't ask before using the export found in Downloads. |
| `-q, --quiet` / `-v, --verbose` | Less or more output. |

To keep settings, run `convoviz config init` and edit the file it creates (`convoviz config path` shows where it is). There you can also rename the "You" and "ChatGPT" headings, hide timestamps, reasoning or tool output, choose the tags added to every note, and list words to leave out of the word clouds.

## Questions

<details>
<summary><strong>Is my data sent anywhere?</strong></summary>

No. convoviz reads the export from your disk and writes the archive to a folder on your disk. It makes no network requests, and the charts page loads nothing from the internet.
</details>

<details>
<summary><strong>Does it work with other note apps?</strong></summary>

The notes are standard Markdown (CommonMark with GitHub tables and footnotes), so any app that imports Markdown files can read them. Folded sections use HTML `<details>` tags, which Obsidian, GitHub and VS Code show as collapsible blocks. `Conversations.base` is the one Obsidian-only file.
</details>

<details>
<summary><strong>Can I turn conversations into PDFs?</strong></summary>

Yes, with [convpdf](https://github.com/mohamed-chs/convpdf), a companion tool:

```sh
npm install -g convpdf
cd ~/Documents/"ChatGPT Archive"
convpdf "conversations/**/*.md" -o PDF/
```
</details>

<details>
<summary><strong>Some images are missing.</strong></summary>

Some 2026 exports leave out generated images or library files on OpenAI's side. convoviz puts a placeholder in the note, and the summary counts what's missing.
</details>

<details>
<summary><strong>Can I export a Business, Enterprise or Edu workspace?</strong></summary>

No: ChatGPT's export feature isn't available for those workspaces, and convoviz needs the export.
</details>

<details>
<summary><strong>Where are my custom instructions and memories?</strong></summary>

Custom instructions are collected in `Custom instructions.md`, one section per version, with the dates each was in use. Saved memories aren't written. When ChatGPT updated its memory during a conversation, the note says so.
</details>

<details>
<summary><strong>I used convoviz 0.x before.</strong></summary>

1.0 writes to a new folder, `~/Documents/ChatGPT Archive`, and leaves your old `ChatGPT-Data` folder alone. Don't point it at the old folder: 1.0 would treat the old notes as ones you edited and keep them. If you made a config file with 0.x, run `convoviz config init --force` to get the new format (your old one is kept as `config.toml.bak`).
</details>

## Something looks wrong?

ChatGPT's export format is undocumented and changes often. When convoviz meets something new, the end-of-run summary says so. To report it, run:

```sh
convoviz inspect --json ~/Downloads/chatgpt-export.zip
```

This describes only the export's structure: field names, content types and counts. It never includes titles, messages, links, file names or ids. Paste it into a [bug report](https://github.com/mohamed-chs/convoviz/issues/new?template=bug_report.yml). Please don't attach your export itself.

## Contributing

Ideas, bug reports and notes that it worked for you are welcome in the [issues](https://github.com/mohamed-chs/convoviz/issues). If convoviz saved you some time, a ⭐ helps other people find it.

To work on the code, see [CONTRIBUTING.md](https://github.com/mohamed-chs/convoviz/blob/main/CONTRIBUTING.md). How the pieces fit together is in [docs/dev/architecture.md](https://github.com/mohamed-chs/convoviz/blob/main/docs/dev/architecture.md), and what's known about the export format is in [docs/dev/export-format.md](https://github.com/mohamed-chs/convoviz/blob/main/docs/dev/export-format.md).

## License

[MIT](https://github.com/mohamed-chs/convoviz/blob/main/LICENSE)
