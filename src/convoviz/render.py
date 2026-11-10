"""Render a conversation as one Markdown note.

The output targets Obsidian first (properties, footnotes, ``$`` math) but stays
plain CommonMark + GFM so it reads well on GitHub, in VS Code, or as text:
collapsible parts use ``<details>`` rather than Obsidian-only callouts.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote

import orjson

from convoviz.fsnames import safe_stem
from convoviz.markdown import (
    blockquote,
    close_fences,
    currency_escaped,
    escape_inline,
    fence_for,
    html_escaped,
    link_url,
    map_prose,
    math_to_dollars,
)
from convoviz.model import (
    ASSISTANT_NAMES,
    CITE_CLOSE,
    CITE_OPEN,
    AttachmentBlock,
    CodeBlock,
    DocumentBlock,
    FileTextBlock,
    Message,
    NoticeBlock,
    OutputBlock,
    QuoteBlock,
    ReasoningBlock,
    Source,
    TextBlock,
)
from convoviz.tree import Fork, Thread, linearize

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime, tzinfo

    from convoviz.model import Attachment, Block, Conversation, Memory, Project

Branches = Literal["all", "active"]


@dataclass(frozen=True, slots=True)
class RenderOptions:
    """User-facing knobs for the Markdown output."""

    user_label: str = "You"
    assistant_label: str | None = None  # None: the assistant's name ("ChatGPT", "Claude")
    timestamps: bool = True
    branches: Branches = "all"
    reasoning: bool = True
    tool_calls: bool = True
    tags: tuple[str, ...] | None = None  # None: the provider ("chatgpt", "claude")
    timezone: tzinfo | None = None  # None = the system's local time zone

    def assistant_for(self, conversation: Conversation) -> str:
        """The heading of the assistant's turns in ``conversation``."""
        return self.assistant_label or ASSISTANT_NAMES.get(conversation.provider, "Assistant")

    def tags_for(self, conversation: Conversation) -> tuple[str, ...]:
        """The tags of ``conversation``'s note."""
        return self.provider_tags(conversation.provider)

    def provider_tags(self, provider: str) -> tuple[str, ...]:
        """The tags of notes from ``provider`` (conversations, projects)."""
        return self.tags if self.tags is not None else (provider,)


@dataclass(frozen=True, slots=True)
class Note:
    """A rendered conversation."""

    text: str
    messages: int  # user + assistant messages on the main branch


def render_conversation(
    conversation: Conversation,
    options: RenderOptions,
    resolve: Callable[[Attachment], str | None],
    link: Callable[[str], str] | None = None,
) -> Note:
    """The complete note: properties, title, turns, footnotes.

    ``link`` turns a path relative to the archive root into a link from the
    note; without it, Canvas documents aren't linked to their saved files.
    """
    thread = linearize(conversation, include_forks=options.branches == "all")
    canvases = (
        {block: link(path) for path, block in document_files(conversation).items()} if link else {}
    )
    renderer = _Renderer(options, options.assistant_for(conversation), resolve, canvases)
    body = renderer.thread(thread, nested=False)
    parts = [
        frontmatter(conversation, thread, options),
        f"# {escape_inline(conversation.title)}",
        body,
        renderer.files(conversation.files),
    ]
    if renderer.footnotes:
        parts.append(renderer.footnote_section())
    text = "\n\n".join(p for p in parts if p).rstrip() + "\n"
    return Note(text, count_messages(thread))


def count_messages(thread: Thread) -> int:
    """Messages as a reader counts them: each prompt, and each reply as a whole.

    One reply is often several stored messages (reasoning, tool calls, answer).
    """
    count = 0
    previous_was_user: bool | None = None
    for message in thread.messages():
        is_user = message.role == "user"
        if is_user or previous_was_user is not False:
            count += 1
        previous_was_user = is_user
    return count


# --------------------------------------------------------------------------- properties


def _yaml(value: object) -> str:
    """A YAML scalar. JSON strings are valid YAML double-quoted strings."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return orjson.dumps(str(value)).decode()


def local_time(moment: datetime, tz: tzinfo | None) -> datetime:
    """Convert to the configured zone (system local when ``tz`` is None)."""
    return moment.astimezone(tz)


def _iso(moment: datetime, tz: tzinfo | None) -> str:
    # Obsidian's Date & time property type: YYYY-MM-DDTHH:MM:SS, no offset.
    return local_time(moment, tz).replace(tzinfo=None, microsecond=0).isoformat()


def frontmatter(conversation: Conversation, thread: Thread, options: RenderOptions) -> str:
    """YAML properties, using the same names and types across every note."""
    messages = thread.messages()
    models = Counter(m.model for m in messages if m.role == "assistant" and m.model)
    if not models and conversation.default_model:
        models[conversation.default_model] = 1
    lines = [
        "---",
        f"title: {_yaml(conversation.title)}",
        "aliases:",
        f"  - {_yaml(conversation.title)}",
        f"created: {_iso(conversation.created, options.timezone)}",
        f"updated: {_iso(conversation.updated, options.timezone)}",
        f"source: {_yaml(conversation.provider)}",
        f"conversation_id: {_yaml(conversation.id)}",
    ]
    if conversation.url:
        lines.append(f"url: {_yaml(conversation.url)}")
    if models:
        lines.append("models:")
        lines.extend(f"  - {_yaml(name)}" for name, _ in models.most_common())
    lines.append(f"messages: {count_messages(thread)}")
    if conversation.project_id:
        lines.append(f"project_id: {_yaml(conversation.project_id)}")
    if conversation.gizmo_id:
        lines.append(f"gpt_id: {_yaml(conversation.gizmo_id)}")
    if conversation.starred:
        lines.append("starred: true")
    if conversation.archived:
        lines.append("archived: true")
    if tags := options.tags_for(conversation):
        lines.append("tags:")
        lines.extend(f"  - {_yaml(tag)}" for tag in tags)
    lines.append("---")
    return "\n".join(lines)


# --------------------------------------------------------------------------- body


_HEADING_RE = re.compile(r"^(#{1,6})(?=[ \t])", re.MULTILINE)
# Leading spaces are swallowed so footnotes hug the word they annotate.
_CITE_RE = re.compile(r"[ \t]*" + re.escape(CITE_OPEN) + r"([\d,]*)" + re.escape(CITE_CLOSE))


def _demoted(segment: str) -> str:
    """Push headings below the turn headings (by two levels) so the note outline stays intact."""
    if "#" not in segment:
        return segment
    return _HEADING_RE.sub(lambda m: "#" * min(6, len(m.group(1)) + 2), segment)


def _details(summary: str, body: str, *, open_: bool = False) -> str:
    """A folded section with a plain-text ``summary``.

    The summary sits in raw HTML, where Markdown escapes would show literally,
    so it is HTML-escaped instead.
    """
    # The blank line after <summary> makes Markdown inside render (CommonMark HTML block rule).
    attr = " open" if open_ else ""
    summary = html.escape(" ".join(summary.split()), quote=False)
    return f"<details{attr}>\n<summary>{summary}</summary>\n\n{body.strip()}\n\n</details>"


def _fenced(content: str, language: str = "") -> str:
    fence = fence_for(content)
    return f"{fence}{language}\n{content.rstrip()}\n{fence}"


def _link_target(target: str) -> str:
    return quote(target, safe="/._-~#:")


def _merge_reasoning(messages: list[Message]) -> list[tuple[Message, Block]]:
    """Join back-to-back reasoning into one panel, as ChatGPT shows it.

    Reasoning steps, the progress notes between them and the recap that ends
    them ("Thought for 12s", which names the panel) become a single block.
    """
    pairs: list[tuple[Message, Block]] = []
    for message in messages:
        for block in message.blocks:
            previous = pairs[-1][1] if pairs else None
            if (
                isinstance(block, ReasoningBlock)
                and isinstance(previous, ReasoningBlock)
                and not previous.title  # a recap closes the panel
            ):
                text = "\n\n".join(t for t in (previous.text, block.text) if t)
                pairs[-1] = (pairs[-1][0], ReasoningBlock(text, block.title or previous.title))
                continue
            pairs.append((message, block))
    return pairs


class _Renderer:
    def __init__(
        self,
        options: RenderOptions,
        assistant: str,
        resolve: Callable[[Attachment], str | None],
        canvases: dict[DocumentBlock, str] | None = None,
    ) -> None:
        self.options = options
        self.assistant = assistant
        self.resolve = resolve
        self.canvases = canvases or {}
        self.footnotes: list[Source] = []
        self._footnote_ids: dict[tuple[str | None, str | None], int] = {}

    # ----------------------------------------------------------------- structure

    def thread(self, thread: Thread, *, nested: bool) -> str:
        """Group consecutive non-user messages into one turn, render forks in between."""
        out: list[str] = []
        turn: list[Message] = []

        def flush() -> None:
            if turn:
                rendered = self.turn(turn, nested=nested)
                if rendered:
                    out.append(rendered)
                turn.clear()

        for item in thread.items:
            if isinstance(item, Fork):
                flush()
                out.append(self.fork(item))
                continue
            if turn and (item.role == "user") != (turn[0].role == "user"):
                flush()
            elif turn and (item.role == "user" or item.scheduled):
                flush()  # two prompts in a row, or a scheduled run: separate turns
            turn.append(item)
        flush()
        return "\n\n".join(out)

    def fork(self, fork: Fork) -> str:
        count = len(fork.alternatives)
        noun = "version" if count == 1 else "versions"
        sections = []
        for number, alternative in enumerate(fork.alternatives, 1):
            body = self.thread(alternative, nested=True)
            label = f"**Other version {number} of {count}**" if count > 1 else ""
            sections.append(f"{label}\n\n{body}" if label else body)
        return _details(
            f"↳ {count} other {noun} of the next part",
            "\n\n---\n\n".join(sections),
        )

    def turn(self, messages: list[Message], *, nested: bool) -> str:
        rendered = [self.block(b, m) for m, b in _merge_reasoning(messages)]
        body = "\n\n".join(b for b in rendered if b)
        if not body.strip():
            return ""
        first = messages[0]
        label = self.options.user_label if first.role == "user" else self.assistant
        stamp = ""
        when = next((m.time for m in messages if m.time), None)
        if self.options.timestamps and when is not None:
            stamp = f" · {local_time(when, self.options.timezone):%Y-%m-%d %H:%M}"
        heading = (
            f"**{escape_inline(label)}**{stamp}" if nested else f"## {escape_inline(label)}{stamp}"
        )
        return f"{heading}\n\n{body}"

    # ----------------------------------------------------------------- blocks

    def block(self, block: Block, message: Message) -> str:
        match block:
            case TextBlock():
                return self.text(block, message)
            case ReasoningBlock():
                if not self.options.reasoning:
                    return ""
                title = block.title or "Reasoning"
                if not block.text.strip():
                    return f"*{escape_inline(title)}*"
                return _details(title, self.prose(block.text))
            case CodeBlock():
                if not self.options.tool_calls or not block.code.strip():
                    return ""
                summary = block.title or "Code" + (f" ({block.language})" if block.language else "")
                return _details(summary, _fenced(block.code, block.language))
            case OutputBlock():
                if not self.options.tool_calls or not block.text.strip():
                    return ""
                return _details(block.title or "Output", _fenced(block.text))
            case QuoteBlock():
                if not self.options.tool_calls or not block.text.strip():
                    return ""
                return self.quote(block)
            case AttachmentBlock():
                return self.attachment(block)
            case FileTextBlock():
                return self.file_text(block)
            case DocumentBlock():
                return self.document(block)
            case NoticeBlock():
                return f"*{escape_inline(block.text)}*"
        return ""

    def text(self, block: TextBlock, message: Message) -> str:
        text = block.text
        if not text.strip():
            return ""
        if message.role == "user":
            text = map_prose(text, html_escaped, currency_escaped, _demoted)
            return close_fences(text).strip()
        text = self.prose(text)
        if block.sources:
            text = _CITE_RE.sub(lambda m: self.cite(m.group(1), block.sources), text)
        return text

    def prose(self, text: str) -> str:
        """Assistant-style Markdown: math delimiters, currency, headings, fences."""
        text = map_prose(text, currency_escaped, math_to_dollars, _demoted)
        return close_fences(text).strip()

    def cite(self, ids: str, sources: tuple[Source, ...]) -> str:
        refs: list[int] = []
        for raw in ids.split(","):
            if not raw.isdigit() or int(raw) >= len(sources):
                continue
            source = sources[int(raw)]
            key = (source.url, None if source.url else source.title)
            if key not in self._footnote_ids:
                self.footnotes.append(source)
                self._footnote_ids[key] = len(self.footnotes)
            number = self._footnote_ids[key]
            if number not in refs:
                refs.append(number)
        return "".join(f"[^{n}]" for n in refs)

    def footnote_section(self) -> str:
        lines = []
        for number, source in enumerate(self.footnotes, 1):
            title = escape_inline(source.title or source.url or "Source")
            text = f"[{title}]({link_url(source.url)})" if source.url else title
            if source.attribution and source.attribution not in (source.title or ""):
                text += f" — {escape_inline(source.attribution)}"
            lines.append(f"[^{number}]: {text}")
        return "\n".join(lines)

    def quote(self, block: QuoteBlock) -> str:
        quoted = blockquote(block.text.strip())
        title = escape_inline(block.title or block.url or "")
        if block.url:
            title = f"[{title}]({link_url(block.url)})"
        return f"{quoted}\n>\n> — {title}" if title else quoted

    def attachment(self, block: AttachmentBlock) -> str:
        attachment = block.attachment
        name = attachment.name or {
            "image": "Image",
            "audio": "Audio",
            "video": "Video",
        }.get(attachment.kind, "File")
        target = self.resolve(attachment)
        if target is None:
            return f"*[{escape_inline(name)} not included in the export]*"
        label = escape_inline(block.caption or attachment.name or "")
        if attachment.kind == "image":
            return f"![{label}]({_link_target(target)})"
        return f"[📎 {escape_inline(name)}]({_link_target(target)})"

    def file_text(self, block: FileTextBlock) -> str:
        """An attached file shown as its extracted text, folded and verbatim."""
        if not block.text.strip():
            return ""
        name = block.name or "Pasted text"
        suffix = PurePosixPath(block.name or "").suffix.lower()
        language = _FENCE_LANGUAGES.get(suffix, "")
        return _details(f"📎 {name}", _fenced(block.text, language))

    def files(self, attachments: list[Attachment]) -> str:
        """Files linked to the conversation outside its messages."""
        if not attachments:
            return ""
        items = [self.attachment(AttachmentBlock(a)) for a in attachments]
        return "## Files from this conversation\n\n" + "\n\n".join(items)

    def document(self, block: DocumentBlock) -> str:
        version = f" (v{block.version})" if block.version > 1 else ""
        doc_type = block.doc_type or ""
        if doc_type.startswith("code/"):
            body = _fenced(block.content, doc_type.removeprefix("code/"))
        else:
            body = self.prose(block.content)
        label = f"{block.kind}: {block.title}" if block.title else block.kind
        if saved := self.canvases.get(block):
            name = escape_inline(saved.rsplit("/", 1)[-1])
            body += f"\n\n*Final version saved as [{name}]({_link_target(saved)}).*"
        return _details(f"🗎 {label}{version}", body, open_=True)


# Code-fence languages of attached files whose text is shown.
_FENCE_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".json": "json",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".sh": "bash",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".csv": "csv",
    ".xml": "xml",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".tex": "latex",
}


# --------------------------------------------------------------------------- pages

CANVAS_DIR = "canvas"
ARTIFACTS_DIR = "artifacts"
# Where the final version of each kind of document is saved.
_DOCUMENT_DIRS = {
    "Canvas": CANVAS_DIR,
    "Artifact": ARTIFACTS_DIR,
    "File": ARTIFACTS_DIR,
    "Widget": ARTIFACTS_DIR,
}
_CANVAS_EXTENSIONS = {
    "document": ".md",
    "document/markdown": ".md",
    "text/plain": ".txt",
    "code/python": ".py",
    "code/javascript": ".js",
    "code/typescript": ".ts",
    "code/react": ".jsx",
    "code/html": ".html",
    "code/css": ".css",
    "code/json": ".json",
    "code/bash": ".sh",
    "code/shell": ".sh",
    "code/sql": ".sql",
    "code/java": ".java",
    "code/kotlin": ".kt",
    "code/swift": ".swift",
    "code/go": ".go",
    "code/rust": ".rs",
    "code/ruby": ".rb",
    "code/php": ".php",
    "code/c": ".c",
    "code/cpp": ".cpp",
    "code/csharp": ".cs",
    "code/r": ".r",
    "code/yaml": ".yaml",
    "code/latex": ".tex",
    "code/tsx": ".tsx",
    "code/svg": ".svg",
    "code/mermaid": ".mmd",
    "code/csv": ".csv",
    "code/xml": ".xml",
    "code/text": ".txt",
}


def document_files(conversation: Conversation) -> dict[str, DocumentBlock]:
    """The final version of each document on the active branch, by file path.

    ChatGPT Canvas documents go to ``canvas/``, Claude artifacts and the files
    Claude wrote to ``artifacts/``. Paths are relative to the archive root and
    stable across runs: the document's title plus the conversation id, so they
    don't collide.
    """
    latest: dict[tuple[str, str], DocumentBlock] = {}
    for message in linearize(conversation, include_forks=False).messages():
        for block in message.blocks:
            if isinstance(block, DocumentBlock) and block.kind in _DOCUMENT_DIRS:
                latest[_DOCUMENT_DIRS[block.kind], block.title] = block
    suffix = conversation.id.replace("-", "")[:8]
    files: dict[str, DocumentBlock] = {}
    for (folder, title), block in latest.items():
        ext = _CANVAS_EXTENSIONS.get(block.doc_type or "document", ".txt")
        stem = safe_stem(PurePosixPath(title).stem if block.kind == "File" else title)
        if block.kind == "File":
            ext = PurePosixPath(title).suffix or ext
        files[f"{folder}/{stem} ({suffix}){ext}"] = block
    return files


INSTRUCTIONS_PAGE = "Custom instructions.md"
_INSTRUCTION_FIELDS = (
    ("user_profile", "About you"),
    ("user_instructions", "How ChatGPT should respond"),
)


def render_instructions(
    conversations: list[Conversation],
    link: Callable[[Conversation], str | None],
    tz: tzinfo | None,
) -> str | None:
    """Every version of the custom instructions, newest first, with where it was used.

    ``link`` gives the relative link to a conversation's note (or ``None``).
    Returns ``None`` when no conversation recorded custom instructions.
    """
    versions: dict[tuple[str, ...], list[Conversation]] = {}
    for conversation in conversations:
        found = conversation.extra.get("custom_instructions")
        if not isinstance(found, dict) or not found:
            continue
        key = tuple(str(found.get(name) or "") for name, _ in _INSTRUCTION_FIELDS)
        versions.setdefault(key, []).append(conversation)
    if not versions:
        return None

    def day(conversation: Conversation) -> str:
        return f"{local_time(conversation.created, tz):%Y-%m-%d}"

    ordered = sorted(versions.items(), key=lambda kv: max(c.created for c in kv[1]), reverse=True)
    lines = [
        "# Custom instructions",
        "",
        (
            "What you told ChatGPT about yourself and how it should respond, as recorded "
            "in your conversations. Newest version first, with the dates it was in use."
        ),
    ]
    for texts, used in ordered:
        used.sort(key=lambda c: c.created)
        first, last = used[0], used[-1]
        span = day(first) if day(first) == day(last) else f"{day(first)} to {day(last)}"
        count = f"{len(used)} conversation{'s' if len(used) != 1 else ''}"
        lines += ["", f"## {span} · {count}", ""]
        for (_, heading), text in zip(_INSTRUCTION_FIELDS, texts, strict=True):
            if text:
                body = map_prose(text, html_escaped, currency_escaped, _demoted)
                lines += [f"**{heading}**", "", close_fences(body).strip(), ""]
        target = link(last)
        title = escape_inline(last.title)
        lines.append(f"Last used in [{title}]({target})." if target else f"Last used in {title}.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- projects, memory

PROJECTS_DIR = "projects"
MEMORY_PAGE = "Claude memory.md"
# Knowledge documents are saved as the text Claude extracted, so a PDF's text
# goes to "report.pdf.md" rather than a broken "report.pdf".
_TEXT_SUFFIXES = frozenset(
    {*_FENCE_LANGUAGES, ".txt", ".markdown", ".tsv", ".toml", ".ini", ".rst", ".log", ".svg"}
)


def project_paths(projects: list[Project]) -> dict[str, Project]:
    """``projects/<name>.md`` for each project (plus the id when names clash)."""
    stems = Counter(safe_stem(p.name).casefold() for p in projects)
    paths: dict[str, Project] = {}
    for project in projects:
        stem = safe_stem(project.name)
        if stems[stem.casefold()] > 1:
            stem = f"{stem} ({project.id.replace('-', '')[:8]})"
        paths[f"{PROJECTS_DIR}/{stem}.md"] = project
    return paths


def project_doc_paths(note_path: str, project: Project) -> list[str]:
    """Where each knowledge document of ``project`` goes: a folder next to its note."""
    folder = note_path.removesuffix(".md")
    used: Counter[str] = Counter()
    paths: list[str] = []
    for doc in project.docs:
        name = PurePosixPath(doc.name)
        suffix = name.suffix.lower()
        stem = safe_stem(name.stem if suffix in _TEXT_SUFFIXES else name.name)
        ext = suffix if suffix in _TEXT_SUFFIXES else ".md"
        candidate = f"{stem}{ext}"
        used[candidate.casefold()] += 1
        if used[candidate.casefold()] > 1:
            candidate = f"{stem} ({used[candidate.casefold()]}){ext}"
        paths.append(f"{folder}/{candidate}")
    return paths


def render_project(
    project: Project, doc_paths: list[str], note_path: str, options: RenderOptions
) -> str:
    """A project's note: its description, instructions and knowledge documents."""
    tz = options.timezone
    lines = ["---", f"title: {_yaml(project.name)}"]
    if project.created:
        lines.append(f"created: {_iso(project.created, tz)}")
    if project.updated:
        lines.append(f"updated: {_iso(project.updated, tz)}")
    lines.append(f"project_id: {_yaml(project.id)}")
    if project.url:
        lines.append(f"url: {_yaml(project.url)}")
    if tags := options.provider_tags(project.provider):
        lines.append("tags:")
        lines.extend(f"  - {_yaml(tag)}" for tag in tags)
    lines += ["---", "", f"# {escape_inline(project.name)}"]
    if project.description:
        lines += ["", _prose(project.description)]
    if project.instructions:
        lines += ["", "## Instructions", "", _prose(project.instructions)]
    if project.docs:
        lines += ["", "## Knowledge", ""]
        base = PurePosixPath(note_path).parent
        for doc, path in zip(project.docs, doc_paths, strict=True):
            target = _link_target(str(PurePosixPath(path).relative_to(base)))
            added = f" · added {local_time(doc.created, tz):%Y-%m-%d}" if doc.created else ""
            lines.append(f"- [{escape_inline(doc.name)}]({target}){added}")
    name = ASSISTANT_NAMES.get(project.provider, project.provider)
    lines += [
        "",
        (
            f"*Conversations don't record their project in {name}'s export, so they "
            "aren't linked from here.*"
        ),
    ]
    return "\n".join(lines) + "\n"


_FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---[ \t]*(?:\n|\Z)", re.DOTALL)


def render_memory(memory: Memory, tz: tzinfo | None) -> str:
    """What the assistant remembered: its summary, then each memory file."""
    name = ASSISTANT_NAMES.get(memory.provider, memory.provider)
    lines = [
        f"# {name} memory",
        "",
        f"What {name} remembered about you when the export was made.",
    ]
    if memory.summary:
        lines += ["", "## Summary", "", _prose(memory.summary)]
    for file in sorted(memory.files, key=lambda f: f.path):
        lines += ["", f"## {escape_inline(file.path)}", ""]
        if file.updated:
            lines += [f"*Updated {local_time(file.updated, tz):%Y-%m-%d}*", ""]
        lines.append(_prose(_FRONT_MATTER_RE.sub("", file.content.replace("\r\n", "\n"))))
    return "\n".join(lines) + "\n"


def _prose(text: str) -> str:
    """Markdown the user or the assistant wrote, kept below the page's own headings."""
    return close_fences(map_prose(text, html_escaped, currency_escaped, _demoted)).strip()
