"""The claude.ai adapter, on synthetic conversations shaped like the research notes."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC
from typing import TYPE_CHECKING, Any

from factory import (
    CLAUDE_ROOT,
    ClaudeBuilder,
    claude_memory,
    claude_project,
    tool_result,
    tool_use,
    write_zip,
)

from convoviz.claude import read_memory, read_projects
from convoviz.claude.parse import parse_conversation
from convoviz.diagnostics import Diagnostics
from convoviz.model import (
    CITE_CLOSE,
    CITE_OPEN,
    AttachmentBlock,
    CodeBlock,
    Conversation,
    DocumentBlock,
    FileTextBlock,
    Message,
    NoticeBlock,
    OutputBlock,
    ReasoningBlock,
    TextBlock,
)
from convoviz.render import (
    RenderOptions,
    document_files,
    project_doc_paths,
    project_paths,
    render_conversation,
    render_memory,
    render_project,
)
from convoviz.source import ExportSource
from convoviz.tree import Fork, linearize

if TYPE_CHECKING:
    from pathlib import Path


def _parse(raw: dict[str, Any], diagnostics: Diagnostics | None = None) -> Conversation:
    conversation = parse_conversation(raw, diagnostics or Diagnostics())
    assert conversation is not None
    return conversation


def _main(conversation: Conversation) -> list[Message]:
    return linearize(conversation, include_forks=False).messages()


def _blocks(conversation: Conversation, role: str = "assistant") -> list[Any]:
    return [b for m in _main(conversation) if m.role == role for b in m.blocks]


def test_basic_conversation() -> None:
    b = ClaudeBuilder(name="Soup")
    b.human("How do I make soup?")
    b.assistant("Boil water.")
    conversation = _parse(b.build())
    assert conversation.provider == "claude"
    assert conversation.title == "Soup"
    assert conversation.url == f"https://claude.ai/chat/{b.uuid}"
    assert [m.role for m in _main(conversation)] == ["user", "assistant"]
    assert _main(conversation)[1].text == "Boil water."
    assert conversation.created.year == 2026


def test_title_falls_back_to_first_prompt() -> None:
    b = ClaudeBuilder(name="")
    b.human("  Explain   the difference between a list and a tuple in Python, with examples  ")
    b.assistant("Sure.")
    assert _parse(b.build()).title == "Explain the difference between a list and a tuple in…"
    short = ClaudeBuilder(name="")
    short.human("Hi")
    assert _parse(short.build()).title == "Hi"
    empty = ClaudeBuilder(name="")
    assert _parse(empty.build()).title == "Untitled"


def test_branches_follow_parent_links() -> None:
    b = ClaudeBuilder()
    first = b.human("q")
    b.assistant("old answer")
    b.assistant("new answer", parent=first)  # regenerated: a sibling of the old answer
    conversation = _parse(b.build())
    assert [m.text for m in _main(conversation)] == ["q", "new answer"]
    forks = linearize(conversation, include_forks=True)
    assert any(isinstance(item, Fork) for item in forks.items)


def test_several_roots_and_orphans() -> None:
    b = ClaudeBuilder()
    b.human("first", parent=CLAUDE_ROOT)
    b.assistant("reply")
    b.human("lost parent", parent="not-in-this-conversation")
    diagnostics = Diagnostics()
    conversation = _parse(b.build(), diagnostics)
    assert diagnostics.counters["claude_orphan_messages"] == 1
    # The orphan stays on the branch, after the message before it.
    assert [m.text for m in _main(conversation)] == ["first", "reply", "lost parent"]


def test_legacy_messages_without_parent_links_are_chained_by_time() -> None:
    b = ClaudeBuilder()
    b.human("one", time=100)
    b.assistant("two", time=200)
    raw = b.build()
    for message in raw["chat_messages"]:
        del message["parent_message_uuid"]
    raw["chat_messages"].reverse()
    assert [m.text for m in _main(_parse(raw))] == ["one", "two"]


def test_text_falls_back_to_top_level_text() -> None:
    b = ClaudeBuilder()
    b.human("q")
    raw = b.build()
    raw["chat_messages"][0]["content"] = []
    assert _main(_parse(raw))[0].text == "q"


def test_unusable_conversation_is_skipped() -> None:
    diagnostics = Diagnostics()
    assert parse_conversation({"name": "x"}, diagnostics) is None
    assert diagnostics.skipped


def test_citations_become_sources() -> None:
    b = ClaudeBuilder()
    b.human("news?")
    text = "Rain is expected tomorrow."
    citation = {
        "uuid": "c1",
        "start_index": 0,
        "end_index": 10,  # inside "expected": moved to the end of the word
        "details": {"type": "web_search_citation", "url": "https://example.com/w"},
    }
    b.assistant(
        [
            tool_use("web_search", {"query": "weather"}),
            tool_result(
                "web_search",
                [{"type": "knowledge", "title": "Weather", "url": "https://example.com/w"}],
            ),
            {"type": "text", "text": text, "citations": [citation, {**citation, "end_index": 26}]},
        ]
    )
    (block,) = [b for b in _blocks(_parse(b.build())) if isinstance(b, TextBlock)]
    assert block.sources[0].url == "https://example.com/w"
    assert block.sources[0].title == "Weather"
    assert block.text.count(CITE_OPEN) == 2
    assert block.text.startswith(f"Rain is expected{CITE_OPEN}0{CITE_CLOSE} tomorrow.")


def test_thinking_and_hidden_blocks() -> None:
    b = ClaudeBuilder()
    b.human("q")
    b.assistant(
        [
            {"type": "injected_prompt_block", "text": "context claude.ai added"},
            {"type": "thinking", "thinking": "Let me see.", "summaries": [{"summary": "Seeing"}]},
            {"type": "thinking", "thinking": "   "},
            {"type": "token_budget"},
            {"type": "text", "text": "Answer."},
        ]
    )
    diagnostics = Diagnostics()
    blocks = _blocks(_parse(b.build(), diagnostics))
    assert blocks == [ReasoningBlock("Let me see.", "Seeing"), TextBlock("Answer.")]
    assert diagnostics.counters["hidden:injected_prompt_block"] == 1


def test_unknown_block_keeps_its_text() -> None:
    b = ClaudeBuilder()
    b.assistant([{"type": "hologram", "text": "still readable"}])
    diagnostics = Diagnostics()
    assert _blocks(_parse(b.build(), diagnostics)) == [TextBlock("still readable")]
    assert diagnostics.unknown["claude-block:hologram"] == 1


def test_artifacts_create_and_update_per_branch() -> None:
    b = ClaudeBuilder()
    prompt = b.human("write a poem")
    create = {"command": "create", "id": "poem", "type": "text/markdown", "title": "Poem"}
    b.assistant([tool_use("artifacts", {**create, "content": "Roses are red"})])
    b.human("bluer")
    update = {"command": "update", "id": "poem", "old_str": "red", "new_str": "blue"}
    b.assistant([tool_use("artifacts", update)])
    # A regenerated first answer starts its own history of the artifact.
    b.assistant([tool_use("artifacts", {**create, "content": "Violets"})], parent=prompt)
    conversation = _parse(b.build())
    documents = [b for b in _blocks(conversation) if isinstance(b, DocumentBlock)]
    assert [(d.content, d.version) for d in documents] == [("Violets", 1)]
    every = [
        b
        for n in conversation.nodes.values()
        if n.message
        for b in n.message.blocks
        if isinstance(b, DocumentBlock)
    ]
    assert sorted((d.content, d.version) for d in every) == [
        ("Roses are blue", 2),
        ("Roses are red", 1),
        ("Violets", 1),
    ]
    assert {d.kind for d in every} == {"Artifact"}


def test_code_artifact_type() -> None:
    b = ClaudeBuilder()
    data = {
        "command": "create",
        "id": "a",
        "type": "application/vnd.ant.code",
        "language": "Python",
        "title": "Script",
        "content": "print(1)",
    }
    b.assistant([tool_use("artifacts", data)])
    (document,) = _blocks(_parse(b.build()))
    assert document.doc_type == "code/python"


def test_update_without_artifact_is_reported() -> None:
    b = ClaudeBuilder()
    b.assistant([tool_use("artifacts", {"command": "update", "id": "x", "old_str": "a"})])
    diagnostics = Diagnostics()
    assert _blocks(_parse(b.build(), diagnostics)) == []
    assert diagnostics.unknown["artifact:update-without-artifact"] == 1


def test_files_written_and_edited_under_both_tool_names() -> None:
    b = ClaudeBuilder()
    b.human("make a script")
    path = "/mnt/user-data/outputs/hello.py"
    b.assistant(
        [
            tool_use("create_file", {"path": path, "file_text": "print('hi')\n"}, "t1"),
            tool_use("str_replace", {"path": path, "old_str": "hi", "new_str": "hey"}, "t2"),
            # The October 2026 names.
            tool_use("Edit", {"file_path": path, "old_string": "hey", "new_string": "yo"}, "t3"),
            tool_use("Write", {"file_path": "/home/claude/notes.md", "content": "# Notes"}, "t4"),
            tool_use("Edit", {"file_path": "/uploads/unknown.txt", "old_string": "a"}, "t5"),
        ]
    )
    diagnostics = Diagnostics()
    conversation = _parse(b.build(), diagnostics)
    documents = [b for b in _blocks(conversation) if isinstance(b, DocumentBlock)]
    assert [(d.title, d.content, d.version) for d in documents] == [
        ("hello.py", "print('hi')\n", 1),
        ("hello.py", "print('hey')\n", 2),
        ("hello.py", "print('yo')\n", 3),
        ("notes.md", "# Notes", 1),
    ]
    assert documents[0].doc_type == "code/python"
    assert documents[3].doc_type == "document"
    assert diagnostics.counters["claude_edits_of_unknown_files"] == 1
    files = document_files(conversation)
    assert set(files) == {
        f"artifacts/hello ({b.uuid[:8]}).py",
        f"artifacts/notes ({b.uuid[:8]}).md",
    }
    assert files[f"artifacts/hello ({b.uuid[:8]}).py"].content == "print('yo')\n"


def test_presented_files() -> None:
    b = ClaudeBuilder()
    path = "/mnt/user-data/outputs/report.md"
    b.assistant(
        [
            tool_use("create_file", {"path": path, "file_text": "x"}, "t1"),
            tool_use("present_files", {"filepaths": [path]}, "t2"),
            tool_result(
                "present_files",
                [
                    {"type": "local_resource", "file_path": path, "name": "report.md"},
                    {
                        "type": "local_resource",
                        "file_path": "/mnt/user-data/outputs/deck.pptx",
                        "name": "deck.pptx",
                        "mime_type": "application/vnd.ms-powerpoint",
                    },
                ],
                "t2",
            ),
        ]
    )
    blocks = _blocks(_parse(b.build()))
    assert isinstance(blocks[0], DocumentBlock)
    (missing,) = [b for b in blocks if isinstance(b, AttachmentBlock)]
    assert missing.attachment.name == "deck.pptx"


def test_shell_repl_and_results_with_empty_names() -> None:
    b = ClaudeBuilder()
    b.assistant(
        [
            tool_use("Bash", {"command": "ls"}, "t1"),
            tool_result("", [{"type": "text", "text": "a.txt"}], "t1"),  # renamed: no name
            tool_use("repl", {"code": "1+1"}, "t2"),
            tool_result("repl", [{"type": "text", "text": "boom"}], "t2", error=True),
        ]
    )
    blocks = _blocks(_parse(b.build()))
    assert blocks == [
        CodeBlock("ls", "bash", "bash_tool"),
        OutputBlock("a.txt", "bash_tool"),
        CodeBlock("1+1", "javascript", "repl", "Analysis (javascript)"),
        OutputBlock("boom", "repl", "Error"),
    ]


def test_mcp_tools_are_shown_and_plumbing_is_counted() -> None:
    b = ClaudeBuilder()
    b.assistant(
        [
            tool_use("GitHub:create_issue", {"title": "Bug"}, "t1"),
            tool_result("GitHub:create_issue", [{"type": "text", "text": "#12"}], "t1"),
            tool_use("mcp__linear__list_issues", {}, "t2"),
            tool_use("web_fetch", {"url": "https://example.com"}, "t3"),
            tool_use("memory_user_edits", {"command": "add", "control": "x"}, "t4"),
            tool_use("memory_user_edits", {"command": "view"}, "t5"),
            tool_use("launch_extended_search_task", {"command": "q"}, "t6"),
        ]
    )
    diagnostics = Diagnostics()
    blocks = _blocks(_parse(b.build(), diagnostics))
    assert blocks[0] == CodeBlock('{\n  "title": "Bug"\n}', "json", "GitHub:create_issue",
                                  "create_issue (GitHub)")  # fmt: skip
    assert blocks[1] == OutputBlock("#12", "GitHub:create_issue")
    assert blocks[2].title == "list_issues (linear)"
    assert blocks[3:] == [NoticeBlock("Memory updated"), NoticeBlock("Started research")]
    assert diagnostics.counters["tool_call:web_fetch"] == 1


def test_widget() -> None:
    b = ClaudeBuilder()
    b.assistant([tool_use("visualize:show_widget", {"title": "Chart", "widget_code": "<svg/>"})])
    (block,) = _blocks(_parse(b.build()))
    assert (block.kind, block.doc_type, block.title) == ("Widget", "code/svg", "Chart")


def test_attachments_and_files() -> None:
    b = ClaudeBuilder()
    b.human(
        "see attached",
        attachments=[
            {
                "file_name": "notes.txt",
                "file_size": 5,
                "file_type": "txt",
                "extracted_content": "hi",
            },
            {"file_name": "", "file_type": "txt", "extracted_content": "pasted"},
        ],
        files=[{"file_name": "notes.txt", "file_uuid": "f1"}, {"file_name": "cat.png"}],
    )
    diagnostics = Diagnostics()
    blocks = _blocks(_parse(b.build(), diagnostics), "user")
    assert blocks[:2] == [
        FileTextBlock("hi", "notes.txt", "txt"),
        FileTextBlock("pasted", None, "txt"),
    ]
    image = blocks[2]
    assert isinstance(image, AttachmentBlock)
    assert (image.attachment.name, image.attachment.kind) == ("cat.png", "image")
    assert isinstance(blocks[3], TextBlock)
    assert diagnostics.counters["claude_files_without_payload"] == 1


def test_render_uses_claude_labels_and_shows_files() -> None:
    b = ClaudeBuilder(name="Files")
    b.human("read", attachments=[{"file_name": "a.py", "extracted_content": "x = 1"}])
    b.assistant("ok")
    note = render_conversation(_parse(b.build()), RenderOptions(), lambda _: None)
    assert 'source: "claude"' in note.text
    assert '  - "claude"' in note.text
    assert "Claude" in note.text
    assert "📎 a.py" in note.text
    assert "```python\nx = 1\n```" in note.text
    custom = RenderOptions(assistant_label="Bot", tags=("ai",))
    text = render_conversation(_parse(b.build()), custom, lambda _: None).text
    assert "Bot" in text
    assert '  - "ai"' in text
    assert '  - "claude"' not in text


def _source(tmp_path: Path, members: dict[str, Any]) -> ExportSource:
    return ExportSource.open(write_zip(tmp_path / "x.zip", members), tmp_path / "work")


def test_projects_in_both_layouts(tmp_path: Path) -> None:
    starter = claude_project("How to use Claude", uuid="s", is_starter_project=True)
    legacy = {"projects.json": [claude_project(), starter], "conversations.json": []}
    diagnostics = Diagnostics()
    (project,) = read_projects(_source(tmp_path, legacy), diagnostics)
    assert diagnostics.counters["claude_starter_projects_skipped"] == 1
    assert project.name == "Garden"
    assert project.instructions == "Answer as a # gardener."
    assert project.docs[0].name == "soil.pdf"
    assert project.url == f"https://claude.ai/project/{project.id}"
    split = {"projects/a.json": claude_project(), "projects/b.json": claude_project(uuid="2")}
    assert len(read_projects(_source(tmp_path, split), Diagnostics())) == 2


def test_memory_in_both_layouts(tmp_path: Path) -> None:
    legacy = {"memories.json": [claude_memory(memory_files=None)], "conversations.json": []}
    memory = read_memory(_source(tmp_path, legacy), Diagnostics())
    assert memory is not None
    assert memory.summary.endswith("Grows tomatoes.")
    assert memory.files == ()
    current = read_memory(_source(tmp_path, {"memories/acc.json": claude_memory()}), Diagnostics())
    assert current is not None
    assert current.files[0].path == "/areas/garden.md"
    empty = {"memories/acc.json": {"conversations_memory": "", "account_uuid": "x"}}
    assert read_memory(_source(tmp_path, empty), Diagnostics()) is None


def test_project_pages(tmp_path: Path) -> None:
    source = _source(tmp_path, {"projects/a.json": claude_project()})
    (project,) = read_projects(source, Diagnostics())
    twin = replace(project, id="ffffffff-0000")
    paths = project_paths([project, twin])
    assert set(paths) == {"projects/Garden (b0b0b0b0).md", "projects/Garden (ffffffff).md"}
    docs = project_doc_paths("projects/Garden.md", replace(project, docs=project.docs * 2))
    assert docs == ["projects/Garden/soil.pdf.md", "projects/Garden/soil.pdf (2).md"]
    text = render_project(project, docs[:1], "projects/Garden.md", RenderOptions())
    assert 'project_id: "b0b0b0b0-0000-4000-8000-000000000001"' in text
    assert '  - "claude"' in text
    assert "## Instructions\n\nAnswer as a # gardener." in text
    assert "- [soil.pdf](Garden/soil.pdf.md) · added 2026-01-01" in text


def test_memory_page(tmp_path: Path) -> None:
    memory = read_memory(_source(tmp_path, {"memories/acc.json": claude_memory()}), Diagnostics())
    assert memory is not None
    text = render_memory(memory, UTC)
    assert text.startswith("# Claude memory\n")
    assert "## /areas/garden.md\n\n*Updated 2026-01-02*\n\n### Garden\nRaised beds." in text
    assert "title: garden" not in text
