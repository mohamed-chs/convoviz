"""Projects and memory from a claude.ai export.

Projects come as ``projects.json`` (an array, before mid-2026) or one
``projects/<uuid>.json`` per project. Memory comes as ``memories.json`` (a
one-element array, before the August 2026 export) or ``memories/<account>.json``
(an object): ``conversations_memory`` text and, since September 2026,
``memory_files``. Both are optional and read the same tolerant way as
conversations. See ``docs/dev/claude-export-format.md``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from convoviz.export import load_json
from convoviz.model import Memory, MemoryFile, Project, ProjectDoc
from convoviz.values import as_str, as_time

if TYPE_CHECKING:
    from collections.abc import Iterator

    from convoviz.diagnostics import Diagnostics
    from convoviz.source import ExportSource, Member

_PROJECTS_RE = re.compile(r"(?i)(?:^|/)projects(?:\.json|/[^/]+\.json)$")
_MEMORIES_RE = re.compile(r"(?i)(?:^|/)memories(?:\.json|/[^/]+\.json)$")


def has_extras(source: ExportSource) -> bool:
    """Whether ``source`` holds Claude projects or memory."""
    return any(_PROJECTS_RE.search(path) or _MEMORIES_RE.search(path) for path in _paths(source))


def read_projects(source: ExportSource, diagnostics: Diagnostics) -> list[Project]:
    """Every project in ``source``, except claude.ai's built-in example project."""
    projects: list[Project] = []
    for item in _objects(source, _PROJECTS_RE, diagnostics):
        project = _project(item, diagnostics)
        if project is not None:
            projects.append(project)
    return projects


def read_memory(source: ExportSource, diagnostics: Diagnostics) -> Memory | None:
    """The account's memory, or None if the export has none (or it is empty)."""
    for item in _objects(source, _MEMORIES_RE, diagnostics):
        summary = (as_str(item.get("conversations_memory")) or "").strip()
        files = tuple(
            MemoryFile(path, content, as_time(entry.get("updated_at")))
            for entry in item.get("memory_files") or ()
            if isinstance(entry, dict)
            and (path := as_str(entry.get("path")))
            and (content := as_str(entry.get("content")) or "").strip()
        )
        if summary or files:
            return Memory(summary, files)
    return None


def _objects(source: ExportSource, pattern: re.Pattern[str], diagnostics: Diagnostics) -> list[Any]:
    root = source.root_prefix
    members: list[Member] = sorted((m for m in source if pattern.search(root + m.path)), key=_path)
    out: list[Any] = []
    for member in members:
        data = load_json(member, diagnostics)
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict):
                out.append(item)
            elif item is not None:
                diagnostics.skip(f"{member.path}: an entry is not an object")
    return out


def _path(member: Member) -> str:
    return member.path


def _paths(source: ExportSource) -> Iterator[str]:
    """Member paths as they are in the archive (a ZIP's single top folder kept)."""
    return (source.root_prefix + m.path for m in source)


def _project(item: dict[str, Any], diagnostics: Diagnostics) -> Project | None:
    project_id = as_str(item.get("uuid"))
    name = (as_str(item.get("name")) or "").strip()
    if not project_id:
        diagnostics.skip("a project without an id")
        return None
    if item.get("is_starter_project") is True:
        diagnostics.count("claude_starter_projects_skipped")
        return None
    docs = tuple(
        ProjectDoc(
            (as_str(doc.get("filename")) or "").strip() or "Untitled document",
            content,
            as_time(doc.get("created_at")),
        )
        for doc in item.get("docs") or ()
        if isinstance(doc, dict) and (content := as_str(doc.get("content")) or "").strip()
    )
    return Project(
        id=project_id,
        name=name or "Untitled project",
        description=(as_str(item.get("description")) or "").strip(),
        instructions=(as_str(item.get("prompt_template")) or "").strip(),
        created=as_time(item.get("created_at")),
        updated=as_time(item.get("updated_at")),
        docs=docs,
        url=f"https://claude.ai/project/{project_id}",
    )
