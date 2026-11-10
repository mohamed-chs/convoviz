"""Build synthetic ChatGPT exports for tests (no real user data, ever)."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

BASE_TIME = 1_767_225_600.0  # 2026-01-01T00:00:00Z


@dataclass
class ConversationBuilder:
    """Fluent builder for one raw conversation in the 2026 shape (parent links only)."""

    title: str = "Test conversation"
    conversation_id: str = "6a1b2c3d-0000-4000-8000-000000000001"
    create_time: float = BASE_TIME
    legacy_children: bool = False
    mapping: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_node: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    _counter: int = 0
    _last: str | None = None

    def __post_init__(self) -> None:
        self.mapping["root"] = {"id": "root", "message": None, "parent": None}
        self._last = "root"

    def add(
        self,
        role: str,
        content: dict[str, Any] | str,
        *,
        parent: str | None = None,
        metadata: dict[str, Any] | None = None,
        recipient: str | None = "all",
        name: str | None = None,
        node_id: str | None = None,
        time: float | None = None,
    ) -> str:
        """Append a message; returns its node id. ``parent`` defaults to the last node."""
        self._counter += 1
        node_id = node_id or f"n{self._counter}"
        if isinstance(content, str):
            content = {"content_type": "text", "parts": [content]}
        author: dict[str, Any] = {"role": role}
        if name:
            author["name"] = name
        self.mapping[node_id] = {
            "id": node_id,
            "parent": parent or self._last,
            "message": {
                "id": node_id,
                "author": author,
                "create_time": time if time is not None else self.create_time + self._counter * 60,
                "content": content,
                "metadata": metadata or {},
            },
        }
        if recipient is not None:  # the 2026 export leaves it out
            self.mapping[node_id]["message"]["recipient"] = recipient
        self._last = node_id
        self.current_node = node_id
        return node_id

    def user(self, text: str, **kwargs: Any) -> str:
        """Add a user text message."""
        return self.add("user", text, **kwargs)

    def assistant(self, text: str, *, model: str = "gpt-5-2", **kwargs: Any) -> str:
        """Add an assistant text message."""
        metadata = {"model_slug": model, **kwargs.pop("metadata", {})}
        return self.add("assistant", text, metadata=metadata, **kwargs)

    def build(self) -> dict[str, Any]:
        """The raw conversation dict."""
        mapping = json.loads(json.dumps(self.mapping))
        if self.legacy_children:
            for node in mapping.values():
                node["children"] = [k for k, v in mapping.items() if v.get("parent") == node["id"]]
        times = [
            t
            for n in mapping.values()
            if isinstance(n.get("message"), dict)
            and isinstance(t := n["message"].get("create_time"), int | float)
        ]
        return {
            "title": self.title,
            "create_time": self.create_time,
            "update_time": max(times, default=self.create_time),
            "mapping": mapping,
            "current_node": self.current_node,
            "conversation_id": self.conversation_id,
            "id": self.conversation_id,
            "default_model_slug": "gpt-5-2",
            **self.extra,
        }


def write_export(
    path: Path,
    conversations: list[Any],
    *,
    files: dict[str, bytes] | None = None,
    sharded: int = 0,
    as_zip: bool = True,
    extra_json: dict[str, Any] | None = None,
) -> Path:
    """Write an export as a ZIP (default) or folder. ``sharded`` = conversations per shard."""
    members: dict[str, bytes] = {}
    if sharded:
        shards = [conversations[i : i + sharded] for i in range(0, len(conversations), sharded)]
        names = [f"conversations-{i:03d}.json" for i in range(len(shards))]
        for name, shard in zip(names, shards, strict=True):
            members[name] = json.dumps(shard).encode()
        manifest = {
            "version": 1,
            "logical_files": {
                "conversations.json": {
                    "files": names,
                    "sharded": True,
                    "shard_count": len(names),
                }
            },
        }
        members["export_manifest.json"] = json.dumps(manifest).encode()
    else:
        members["conversations.json"] = json.dumps(conversations).encode()
    members["chat.html"] = b"<html></html>"
    for name, data in (extra_json or {}).items():
        members[name] = json.dumps(data).encode()
    members.update(files or {})
    if as_zip:
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
    else:
        path.mkdir(parents=True, exist_ok=True)
        for name, data in members.items():
            (path / name).parent.mkdir(parents=True, exist_ok=True)
            (path / name).write_bytes(data)
    return path


PNG_1PX = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00"
    b"\x00\x00\x00IEND\xaeB`\x82"
)


def thinking_turn(builder: ConversationBuilder) -> None:
    """A thinking turn as the 2026 export has it: no recipients, no preamble flags."""
    builder.user("q", recipient=None)
    thoughts = {"content_type": "thoughts", "thoughts": [{"summary": "Plan", "content": "Do it"}]}
    builder.add("assistant", thoughts, recipient=None)
    builder.assistant("Checking the docs.", recipient=None)
    builder.add("assistant", thoughts, recipient=None)
    builder.add(
        "assistant",
        {"content_type": "reasoning_recap", "content": "Thought for 9s"},
        recipient=None,
    )
    builder.assistant("The answer.", recipient=None)
