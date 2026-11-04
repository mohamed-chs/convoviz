"""Conversation tree reconstruction and linearization.

ChatGPT stores a conversation as a tree: every edit or regeneration starts a new
branch. Since 2026, exported nodes only carry a ``parent`` pointer (``children`` is
gone), so the tree is always rebuilt from parent links here. Legacy ``children``
lists are honored only as extra hints.

A :class:`Thread` is the linear reading order of one branch, with
:class:`Fork` markers wherever other branches split off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from convoviz.model import Conversation, Message, Node

_EPOCH = datetime.min.replace(tzinfo=UTC)


@dataclass(slots=True)
class Fork:
    """Alternative branches that diverge at this point of a thread."""

    alternatives: list[Thread]


@dataclass(slots=True)
class Thread:
    """Messages of one branch in reading order, interleaved with forks."""

    items: list[Message | Fork] = field(default_factory=list)

    def messages(self) -> list[Message]:
        """Only the messages of this branch (forks excluded)."""
        return [i for i in self.items if not isinstance(i, Fork)]

    def message_count(self) -> int:
        """Number of messages on this branch."""
        return sum(1 for i in self.items if not isinstance(i, Fork))


@dataclass(slots=True)
class Tree:
    """Navigable view over a conversation's nodes."""

    nodes: dict[str, Node]
    children: dict[str | None, list[str]]  # None key holds the roots

    @classmethod
    def build(cls, conversation: Conversation) -> Tree:
        """Build parent→children links from every available signal."""
        nodes = conversation.nodes
        children: dict[str | None, list[str]] = {}
        seen_edges: set[tuple[str | None, str]] = set()

        def link(parent: str | None, child: str) -> None:
            if (parent, child) in seen_edges or parent == child:
                return
            seen_edges.add((parent, child))
            children.setdefault(parent, []).append(child)

        for node in nodes.values():
            parent = node.parent if node.parent in nodes else None
            link(parent, node.id)
        # Legacy hints: only for children whose own parent pointer is missing.
        for node in nodes.values():
            for child_id in node.children_hint:
                child = nodes.get(child_id)
                if child is not None and child.parent not in nodes:
                    children[None] = [c for c in children.get(None, []) if c != child_id]
                    seen_edges.discard((None, child_id))
                    link(node.id, child_id)

        tree = cls(nodes, children)
        for key, ids in children.items():
            children[key] = sorted(ids, key=tree._sort_key)
        return tree

    def _sort_key(self, node_id: str) -> tuple[datetime, str]:
        node = self.nodes[node_id]
        return (node.time or _EPOCH, node_id)

    def path_to(self, leaf: str) -> list[str]:
        """Node ids from the root down to ``leaf`` (cycle-safe)."""
        path: list[str] = []
        seen: set[str] = set()
        current: str | None = leaf
        while current is not None and current in self.nodes and current not in seen:
            seen.add(current)
            path.append(current)
            current = self.nodes[current].parent
        path.reverse()
        return path

    def latest_leaf(self, start: str | None) -> str | None:
        """Follow the most recent child at every step, starting from ``start``."""
        current = start
        seen: set[str | None] = set()
        while current not in seen:
            seen.add(current)
            kids = self.children.get(current)
            if not kids:
                return current
            current = max(kids, key=self._subtree_latest)
        return current

    def _subtree_latest(self, node_id: str) -> tuple[datetime, str]:
        latest = self._sort_key(node_id)
        stack = [node_id]
        seen: set[str] = set()
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            latest = max(latest, self._sort_key(nid))
            stack.extend(self.children.get(nid, ()))
        return latest

    def active_path(self, current_node: str | None) -> list[str]:
        """The branch the user last saw (falls back to the most recent leaf)."""
        if current_node and current_node in self.nodes:
            return self.path_to(current_node)
        leaf = self.latest_leaf(None)
        return self.path_to(leaf) if leaf else []

    def thread(self, path: list[str], *, include_forks: bool) -> Thread:
        """Linearize ``path``; optionally attach alternative branches as forks."""
        return self._thread(path, include_forks=include_forks, visited=set())

    def _thread(self, path: list[str], *, include_forks: bool, visited: set[str]) -> Thread:
        thread = Thread()
        visited.update(path)
        parent: str | None = None
        for node_id in path:
            if include_forks:
                alternatives = []
                for sibling in self.children.get(parent, ()):
                    if sibling == node_id or sibling in visited:
                        continue
                    leaf = self.latest_leaf(sibling)
                    branch_path = self.path_to(leaf) if leaf else [sibling]
                    branch_path = branch_path[branch_path.index(sibling) :]
                    alt = self._thread(branch_path, include_forks=include_forks, visited=visited)
                    if alt.message_count():
                        alternatives.append(alt)
                if alternatives:
                    thread.items.append(Fork(alternatives))
            message = self.nodes[node_id].message
            if message is not None:
                thread.items.append(message)
            parent = node_id
        return thread


def linearize(conversation: Conversation, *, include_forks: bool) -> Thread:
    """Reading order of a conversation's active branch."""
    tree = Tree.build(conversation)
    return tree.thread(tree.active_path(conversation.current_node), include_forks=include_forks)


def all_messages(conversation: Conversation) -> list[Message]:
    """Every visible message on every branch (for statistics)."""
    return [n.message for n in conversation.nodes.values() if n.message is not None]
