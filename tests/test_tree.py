from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from convoviz.model import Conversation, Message, Node, TextBlock
from convoviz.tree import Tree, linearize

if TYPE_CHECKING:
    from collections.abc import Sequence

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _conv(
    edges: Sequence[tuple[str, str | None]], current: str | None, **hints: tuple[str, ...]
) -> Conversation:
    nodes = {}
    for index, (node_id, parent) in enumerate(edges):
        time = T0 + timedelta(minutes=index)
        message = (
            None if node_id == "root" else Message(node_id, "user", [TextBlock(node_id)], time)
        )
        nodes[node_id] = Node(node_id, parent, message, time, hints.get(node_id, ()))
    return Conversation("c", "t", T0, T0, nodes, current)


def _ids(items: list) -> list:
    return [
        i.id
        if isinstance(i, Message)
        else [[m.id for m in alt.messages()] for alt in i.alternatives]
        for i in items
    ]


def test_parent_links_alone_rebuild_the_tree() -> None:
    conv = _conv([("root", None), ("a", "root"), ("b", "a"), ("c", "b")], "c")
    assert [m.id for m in linearize(conv, include_forks=False).messages()] == ["a", "b", "c"]


def test_active_branch_and_forks() -> None:
    edges = [("root", None), ("q", "root"), ("a1", "q"), ("a2", "q"), ("f1", "a1"), ("f2", "a2")]
    conv = _conv(edges, "f1")
    active = linearize(conv, include_forks=False)
    assert [m.id for m in active.messages()] == ["q", "a1", "f1"]
    with_forks = linearize(conv, include_forks=True)
    assert _ids(with_forks.items) == ["q", [["a2", "f2"]], "a1", "f1"]


def test_missing_current_node_falls_back_to_latest_leaf() -> None:
    edges = [("root", None), ("q", "root"), ("old", "q"), ("new", "q")]
    conv = _conv(edges, "gone")
    assert [m.id for m in linearize(conv, include_forks=False).messages()] == ["q", "new"]


def test_legacy_children_hints_fill_missing_parents() -> None:
    conv = _conv([("root", None), ("a", None), ("b", None)], "b", root=("a",), a=("b",))
    tree = Tree.build(conv)
    assert tree.path_to("b") == ["b"]  # parent pointers are missing...
    assert tree.children["root"] == ["a"]  # ...but hints recover the structure
    assert tree.children["a"] == ["b"]


def test_cycles_do_not_hang() -> None:
    conv = _conv([("a", "b"), ("b", "a")], "a")
    thread = linearize(conv, include_forks=True)
    assert len(thread.messages()) <= 2
