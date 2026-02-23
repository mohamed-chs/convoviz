"""Conversation model - pure data class.

Object path: conversations.json -> conversation (one of the list items)
"""

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, PrivateAttr, field_validator

from convoviz.models.message import AuthorRole
from convoviz.models.node import Node, build_node_tree, node_sort_key
from convoviz.utils import month_start, year_start


class Conversation(BaseModel):
    """A single ChatGPT conversation.

    This is a pure data model - rendering and I/O logic are in separate modules.
    """

    title: str = ""
    create_time: datetime

    @field_validator("title", mode="before")
    @classmethod
    def _coerce_null_title(cls, v: Any) -> str:
        return v if v is not None else ""

    update_time: datetime
    mapping: dict[str, Node]
    current_node: str
    is_starred: bool | None = None
    voice: str | dict[str, Any] | None = None
    plugin_ids: list[str] | None = None
    conversation_id: str
    _node_mapping_cache: dict[str, Node] | None = PrivateAttr(default=None)

    @property
    def node_mapping(self) -> dict[str, Node]:
        """Get the connected node tree."""
        if self._node_mapping_cache is None:
            self._node_mapping_cache = build_node_tree(self.mapping)
        return self._node_mapping_cache

    @property
    def ordered_nodes(self) -> list[Node]:
        """Get the nodes in the current active branch in chronological order."""
        nodes = []
        mapping = self.node_mapping
        current = mapping.get(self.current_node)
        seen: set[str] = set()
        while current and current.id not in seen:
            seen.add(current.id)
            nodes.append(current)
            current = current.parent_node
        return list(reversed(nodes))

    @property
    def all_message_nodes(self) -> list[Node]:
        """Get all nodes that have messages (including hidden/internal ones)."""
        return [node for node in self.node_mapping.values() if node.has_message]

    @property
    def visible_message_nodes(self) -> list[Node]:
        """Get all nodes that have *visible* (non-hidden) messages."""
        return [
            node
            for node in self.node_mapping.values()
            if (message := node.message) is not None and not message.is_hidden
        ]

    def _sorted_message_nodes(self, *, include_hidden: bool) -> list[Node]:
        """Return message nodes sorted by create_time, then node id."""
        nodes = self.all_message_nodes if include_hidden else self.visible_message_nodes
        return sorted(nodes, key=node_sort_key)

    def nodes_by_author(
        self, *authors: AuthorRole, include_hidden: bool = False
    ) -> list[Node]:
        """Get nodes with messages from specified authors.

        Args:
            *authors: Author roles to filter by. Defaults to ("user",) if empty.
            include_hidden: Whether to include hidden/internal messages.

        """
        if not authors:
            authors = ("user",)
        nodes = self.all_message_nodes if include_hidden else self.visible_message_nodes
        return [
            node
            for node in nodes
            if node.message and node.message.author.role in authors
        ]

    @property
    def url(self) -> str:
        """Get the ChatGPT URL for this conversation."""
        return f"https://chatgpt.com/c/{self.conversation_id}"

    @property
    def content_types(self) -> list[str]:
        """Get all unique content types in the conversation.

        Excludes hidden messages.
        """
        return sorted(
            {
                node.message.content.content_type
                for node in self.visible_message_nodes
                if node.message
            }
        )

    def message_count(self, *authors: AuthorRole) -> int:
        """Count messages from specified authors."""
        return len(self.nodes_by_author(*authors))

    @property
    def model(self) -> str | None:
        """Get the ChatGPT model used for this conversation."""
        for node in self._sorted_message_nodes(include_hidden=True):
            message = node.message
            if (
                message
                and message.author.role == "assistant"
                and message.metadata.model_slug
            ):
                return message.metadata.model_slug
        return None

    @property
    def plugins(self) -> list[str]:
        """Get all plugins used in this conversation."""
        plugins = set()
        for node in self._sorted_message_nodes(include_hidden=True):
            message = node.message
            if not message or message.author.role != "tool":
                continue
            invoked = message.metadata.invoked_plugin
            if not isinstance(invoked, dict):
                continue
            namespace = invoked.get("namespace")
            if namespace:
                plugins.add(namespace)
        return sorted(plugins)

    @property
    def custom_instructions(self) -> dict[str, str]:
        """Get custom instructions used for this conversation."""
        # Custom instructions are often hidden system messages,
        # so we must include hidden nodes.
        system_nodes = self.nodes_by_author("system", include_hidden=True)
        for node in system_nodes:
            context_message = node.message
            if context_message and context_message.metadata.is_user_system_message:
                return context_message.metadata.user_context_message_data or {}
        return {}

    @property
    def canvas_documents(self) -> list[dict[str, Any]]:
        """Get all Canvas documents created in this conversation's active branch.

        Returns:
            List of dicts with {name, type, content,
            conversation_id, conversation_title}

        """
        docs: list[dict[str, Any]] = []
        for node in self.ordered_nodes:
            if not node.message:
                continue
            doc = node.message.canvas_document
            if doc:
                docs.append(
                    {
                        **doc,
                        "conversation_id": self.conversation_id,
                        "conversation_title": self.title or "Untitled",
                    }
                )
        return docs

    def timestamps(self, *authors: AuthorRole) -> list[float]:
        """Get message timestamps from specified authors.

        Useful for generating time-based visualizations.
        """
        if not authors:
            authors = ("user",)
        return [
            node.message.create_time.timestamp()
            for node in self.nodes_by_author(*authors)
            if node.message and node.message.create_time
        ]

    def plaintext(self, *authors: AuthorRole) -> str:
        """Get concatenated plain text from specified authors.

        Useful for word cloud generation.
        """
        if not authors:
            authors = ("user",)
        return "\n".join(
            node.message.text
            for node in self.nodes_by_author(*authors)
            if node.message and node.message.has_content
        )

    @property
    def week_start(self) -> datetime:
        """Get the Monday of the week this conversation was created."""
        start_of_week = self.create_time - timedelta(days=self.create_time.weekday())
        return start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)

    @property
    def month_start(self) -> datetime:
        """Get the first day of the month this conversation was created."""
        return month_start(self.create_time)

    @property
    def year_start(self) -> datetime:
        """Get January 1st of the year this conversation was created."""
        return year_start(self.create_time)

    @property
    def citation_map(self) -> dict[str, dict[str, str | None]]:
        """Aggregate citation metadata from all messages in the conversation.

        Traverses all nodes (including hidden ones) to collect embedded
        citation definitions from tool outputs (e.g. search results).
        """
        aggregated_map = {}
        for node in self.all_message_nodes:
            if not node.message:
                continue
            # Extract citations from message parts
            aggregated_map.update(node.message.internal_citation_map)
        return aggregated_map
