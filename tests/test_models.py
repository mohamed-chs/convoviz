"""Tests for the models."""

import copy
from datetime import UTC, datetime, timedelta

from convoviz.models import Conversation
from convoviz.models.collection import ConversationCollection
from convoviz.models.message import (
    Message,
    MessageAuthor,
    MessageContent,
    MessageMetadata,
)
from convoviz.models.node import Node, build_node_tree


def test_url(mock_conversation: Conversation) -> None:
    """Test url property."""
    assert mock_conversation.url == "https://chatgpt.com/c/conversation_111"


def test_content_types(mock_conversation: Conversation) -> None:
    """Test content_types property."""
    assert set(mock_conversation.content_types) == {"text"}


def test_message_count(mock_conversation: Conversation) -> None:
    """Test message_count method."""
    assert mock_conversation.message_count("user", "assistant") == 2
    assert mock_conversation.message_count("user") == 1
    assert mock_conversation.message_count("assistant") == 1
    # Internal system messages are hidden by default and should not be counted.
    assert mock_conversation.message_count("system") == 0


def test_plaintext(mock_conversation: Conversation) -> None:
    """Test plaintext method."""
    assert mock_conversation.plaintext("user") == "user message 111"
    assert mock_conversation.plaintext("assistant") == "assistant message 111"
    # Internal system messages are hidden by default and should not contribute plaintext.
    assert mock_conversation.plaintext("system") == ""


def test_model(mock_conversation: Conversation) -> None:
    """Test model property."""
    assert mock_conversation.model == "gpt-4"


def test_plugins(mock_conversation: Conversation) -> None:
    """Test plugins property."""
    assert len(mock_conversation.plugins) == 0


def test_plugins_sorted_and_safe_namespace() -> None:
    """Test plugin namespaces are sorted and missing namespaces are ignored."""
    ts = datetime(2024, 1, 1, tzinfo=UTC).timestamp()
    conv = Conversation(
        title="Plugin Test",
        create_time=ts,
        update_time=ts,
        mapping={
            "root": {
                "id": "root",
                "message": None,
                "parent": None,
                "children": ["tool_a"],
            },
            "tool_a": {
                "id": "tool_a",
                "message": {
                    "id": "tool_a",
                    "author": {"role": "tool", "metadata": {}},
                    "create_time": ts,
                    "update_time": ts,
                    "content": {"content_type": "text", "parts": ["a"]},
                    "status": "finished_successfully",
                    "end_turn": True,
                    "weight": 1.0,
                    "metadata": {"invoked_plugin": {"namespace": "zeta"}},
                    "recipient": "all",
                },
                "parent": "root",
                "children": ["tool_b"],
            },
            "tool_b": {
                "id": "tool_b",
                "message": {
                    "id": "tool_b",
                    "author": {"role": "tool", "metadata": {}},
                    "create_time": ts + 1,
                    "update_time": ts + 1,
                    "content": {"content_type": "text", "parts": ["b"]},
                    "status": "finished_successfully",
                    "end_turn": True,
                    "weight": 1.0,
                    "metadata": {"invoked_plugin": {"namespace": "alpha"}},
                    "recipient": "all",
                },
                "parent": "tool_a",
                "children": ["tool_missing"],
            },
            "tool_missing": {
                "id": "tool_missing",
                "message": {
                    "id": "tool_missing",
                    "author": {"role": "tool", "metadata": {}},
                    "create_time": ts + 2,
                    "update_time": ts + 2,
                    "content": {"content_type": "text", "parts": ["c"]},
                    "status": "finished_successfully",
                    "end_turn": True,
                    "weight": 1.0,
                    "metadata": {"invoked_plugin": {"name": "no_namespace"}},
                    "recipient": "all",
                },
                "parent": "tool_b",
                "children": [],
            },
        },
        current_node="tool_missing",
        conversation_id="plugins_conv",
    )

    assert conv.plugins == ["alpha", "zeta"]


def test_internal_citation_map_ignores_non_dict_ref_id() -> None:
    """Test that malformed citation entries are ignored safely."""
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    msg = Message(
        id="msg",
        author=MessageAuthor(role="assistant"),
        content=MessageContent(
            content_type="text",
            parts=[{"type": "search_result", "ref_id": "turn0search1"}],
        ),
        metadata=MessageMetadata(),
        create_time=ts,
        update_time=ts,
        status="finished_successfully",
        end_turn=True,
        weight=1.0,
        recipient="all",
    )
    assert msg.internal_citation_map == {}


def test_message_missing_status_and_weight_uses_defaults() -> None:
    """Test that messages without status/weight validate (July 2026 exports)."""
    msg = Message.model_validate(
        {
            "id": "msg",
            "author": {"name": None, "role": "tool", "metadata": {}},
            "content": {"content_type": "text", "parts": ["hello"]},
            "metadata": {},
        }
    )
    assert msg.status == "finished_successfully"
    assert msg.weight == 1.0


def test_conversation_citation_map_aggregates_metadata_groups() -> None:
    """Test that citation_map aggregates from metadata.search_result_groups."""
    ts = datetime(2024, 1, 1, tzinfo=UTC).timestamp()
    conv = Conversation(
        title="Citations",
        create_time=ts,
        update_time=ts,
        mapping={
            "root": {
                "id": "root",
                "message": None,
                "parent": None,
                "children": ["tool_node"],
            },
            "tool_node": {
                "id": "tool_node",
                "message": {
                    "id": "tool_node",
                    "author": {"role": "tool", "metadata": {}},
                    "create_time": ts,
                    "update_time": ts,
                    "content": {"content_type": "text", "parts": []},
                    "status": "finished_successfully",
                    "end_turn": True,
                    "weight": 1.0,
                    "metadata": {
                        "search_result_groups": [
                            {
                                "entries": [
                                    {
                                        "ref_id": {
                                            "ref_type": "search",
                                            "turn_index": 0,
                                            "ref_index": 1,
                                        },
                                        "title": "Source A",
                                        "url": "http://example.com/a",
                                    },
                                ],
                            },
                        ],
                    },
                    "recipient": "all",
                },
                "parent": "root",
                "children": [],
            },
        },
        current_node="tool_node",
        conversation_id="cite_conv",
    )

    citation_map = conv.citation_map
    assert citation_map["turn0search1"]["title"] == "Source A"
    assert citation_map["turn0search1"]["url"] == "http://example.com/a"


def test_timestamps(mock_conversation: Conversation) -> None:
    """Test timestamps method."""
    timestamps = mock_conversation.timestamps("user")
    assert len(timestamps) == 1


def test_nodes_by_author(mock_conversation: Conversation) -> None:
    """Test nodes_by_author method."""
    user_nodes = mock_conversation.nodes_by_author("user")
    assert len(user_nodes) == 1
    assert user_nodes[0].message is not None
    assert user_nodes[0].message.author.role == "user"
    # Internal system messages are hidden by default.
    assert mock_conversation.nodes_by_author("system") == []


def test_all_message_nodes(mock_conversation: Conversation) -> None:
    """Test all_message_nodes property."""
    nodes = mock_conversation.all_message_nodes
    assert len(nodes) == 3  # system, user, assistant


def test_message_visibility() -> None:
    """Test message is_empty and is_hidden properties."""
    base_data = {
        "id": "msg_id",
        "create_time": datetime.now(),
        "update_time": datetime.now(),
        "status": "finished_successfully",
        "end_turn": True,
        "weight": 1.0,
        "recipient": "all",
    }

    # Case 1: Empty message
    msg = Message(
        author=MessageAuthor(role="user"),
        content=MessageContent(content_type="text", parts=[""]),
        metadata=MessageMetadata(),
        **base_data,
    )
    assert msg.is_empty
    assert msg.is_hidden

    # Case 2: Visible user message
    msg = Message(
        author=MessageAuthor(role="user"),
        content=MessageContent(content_type="text", parts=["Hello"]),
        metadata=MessageMetadata(),
        **base_data,
    )
    assert not msg.is_empty
    assert not msg.is_hidden

    # Case 3: Internal System message (hidden)
    msg = Message(
        author=MessageAuthor(role="system"),
        content=MessageContent(content_type="text", parts=["You are ChatGPT"]),
        metadata=MessageMetadata(),
        **base_data,
    )
    assert not msg.is_empty
    assert msg.is_hidden

    # Case 4: User System message (visible)
    msg = Message(
        author=MessageAuthor(role="system"),
        content=MessageContent(content_type="text", parts=["My instructions"]),
        metadata=MessageMetadata(is_user_system_message=True),
        **base_data,
    )
    assert not msg.is_hidden

    # Case 5: Browser tool (hidden)
    msg = Message(
        author=MessageAuthor(role="tool", name="browser"),
        content=MessageContent(content_type="text", parts=["search query"]),
        metadata=MessageMetadata(),
        **base_data,
    )
    assert msg.is_hidden

    # Case 6: Browsing status (hidden)
    msg = Message(
        author=MessageAuthor(role="assistant"),
        content=MessageContent(
            content_type="tether_browsing_display",
            parts=["Browsing..."],
        ),
        metadata=MessageMetadata(),
        **base_data,
    )
    assert msg.is_hidden

    # Case 7: Assistant tool call to browser (hidden)
    msg = Message(
        author=MessageAuthor(role="assistant"),
        content=MessageContent(content_type="code", parts=["search('foo')"]),
        metadata=MessageMetadata(),
        **{**base_data, "recipient": "browser"},
    )
    assert msg.is_hidden

    # Case 8: Assistant tool call with recipient=all (hidden due to content_type=code)
    msg = Message(
        author=MessageAuthor(role="assistant"),
        content=MessageContent(content_type="code", parts=["search('foo')"]),
        metadata=MessageMetadata(),
        **{**base_data, "recipient": "all"},
    )
    assert msg.is_hidden

    # Case 9: Image-only attachment should stay visible
    msg = Message(
        author=MessageAuthor(role="user"),
        content=MessageContent(content_type="text"),
        metadata=MessageMetadata(
            attachments=[
                {"id": "file-123", "name": "pic.png", "mime_type": "image/png"},
            ],
        ),
        **base_data,
    )
    assert msg.text == ""
    assert not msg.is_empty
    assert not msg.is_hidden


def test_collection_update_merges_new_conversations_even_if_older(
    mock_conversation: Conversation,
    mock_conversation_data: dict,
) -> None:
    """A new conversation should not be skipped just because it's older than last_updated."""
    base = ConversationCollection(conversations=[mock_conversation])

    older_new_data = copy.deepcopy(mock_conversation_data)
    older_new_data["title"] = "conversation 222"
    older_new_data["conversation_id"] = "conversation_222"
    # Deliberately set update_time earlier than base.last_updated
    older_new_data["update_time"] = older_new_data["create_time"]

    other = ConversationCollection(conversations=[Conversation(**older_new_data)])
    base.update(other)

    ids = {c.conversation_id for c in base.conversations}
    assert "conversation_111" in ids
    assert "conversation_222" in ids


def test_collection_update_keeps_newest_when_ids_collide(
    mock_conversation: Conversation,
) -> None:
    """When IDs collide, the newest update_time should win."""
    base = ConversationCollection(conversations=[mock_conversation])

    updated = mock_conversation.model_copy(deep=True)
    updated.update_time = updated.update_time.replace(year=updated.update_time.year + 1)

    base.update(ConversationCollection(conversations=[updated]))
    assert (
        base.index[mock_conversation.conversation_id].update_time == updated.update_time
    )


def test_collection_update_prefers_richer_snapshot_when_timestamps_tie(
    mock_conversation: Conversation,
) -> None:
    """When update_time ties, prefer the snapshot with more nodes."""
    base = ConversationCollection(conversations=[mock_conversation])
    incoming = mock_conversation.model_copy(deep=True)

    incoming.mapping["extra_node"] = Node(
        id="extra_node",
        parent="assistant_node_111",
        children=[],
    )
    incoming.mapping["assistant_node_111"].children.append("extra_node")

    base.update(ConversationCollection(conversations=[incoming]))
    assert len(base.index[mock_conversation.conversation_id].mapping) == len(
        incoming.mapping
    )


def test_message_visibility_extended() -> None:
    """Test extended is_hidden cases from spec v2."""
    base_data = {
        "id": "msg_id",
        "create_time": datetime.now(),
        "update_time": datetime.now(),
        "status": "finished_successfully",
        "end_turn": True,
        "weight": 1.0,
        "recipient": "all",
    }

    # Case 1: is_visually_hidden_from_conversation metadata (hidden)
    msg = Message(
        author=MessageAuthor(role="assistant"),
        content=MessageContent(content_type="text", parts=["Hello"]),
        metadata=MessageMetadata(is_visually_hidden_from_conversation=True),
        **base_data,
    )
    assert msg.is_hidden

    # Case 2: Assistant targeting dalle.text2im (hidden - tool-targeted)
    msg = Message(
        author=MessageAuthor(role="assistant"),
        content=MessageContent(content_type="text", parts=["Generate image"]),
        metadata=MessageMetadata(),
        **{**base_data, "recipient": "dalle.text2im"},
    )
    assert msg.is_hidden

    # Case 3: Tool output (not browser) is visible
    msg = Message(
        author=MessageAuthor(role="tool", name="python"),
        content=MessageContent(content_type="execution_output", result="42"),
        metadata=MessageMetadata(),
        **base_data,
    )
    assert not msg.is_hidden


def test_new_content_types() -> None:
    """Test new content types from spec v2: reasoning_recap, thoughts, tether_quote."""
    base_data = {
        "id": "msg_id",
        "create_time": datetime.now(),
        "update_time": datetime.now(),
        "status": "finished_successfully",
        "end_turn": True,
        "weight": 1.0,
        "recipient": "all",
    }

    # Case 1: reasoning_recap content type (hidden by default - internal reasoning)
    msg = Message(
        author=MessageAuthor(role="assistant"),
        content=MessageContent(
            content_type="reasoning_recap",
            content="I reasoned about X",
        ),
        metadata=MessageMetadata(),
        **base_data,
    )
    assert msg.text == "I reasoned about X"
    assert msg.is_hidden  # Hidden: internal reasoning noise

    # Case 2: thoughts content type (hidden by default - internal reasoning)
    msg = Message(
        author=MessageAuthor(role="assistant"),
        content=MessageContent(
            content_type="thoughts",
            thoughts=[
                {"summary": "Thinking about the problem...", "finished": True},
                {"summary": "Considering alternatives.", "finished": True},
            ],
        ),
        metadata=MessageMetadata(),
        **base_data,
    )
    assert "Thinking about the problem..." in msg.text
    assert "Considering alternatives." in msg.text
    assert msg.is_hidden  # Hidden: internal reasoning noise

    # Case 3: tether_quote with full attribution
    msg = Message(
        author=MessageAuthor(role="assistant"),
        content=MessageContent(
            content_type="tether_quote",
            text="Important quote here",
            url="https://example.com/article",
            domain="example.com",
            title="Article Title",
        ),
        metadata=MessageMetadata(),
        **base_data,
    )
    text = msg.text
    assert "> Important quote here" in text
    assert "[Article Title](https://example.com/article)" in text
    assert not msg.is_hidden

    # Case 4: tether_quote with only domain
    msg = Message(
        author=MessageAuthor(role="assistant"),
        content=MessageContent(
            content_type="tether_quote",
            text="Another quote",
            url="https://example.org/page",
            domain="example.org",
        ),
        metadata=MessageMetadata(),
        **base_data,
    )
    text = msg.text
    assert "> Another quote" in text
    assert "[example.org](https://example.org/page)" in text

    # Case 5: tether_quote with only URL
    msg = Message(
        author=MessageAuthor(role="assistant"),
        content=MessageContent(
            content_type="tether_quote",
            text="Minimal quote",
            url="https://example.net/",
        ),
        metadata=MessageMetadata(),
        **base_data,
    )
    text = msg.text
    assert "> Minimal quote" in text
    assert "<https://example.net/>" in text


# =============================================================================
# Node and Tree Building Tests
# =============================================================================


class TestBuildNodeTree:
    """Tests for the build_node_tree function."""

    def test_simple_tree(self) -> None:
        """Test building a simple linear tree."""
        mapping = {
            "root": Node(id="root", parent=None, children=["child1"]),
            "child1": Node(id="child1", parent="root", children=["child2"]),
            "child2": Node(id="child2", parent="child1", children=[]),
        }

        result = build_node_tree(mapping)

        # Check parent-child relationships are established
        assert result["child1"].parent_node == result["root"]
        assert result["child2"].parent_node == result["child1"]
        assert result["child1"] in result["root"].children_nodes
        assert result["child2"] in result["child1"].children_nodes

    def test_branching_tree(self) -> None:
        """Test building a tree with branches (DAG structure)."""
        mapping = {
            "root": Node(id="root", parent=None, children=["branch_a", "branch_b"]),
            "branch_a": Node(id="branch_a", parent="root", children=["leaf_a"]),
            "branch_b": Node(id="branch_b", parent="root", children=["leaf_b"]),
            "leaf_a": Node(id="leaf_a", parent="branch_a", children=[]),
            "leaf_b": Node(id="leaf_b", parent="branch_b", children=[]),
        }

        result = build_node_tree(mapping)

        # Root should have two children
        assert len(result["root"].children_nodes) == 2
        assert result["branch_a"] in result["root"].children_nodes
        assert result["branch_b"] in result["root"].children_nodes

        # Leaves should be properly connected
        assert result["leaf_a"].parent_node == result["branch_a"]
        assert result["leaf_b"].parent_node == result["branch_b"]

    def test_repeated_calls_reset_connections(self) -> None:
        """Test that repeated calls don't duplicate connections."""
        mapping = {
            "root": Node(id="root", parent=None, children=["child"]),
            "child": Node(id="child", parent="root", children=[]),
        }

        # Call twice
        build_node_tree(mapping)
        result = build_node_tree(mapping)

        # Should still have only one child
        assert len(result["root"].children_nodes) == 1
        assert result["child"].parent_node == result["root"]

    def test_missing_child_reference_handled(self) -> None:
        """Test that missing child references are handled gracefully."""
        mapping = {
            "root": Node(id="root", parent=None, children=["missing_id"]),
        }

        # Should not raise, just skip the missing child
        result = build_node_tree(mapping)
        assert len(result["root"].children_nodes) == 0

    def test_nodes_linked_only_through_parent(self) -> None:
        """Exports with empty or missing `children` still link through `parent`."""
        mapping = {
            "root": Node(id="root", parent=None),
            "user": Node(id="user", parent="root", children=[]),
            "assistant": Node(id="assistant", parent="user"),
        }

        result = build_node_tree(mapping)

        assert result["assistant"].parent_node is result["user"]
        assert result["user"].parent_node is result["root"]
        assert result["root"].children_nodes == [result["user"]]
        assert result["user"].children_nodes == [result["assistant"]]


def test_ordered_nodes_handles_parent_cycles() -> None:
    """Active-branch ordering should not loop forever on malformed cycles."""
    conv = Conversation(
        title="Cycle test",
        create_time=datetime(2024, 1, 1, tzinfo=UTC),
        update_time=datetime(2024, 1, 1, tzinfo=UTC),
        mapping={
            "root": {
                "id": "root",
                "message": None,
                "parent": "child",
                "children": ["child"],
            },
            "child": {
                "id": "child",
                "message": None,
                "parent": "root",
                "children": ["root"],
            },
        },
        current_node="child",
        conversation_id="cycle-conv",
    )

    ordered = conv.ordered_nodes
    assert [node.id for node in ordered] == ["root", "child"]


def test_ordered_nodes_follows_parent_links_without_children() -> None:
    """The whole active branch is found when exports only record `parent`."""
    conv = Conversation(
        title="Parent links",
        create_time=datetime(2024, 1, 1, tzinfo=UTC),
        update_time=datetime(2024, 1, 1, tzinfo=UTC),
        mapping={
            "root": {"id": "root", "message": None, "parent": None},
            "user": {"id": "user", "message": None, "parent": "root"},
            "edited": {"id": "edited", "message": None, "parent": "root"},
            "assistant": {"id": "assistant", "message": None, "parent": "user"},
        },
        current_node="assistant",
        conversation_id="parent-links-conv",
    )

    ordered = conv.ordered_nodes
    assert [node.id for node in ordered] == ["root", "user", "assistant"]


# =============================================================================
# Conversation Date/Time Property Tests
# =============================================================================


class TestConversationTimeProperties:
    """Tests for Conversation time-related properties."""

    def _make_conversation(self, create_time: datetime) -> Conversation:
        """Helper to create a minimal conversation at a specific time."""
        return Conversation(
            title="Test",
            create_time=create_time,
            update_time=create_time + timedelta(hours=1),
            mapping={
                "root": {
                    "id": "root",
                    "message": None,
                    "parent": None,
                    "children": [],
                },
            },
            current_node="root",
            conversation_id="test_conv",
        )

    def test_week_start_monday(self) -> None:
        """Test that week_start returns the Monday of that week."""
        # Wednesday, January 15, 2025
        conv = self._make_conversation(datetime(2025, 1, 15, 14, 30, tzinfo=UTC))
        week_start = conv.week_start

        assert week_start.weekday() == 0  # Monday
        assert week_start.day == 13  # Monday was Jan 13
        assert week_start.hour == 0
        assert week_start.minute == 0

    def test_week_start_on_monday(self) -> None:
        """Test that week_start on a Monday returns that same day."""
        # Monday, January 13, 2025
        conv = self._make_conversation(datetime(2025, 1, 13, 10, 0, tzinfo=UTC))
        week_start = conv.week_start

        assert week_start.day == 13
        assert week_start.weekday() == 0

    def test_month_start(self) -> None:
        """Test that month_start returns the first of the month."""
        conv = self._make_conversation(datetime(2025, 3, 18, 15, 45, tzinfo=UTC))
        month_start = conv.month_start

        assert month_start.day == 1
        assert month_start.month == 3
        assert month_start.year == 2025
        assert month_start.hour == 0

    def test_year_start(self) -> None:
        """Test that year_start returns January 1st."""
        conv = self._make_conversation(datetime(2025, 7, 29, 12, 0, tzinfo=UTC))
        year_start = conv.year_start

        assert year_start.day == 1
        assert year_start.month == 1
        assert year_start.year == 2025


class TestConversationCustomInstructions:
    """Tests for Conversation.custom_instructions property."""

    def test_custom_instructions_extraction(self) -> None:
        """Test extracting custom instructions from a system message."""
        ts = datetime(2024, 1, 1, tzinfo=UTC).timestamp()
        instructions_data = {
            "about_user_message": "I am a software developer",
            "about_model_message": "Be concise",
        }

        conv = Conversation(
            title="Test",
            create_time=ts,
            update_time=ts,
            mapping={
                "root": {
                    "id": "root",
                    "message": None,
                    "parent": None,
                    "children": ["system_node"],
                },
                "system_node": {
                    "id": "system_node",
                    "message": {
                        "id": "system_node",
                        "author": {"role": "system", "metadata": {}},
                        "create_time": ts,
                        "update_time": ts,
                        "content": {
                            "content_type": "text",
                            "parts": ["Custom instructions"],
                        },
                        "status": "finished_successfully",
                        "end_turn": True,
                        "weight": 1.0,
                        "metadata": {
                            "is_user_system_message": True,
                            "user_context_message_data": instructions_data,
                        },
                        "recipient": "all",
                    },
                    "parent": "root",
                    "children": [],
                },
            },
            current_node="system_node",
            conversation_id="test_conv",
        )

        assert conv.custom_instructions == instructions_data

    def test_no_custom_instructions(self, mock_conversation: Conversation) -> None:
        """Test conversation without custom instructions returns empty dict."""
        assert mock_conversation.custom_instructions == {}


# =============================================================================
# Collection Grouping Tests
# =============================================================================


class TestCollectionGrouping:
    """Tests for ConversationCollection grouping methods."""

    def _make_collection_with_dates(
        self,
        dates: list[datetime],
    ) -> ConversationCollection:
        """Helper to create a collection with conversations at specific dates."""
        conversations = []
        for i, dt in enumerate(dates):
            conv = Conversation(
                title=f"Conv {i}",
                create_time=dt,
                update_time=dt + timedelta(hours=1),
                mapping={
                    "root": {
                        "id": "root",
                        "message": None,
                        "parent": None,
                        "children": [],
                    },
                },
                current_node="root",
                conversation_id=f"conv_{i}",
            )
            conversations.append(conv)
        return ConversationCollection(conversations=conversations)

    def test_group_by_week(self) -> None:
        """Test grouping conversations by week."""
        dates = [
            datetime(2024, 1, 8, 10, 0, tzinfo=UTC),  # Week of Jan 8 (Monday)
            datetime(2024, 1, 10, 10, 0, tzinfo=UTC),  # Same week
            datetime(2024, 1, 15, 10, 0, tzinfo=UTC),  # Week of Jan 15
        ]
        collection = self._make_collection_with_dates(dates)

        groups = collection.group_by_week()

        assert len(groups) == 2
        # Find the week of Jan 8 (Monday)
        week_jan_8 = datetime(2024, 1, 8, 0, 0, 0, 0, tzinfo=UTC)
        assert len(groups[week_jan_8].conversations) == 2

    def test_group_by_month(self) -> None:
        """Test grouping conversations by month."""
        dates = [
            datetime(2024, 1, 5, tzinfo=UTC),
            datetime(2024, 1, 20, tzinfo=UTC),
            datetime(2024, 3, 10, tzinfo=UTC),
        ]
        collection = self._make_collection_with_dates(dates)

        groups = collection.group_by_month()

        assert len(groups) == 2  # January and March
        jan = datetime(2024, 1, 1, 0, 0, 0, 0, tzinfo=UTC)
        assert len(groups[jan].conversations) == 2

    def test_group_by_year(self) -> None:
        """Test grouping conversations by year."""
        dates = [
            datetime(2023, 6, 15, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 12, 31, tzinfo=UTC),
        ]
        collection = self._make_collection_with_dates(dates)

        groups = collection.group_by_year()

        assert len(groups) == 2  # 2023 and 2024
        year_2024 = datetime(2024, 1, 1, 0, 0, 0, 0, tzinfo=UTC)
        assert len(groups[year_2024].conversations) == 2

    def test_empty_collection_grouping(self) -> None:
        """Test grouping methods on empty collection."""
        collection = ConversationCollection()

        assert collection.group_by_week() == {}
        assert collection.group_by_month() == {}
        assert collection.group_by_year() == {}


class TestCollectionProperties:
    """Tests for ConversationCollection properties."""

    def test_last_updated_empty(self) -> None:
        """Test last_updated on empty collection."""
        collection = ConversationCollection()
        assert collection.last_updated == datetime.min.replace(tzinfo=UTC)

    def test_last_updated_with_conversations(
        self,
        mock_conversation: Conversation,
    ) -> None:
        """Test last_updated returns most recent update_time."""
        collection = ConversationCollection(conversations=[mock_conversation])
        assert collection.last_updated == mock_conversation.update_time

    def test_index_property(self, mock_conversation: Conversation) -> None:
        """Test index property returns conversations by ID."""
        collection = ConversationCollection(conversations=[mock_conversation])
        index = collection.index

        assert "conversation_111" in index
        assert index["conversation_111"] == mock_conversation
