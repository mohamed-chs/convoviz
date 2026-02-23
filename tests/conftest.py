"""Fixtures for testing."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import pytest

from convoviz.models import Conversation
from convoviz.models.collection import ConversationCollection

# =============================================================================
# Time Constants
# =============================================================================

DATETIME_111 = datetime(
    year=2023,
    month=7,
    day=29,
    hour=8,
    minute=0,
    second=0,
    tzinfo=UTC,
)

DATETIME_112 = DATETIME_111 + timedelta(minutes=5)


# =============================================================================
# Message/Node Building Helpers
# =============================================================================


def make_message_node(
    node_id: str,
    role: str,
    content: str,
    parent_id: str | None,
    children: list[str],
    timestamp: float,
    *,
    content_type: str = "text",
    metadata: dict | None = None,
    recipient: str = "all",
) -> dict:
    """Helper to create a message node dict."""
    return {
        "id": node_id,
        "message": {
            "id": node_id,
            "author": {"role": role, "metadata": {}},
            "create_time": timestamp,
            "update_time": timestamp,
            "content": {"content_type": content_type, "parts": [content]},
            "status": "finished_successfully",
            "end_turn": True,
            "weight": 1.0,
            "metadata": metadata or {"model_slug": "gpt-4"},
            "recipient": recipient,
        },
        "parent": parent_id,
        "children": children,
    }


def make_root_node(node_id: str, children: list[str]) -> dict:
    """Helper to create a root node (no message)."""
    return {
        "id": node_id,
        "message": None,
        "parent": None,
        "children": children,
    }


# =============================================================================
# Basic Conversation Fixture
# =============================================================================


@pytest.fixture
def mock_conversation_data() -> dict:
    """Return a dictionary representing a raw conversation."""
    return {
        "title": "conversation 111",
        "create_time": DATETIME_111.timestamp(),
        "update_time": DATETIME_112.timestamp(),
        "mapping": {
            "root_node_111": {
                "id": "root_node_111",
                "message": None,
                "parent": None,
                "children": ["system_node_111"],
            },
            "system_node_111": {
                "id": "system_node_111",
                "message": {
                    "id": "system_node_111",
                    "author": {"role": "system", "metadata": {}},
                    "create_time": DATETIME_111.timestamp(),
                    "update_time": DATETIME_111.timestamp(),
                    "content": {
                        "content_type": "text",
                        "parts": ["system message 111"],
                    },
                    "status": "finished_successfully",
                    "end_turn": True,
                    "weight": 1.0,
                    "metadata": {"model_slug": "gpt-4"},
                    "recipient": "all",
                },
                "parent": "root_node_111",
                "children": ["user_node_111"],
            },
            "user_node_111": {
                "id": "user_node_111",
                "message": {
                    "id": "user_node_111",
                    "author": {"role": "user", "metadata": {}},
                    "create_time": DATETIME_111.timestamp(),
                    "update_time": DATETIME_111.timestamp(),
                    "content": {"content_type": "text", "parts": ["user message 111"]},
                    "status": "finished_successfully",
                    "end_turn": True,
                    "weight": 1.0,
                    "metadata": {"model_slug": "gpt-4"},
                    "recipient": "all",
                },
                "parent": "system_node_111",
                "children": ["assistant_node_111"],
            },
            "assistant_node_111": {
                "id": "assistant_node_111",
                "message": {
                    "id": "assistant_node_111",
                    "author": {"role": "assistant", "metadata": {}},
                    "create_time": DATETIME_112.timestamp(),
                    "update_time": DATETIME_112.timestamp(),
                    "content": {
                        "content_type": "text",
                        "parts": ["assistant message 111"],
                    },
                    "status": "finished_successfully",
                    "end_turn": True,
                    "weight": 1.0,
                    "metadata": {"model_slug": "gpt-4"},
                    "recipient": "all",
                },
                "parent": "user_node_111",
                "children": [],
            },
        },
        "current_node": "assistant_node_111",
        "plugin_ids": [],
        "conversation_id": "conversation_111",
    }


@pytest.fixture
def mock_conversation(mock_conversation_data: dict) -> Conversation:
    """Return a Conversation object."""
    return Conversation(**mock_conversation_data)


@pytest.fixture
def mock_conversations_json(mock_conversation_data: dict, tmp_path):
    """Create a temporary conversations.json file."""
    data = [mock_conversation_data]
    file_path = tmp_path / "conversations.json"
    with Path.open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def mock_zip_file(mock_conversations_json, tmp_path):
    """Create a temporary zip file containing conversations.json."""
    zip_path = tmp_path / "test_export.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.write(mock_conversations_json, "conversations.json")
    return zip_path


@pytest.fixture
def second_conversation_data() -> dict:
    """Return a second conversation dict distinct from mock_conversation_data."""
    ts = DATETIME_111.timestamp() + 600
    return {
        "title": "conversation 222",
        "create_time": ts,
        "update_time": ts + 300,
        "mapping": {
            "root_node_222": make_root_node("root_node_222", ["user_node_222"]),
            "user_node_222": make_message_node(
                "user_node_222",
                "user",
                "user message 222",
                "root_node_222",
                ["assistant_node_222"],
                ts,
            ),
            "assistant_node_222": make_message_node(
                "assistant_node_222",
                "assistant",
                "assistant message 222",
                "user_node_222",
                [],
                ts + 60,
            ),
        },
        "current_node": "assistant_node_222",
        "conversation_id": "conversation_222",
    }


@pytest.fixture
def mock_multi_zip_file(
    mock_conversation_data: dict, second_conversation_data: dict, tmp_path: Path
):
    """Create a ZIP with split conversation files (conversations-000/001.json)."""
    zip_path = tmp_path / "multi_export.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations-000.json", json.dumps([mock_conversation_data]))
        zf.writestr("conversations-001.json", json.dumps([second_conversation_data]))
    return zip_path


# =============================================================================
# Additional Conversation Fixtures
# =============================================================================


@pytest.fixture
def branching_conversation_data() -> dict:
    """Return a conversation with branching (edited response regeneration).

    Structure:
        root -> user_1 -> assistant_1a (original)
                       -> assistant_1b (regenerated)
    """
    ts = DATETIME_111.timestamp()
    return {
        "title": "Branching Conversation",
        "create_time": ts,
        "update_time": ts + 300,
        "mapping": {
            "root": make_root_node("root", ["user_1"]),
            "user_1": make_message_node(
                "user_1",
                "user",
                "Hello, can you help?",
                "root",
                ["assistant_1a", "assistant_1b"],
                ts,
            ),
            "assistant_1a": make_message_node(
                "assistant_1a",
                "assistant",
                "Of course! How can I assist you?",
                "user_1",
                [],
                ts + 60,
            ),
            "assistant_1b": make_message_node(
                "assistant_1b",
                "assistant",
                "Sure thing! What do you need?",
                "user_1",
                [],
                ts + 120,
            ),
        },
        "current_node": "assistant_1b",
        "conversation_id": "branching_conv",
    }


@pytest.fixture
def branching_conversation(branching_conversation_data: dict) -> Conversation:
    """Return a Conversation object with branching structure."""
    return Conversation(**branching_conversation_data)


@pytest.fixture
def empty_conversation_data() -> dict:
    """Return a minimal conversation with only root node."""
    ts = DATETIME_111.timestamp()
    return {
        "title": "Empty Conversation",
        "create_time": ts,
        "update_time": ts,
        "mapping": {
            "root": make_root_node("root", []),
        },
        "current_node": "root",
        "conversation_id": "empty_conv",
    }


@pytest.fixture
def empty_conversation(empty_conversation_data: dict) -> Conversation:
    """Return a Conversation with no messages."""
    return Conversation(**empty_conversation_data)


@pytest.fixture
def mock_collection(mock_conversation: Conversation) -> ConversationCollection:
    """Return a ConversationCollection with one conversation."""
    return ConversationCollection(conversations=[mock_conversation])
