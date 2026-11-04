"""User configuration: an optional TOML file; command-line flags override it."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, get_args
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import platformdirs

from convoviz.render import Branches, RenderOptions
from convoviz.writer import Layout

APP_NAME = "convoviz"

DEFAULT_CONFIG = """\
# convoviz configuration. Every setting is optional; delete what you don't change.
# Command-line flags override these values.

# Where the archive is written. Re-running into the same folder only updates
# what changed and never overwrites notes you edited (unless --force).
output = "{output}"

# "year": conversations/2026/2026-03-14 Title.md    "flat": conversations/2026-03-14 Title.md
layout = "year"

# Write insights.html (charts of your usage).
insights = true

[markdown]
user_label = "You"
assistant_label = "ChatGPT"
# Show the time of each turn next to its heading.
timestamps = true
# "all": edited/regenerated alternatives are kept, folded at the point they diverge.
# "active": only the branch you last saw in ChatGPT.
branches = "all"
# Include the model's reasoning summaries (folded).
reasoning = true
# Include code the assistant ran, its output, and quoted web pages (folded).
tool_calls = true
tags = ["chatgpt"]
# IANA time zone for dates, e.g. "Europe/Paris". Empty: this computer's time zone.
timezone = ""
"""


class ConfigError(Exception):
    """The configuration file is invalid."""


def default_output() -> Path:
    """``~/Documents/ChatGPT Archive`` (or the platform's equivalent)."""
    return Path(platformdirs.user_documents_dir()) / "ChatGPT Archive"


def config_path() -> Path:
    """Location of the user's config file."""
    return Path(platformdirs.user_config_dir(APP_NAME, appauthor=False)) / "config.toml"


@dataclass(frozen=True, slots=True)
class Config:
    """Resolved settings."""

    output: Path = field(default_factory=default_output)
    layout: Layout = "year"
    insights: bool = True
    render: RenderOptions = field(default_factory=RenderOptions)

    def with_overrides(self, **changes: Any) -> Config:
        """Apply command-line values (``None`` means "not given")."""
        top = {k: v for k, v in changes.items() if v is not None and k in _TOP_KEYS}
        render = {k: v for k, v in changes.items() if v is not None and k not in _TOP_KEYS}
        config = replace(self, **top)
        if render:
            config = replace(config, render=replace(config.render, **render))
        return config


_TOP_KEYS = frozenset({"output", "layout", "insights"})


def load_config(path: Path | None = None) -> Config:
    """Read the config file if it exists; defaults otherwise."""
    path = path or config_path()
    if not path.exists():
        return Config()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return _from_dict(data, path)


def _from_dict(data: dict[str, Any], path: Path) -> Config:
    def fail(message: str) -> ConfigError:
        return ConfigError(f"{path}: {message}")

    unknown = set(data) - {*_TOP_KEYS, "markdown"}
    if unknown:
        raise fail(f"unknown setting(s): {', '.join(sorted(unknown))}")
    config = Config()
    if "output" in data:
        if not isinstance(data["output"], str) or not data["output"].strip():
            raise fail("output must be a folder path")
        config = replace(config, output=Path(data["output"]).expanduser())
    if "layout" in data:
        if data["layout"] not in get_args(Layout):
            raise fail(f"layout must be one of {', '.join(get_args(Layout))}")
        config = replace(config, layout=data["layout"])
    if "insights" in data:
        config = replace(config, insights=_bool(data["insights"], "insights", fail))

    markdown = data.get("markdown", {})
    if not isinstance(markdown, dict):
        raise fail("[markdown] must be a table")
    options: dict[str, Any] = {}
    for key in ("user_label", "assistant_label"):
        if key in markdown:
            if not isinstance(markdown[key], str) or not markdown[key].strip():
                raise fail(f"markdown.{key} must be a non-empty string")
            options[key] = markdown[key].strip()
    for key in ("timestamps", "reasoning", "tool_calls"):
        if key in markdown:
            options[key] = _bool(markdown[key], f"markdown.{key}", fail)
    if "branches" in markdown:
        if markdown["branches"] not in get_args(Branches):
            raise fail(f"markdown.branches must be one of {', '.join(get_args(Branches))}")
        options["branches"] = markdown["branches"]
    if "tags" in markdown:
        tags = markdown["tags"]
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise fail("markdown.tags must be a list of strings")
        options["tags"] = tuple(t.strip() for t in tags if t.strip())
    if markdown.get("timezone"):
        options["timezone"] = parse_timezone(markdown["timezone"], fail)
    unknown_md = set(markdown) - {
        "user_label",
        "assistant_label",
        "timestamps",
        "reasoning",
        "tool_calls",
        "branches",
        "tags",
        "timezone",
    }
    if unknown_md:
        raise fail(f"unknown markdown setting(s): {', '.join(sorted(unknown_md))}")
    return replace(config, render=replace(config.render, **options))


def _bool(value: object, name: str, fail: Any) -> bool:
    if not isinstance(value, bool):
        raise fail(f"{name} must be true or false")
    return value


def parse_timezone(name: object, fail: Any = ConfigError) -> ZoneInfo:
    """An IANA time zone, or a clear error."""
    if not isinstance(name, str):
        raise fail('timezone must be a string such as "Europe/Paris"')
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise fail(f"unknown time zone {name!r}") from exc


def write_default_config(path: Path) -> None:
    """Create a commented config file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output = default_output().as_posix()
    path.write_text(DEFAULT_CONFIG.format(output=output), encoding="utf-8")
