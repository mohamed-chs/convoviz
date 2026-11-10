from __future__ import annotations

import re
import tomllib
from typing import TYPE_CHECKING

import pytest

from convoviz.config import DEFAULT_CONFIG, Config, ConfigError, load_config, write_default_config

if TYPE_CHECKING:
    from pathlib import Path


def test_missing_file_gives_defaults(tmp_path: Path) -> None:
    assert load_config(tmp_path / "none.toml") == Config()


def test_default_config_file_matches_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    write_default_config(path)
    tomllib.loads(path.read_text())
    loaded = load_config(path)
    defaults = Config()
    assert loaded.render == defaults.render
    assert (loaded.layout, loaded.insights, loaded.output) == (
        defaults.layout,
        defaults.insights,
        defaults.output,
    )
    assert "{output}" in DEFAULT_CONFIG


def test_values_are_read(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text(
        'output = "~/notes"\nlayout = "flat"\nignored_words = [" Foo ", "BAR"]\n[markdown]\nbranches = "active"\n'
        'tags = ["ai", "chatgpt"]\ntimezone = "Europe/Paris"\nuser_label = "Me"\n'
    )
    config = load_config(path)
    assert config.layout == "flat"
    assert config.ignored_words == {"foo", "bar"}
    assert config.output.name == "notes"
    assert "~" not in str(config.output)
    assert config.render.branches == "active"
    assert config.render.tags == ("ai", "chatgpt")
    assert str(config.render.timezone) == "Europe/Paris"
    assert config.render.user_label == "Me"


@pytest.mark.parametrize(
    "content",
    [
        "layout = 'tree'",
        "colour = 1",
        "ignored_words = 'foo'",
        "[markdown]\nbranches = 'some'",
        "[markdown]\ntimestamps = 'yes'",
        "[markdown]\ntimezone = 'Mars/Olympus'",
        "[markdown]\nfoo = 1",
        "not toml at all [",
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "c.toml"
    path.write_text(content)
    with pytest.raises(ConfigError):
        load_config(path)


def test_overrides() -> None:
    config = Config().with_overrides(layout="flat", branches="active", output=None)
    assert config.layout == "flat"
    assert config.render.branches == "active"


@pytest.mark.parametrize(
    ("content", "hint"),
    [
        ("timezone = 'Europe/Paris'", "timezone belongs under [markdown]"),
        ("[markdown]\nlayout = 'flat'", "layout belongs above [markdown]"),
        ("[markdown]\ntimezone = 'Mars'", 'such as "Europe/Paris"'),
    ],
)
def test_errors_say_how_to_fix_them(tmp_path: Path, content: str, hint: str) -> None:
    path = tmp_path / "c.toml"
    path.write_text(content)
    with pytest.raises(ConfigError, match=re.escape(hint)):
        load_config(path)
