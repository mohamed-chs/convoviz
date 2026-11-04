from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from factory import PNG_1PX, ConversationBuilder, write_export

from convoviz import cli

if TYPE_CHECKING:
    from pathlib import Path


def _export(tmp_path: Path, name: str = "export.zip") -> Path:
    b = ConversationBuilder(title="Hello")
    b.user("hi")
    b.assistant("hello there")
    return write_export(tmp_path / name, [b.build(), {}], files={"file_1-a.dat": PNG_1PX})


def _run(args: list[str]) -> int:
    return cli.app(args, result_action="return_value")


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "config_path", lambda: tmp_path / "cfg/config.toml")
    monkeypatch.setattr("convoviz.config.config_path", lambda: tmp_path / "cfg/config.toml")
    monkeypatch.setattr("platformdirs.user_downloads_dir", lambda: str(tmp_path / "Downloads"))
    monkeypatch.setattr("platformdirs.user_documents_dir", lambda: str(tmp_path / "Documents"))


def test_convert(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    assert _run([str(_export(tmp_path)), "-o", str(out), "--timezone", "UTC"]) == 0
    assert (out / "conversations/2026/2026-01-01 Hello.md").exists()
    assert (out / "insights.html").exists()
    printed = capsys.readouterr().out
    assert "1 conversation" in printed
    assert "1 new" in printed
    assert _run([str(_export(tmp_path)), "-o", str(out), "-q", "--no-insights"]) == 0
    assert capsys.readouterr().out == ""


def test_finds_export_in_downloads(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    _export(downloads, "abc-2026-11-01.zip")
    assert _run(["-y"]) == 0
    assert "abc-2026-11-01.zip" in capsys.readouterr().out
    assert (tmp_path / "Documents/ChatGPT Archive/index.md").exists()


def test_no_export_found(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["-y"]) == 2
    assert "No ChatGPT export found" in capsys.readouterr().err


def test_not_an_export(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    other = tmp_path / "x.zip"
    write_export(other, [])
    (tmp_path / "folder").mkdir()
    assert _run([str(tmp_path / "folder"), "-o", str(tmp_path / "o")]) == 2
    assert "doesn't contain ChatGPT conversations" in capsys.readouterr().err


def test_bad_timezone(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run([str(_export(tmp_path)), "--timezone", "Nowhere/Land"]) == 2
    assert "unknown time zone" in capsys.readouterr().err


def test_inspect_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["inspect", str(_export(tmp_path)), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["shape"]["conversations"] == 1
    assert "Hello" not in json.dumps(report)


def test_inspect_human(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["inspect", str(_export(tmp_path))]) == 0
    assert "content_types" in capsys.readouterr().out


def test_config_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(["config", "path"])
    assert "not created yet" in capsys.readouterr().out
    assert _run(["config", "init"]) == 0
    assert (tmp_path / "cfg/config.toml").exists()
    assert _run(["config", "init"]) == 1
    assert _run(["config", "init", "--force"]) == 0
