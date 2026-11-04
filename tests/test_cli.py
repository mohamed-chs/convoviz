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


def test_output_problems_are_reported_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    a_file = tmp_path / "afile"
    a_file.write_text("x")
    assert _run([str(_export(tmp_path)), "-o", str(a_file)]) == 2
    assert "is a file" in capsys.readouterr().err

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(13, "Permission denied", str(tmp_path / "locked"))

    monkeypatch.setattr("convoviz.writer.Writer.write", refuse)
    assert _run([str(_export(tmp_path)), "-o", str(tmp_path / "locked")]) == 1
    err = capsys.readouterr().err
    assert "Permission denied" in err
    assert "Traceback" not in err


def test_interrupt_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("convoviz.writer.Writer.write", interrupt)
    assert _run([str(_export(tmp_path)), "-o", str(tmp_path / "o")]) == 130
    assert "Run the same command again" in capsys.readouterr().err


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


def test_inspect_finds_export_in_downloads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["inspect"]) == 2
    assert "No ChatGPT export found" in capsys.readouterr().err
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    _export(downloads, "abc-2026-11-01.zip")
    assert _run(["inspect", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["shape"]["conversations"] == 1
    assert "abc-2026-11-01.zip" in captured.err


def test_config_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(["config", "path"])
    assert "not created yet" in capsys.readouterr().out
    assert _run(["config", "init"]) == 0
    assert (tmp_path / "cfg/config.toml").exists()
    assert _run(["config", "init"]) == 1
    assert _run(["config", "init", "--force"]) == 0


def test_old_config_file_explains_the_upgrade(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "cfg/config.toml"
    config.parent.mkdir()
    old = 'output_folder = "~/Documents/ChatGPT-Data"\noutputs = ["markdown"]\n[wordcloud]\n'
    config.write_text(old)
    assert _run([str(_export(tmp_path)), "-o", str(tmp_path / "out")]) == 2
    assert "convoviz 0.x config file" in capsys.readouterr().err
    assert _run(["config", "init", "--force"]) == 0
    assert (tmp_path / "cfg/config.toml.bak").read_text() == old
    assert _run([str(_export(tmp_path)), "-o", str(tmp_path / "out")]) == 0


def test_errors_are_listed_verbatim_and_capped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from convoviz import writer

    real = writer._atomic_write

    def fail_on_notes(path: Path, data: bytes) -> None:
        if "conversations" in path.parts:
            raise OSError(28, "No space left on device")
        real(path, data)

    monkeypatch.setattr(writer, "_atomic_write", fail_on_notes)
    conversations = []
    for n in range(12):
        b = ConversationBuilder(title=f"Chat [{n}] [/draft]", conversation_id=f"id-{n}")
        b.user("hi")
        b.assistant("hello")
        conversations.append(b.build())
    export = write_export(tmp_path / "export.zip", conversations)
    assert _run([str(export), "-o", str(tmp_path / "out"), "--no-insights"]) == 1
    printed = capsys.readouterr().out
    assert printed.count("could not write") == 10
    assert "could not write 'Chat [0] [/draft]'" in printed
    assert "… and 2 more" in printed


def _damage(path: Path, marker: bytes) -> None:
    """Flip a byte inside a stored ZIP member, so reading it fails its CRC check."""
    data = bytearray(path.read_bytes())
    at = data.index(marker)
    data[at] ^= 0xFF
    path.write_bytes(bytes(data))


def test_damaged_files_in_the_zip_are_reported_and_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    conversations = []
    for n, text in enumerate(["kept", "LOST-SHARD"]):
        b = ConversationBuilder(title=f"Chat {n}", conversation_id=f"id-{n}")
        b.user("hi")
        b.assistant(text)
        b.add(
            "tool",
            {
                "content_type": "multimodal_text",
                "parts": [
                    {
                        "content_type": "image_asset_pointer",
                        "asset_pointer": f"sediment://file_{n}",
                    }
                ],
            },
        )
        conversations.append(b.build())
    image = PNG_1PX + b"LOST-IMAGE"
    export = write_export(
        tmp_path / "export.zip", conversations, sharded=1, files={"file_0-a.dat": image}
    )
    _damage(export, b"LOST-SHARD")
    _damage(export, b"LOST-IMAGE")
    out = tmp_path / "out"
    assert _run([str(export), "-o", str(out), "--no-insights"]) == 1
    printed = capsys.readouterr().out
    assert "could not read conversations-001.json: damaged" in printed
    assert "could not copy file_0-a.dat: damaged" in printed
    assert [p.name for p in (out / "conversations").rglob("*.md")] == ["2026-01-01 Chat 0.md"]
    assert not (out / "attachments").exists()


def test_cut_short_download_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    export = _export(tmp_path)
    export.write_bytes(export.read_bytes()[:100])
    assert _run([str(export), "-o", str(tmp_path / "out")]) == 2
    assert "download the export again" in capsys.readouterr().err
