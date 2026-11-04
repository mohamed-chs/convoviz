from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

import pytest

from convoviz.chatgpt import LoadedExport, load_export
from convoviz.diagnostics import Diagnostics
from convoviz.render import RenderOptions
from convoviz.source import ExportSource

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


@pytest.fixture
def options() -> RenderOptions:
    return RenderOptions(timezone=UTC)


@pytest.fixture
def load(workdir: Path) -> Callable[[Path], tuple[LoadedExport, Diagnostics]]:
    def _load(path: Path) -> tuple[LoadedExport, Diagnostics]:
        diagnostics = Diagnostics()
        source = ExportSource.open(path, workdir)
        return load_export([source], diagnostics), diagnostics

    return _load
