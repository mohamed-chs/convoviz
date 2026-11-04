"""Command-line interface."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import orjson
import platformdirs
from cyclopts import App, Parameter
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm

from convoviz.census import census
from convoviz.chatgpt import load_export
from convoviz.config import (
    ConfigError,
    config_path,
    load_config,
    parse_timezone,
    write_default_config,
)
from convoviz.diagnostics import Diagnostics
from convoviz.insights import render_insights
from convoviz.render import Branches  # noqa: TC001 - cyclopts reads annotations at runtime
from convoviz.source import (
    ExportSource,
    SourceError,
    find_recent_exports,
    group_inputs,
    make_workdir,
    with_sibling_parts,
)
from convoviz.stats import compute
from convoviz.writer import Layout, Writer

if TYPE_CHECKING:
    from convoviz.config import Config
    from convoviz.writer import WriteReport

ISSUES_URL = "https://github.com/mohamed-chs/convoviz/issues"

console = Console(highlight=False, soft_wrap=True)
err = Console(stderr=True, highlight=False, soft_wrap=True)


def _version() -> str:
    try:
        return version("convoviz")
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


app = App(
    name="convoviz",
    help="Turn your ChatGPT data export into a Markdown archive and usage charts.",
    version=_version,
    help_format="markdown",
)
config_app = App(name="config", help="Show or create the configuration file.")
app.command(config_app)


class UsageError(Exception):
    """A problem the user can fix; printed without a traceback."""


@app.default
def convert(
    exports: Annotated[list[Path] | None, Parameter(name="EXPORT")] = None,
    /,
    *,
    output: Annotated[Path | None, Parameter(name=["--output", "-o"])] = None,
    layout: Layout | None = None,
    branches: Branches | None = None,
    timezone: str | None = None,
    insights: bool | None = None,
    force: Annotated[bool, Parameter(negative="")] = False,
    yes: Annotated[bool, Parameter(name=["--yes", "-y"], negative="")] = False,
    quiet: Annotated[bool, Parameter(name=["--quiet", "-q"], negative="")] = False,
    verbose: Annotated[int, Parameter(name=["--verbose", "-v"], count=True)] = 0,
    config: Path | None = None,
) -> int:
    """Convert a ChatGPT export into Markdown notes.

    Re-running with a newer export updates the archive in place: only changed
    conversations are rewritten, and notes you edited are kept.

    Parameters
    ----------
    exports
        The export: the ZIP from OpenAI's email (or several part ZIPs), an extracted
        folder, or conversations.json. Default: the newest export in Downloads.
    output
        Folder to write the archive to. Default: from the config file, else
        ~/Documents/ChatGPT Archive.
    layout
        "year" puts notes in one folder per year; "flat" in a single folder.
    branches
        "all" keeps edited/regenerated alternatives (folded); "active" keeps only
        the branch you last viewed.
    timezone
        IANA time zone for dates (e.g. Europe/Paris). Default: this computer's.
    insights
        Write insights.html with charts of your usage.
    force
        Overwrite notes even if you edited them since the last run.
    yes
        Don't ask for confirmation.
    quiet
        Only print errors.
    verbose
        Log details while running (repeat for more).
    config
        Use this config file instead of the default one.
    """
    _setup_logging(verbose)
    try:
        settings = load_config(config).with_overrides(
            output=output.expanduser() if output else None,
            layout=layout,
            insights=insights,
            branches=branches,
            timezone=parse_timezone(timezone) if timezone else None,
        )
        paths = _resolve_inputs(exports, assume_yes=yes, quiet=quiet)
        return _run(paths, settings, force=force, quiet=quiet)
    except (UsageError, SourceError, ConfigError) as exc:
        err.print(f"[red]Error:[/] {exc}")
        return 2


@app.command(name="inspect")
def inspect_export(
    exports: Annotated[list[Path], Parameter(name="EXPORT")],
    /,
    *,
    json: Annotated[bool, Parameter(negative="")] = False,
) -> int:
    """Describe an export's structure, anonymously, for bug reports.

    Lists only field names, type tags and counts: never titles, message text,
    links, file names or ids. Paste the --json output into a GitHub issue.

    Parameters
    ----------
    exports
        The export ZIP(s), folder, or conversations.json.
    json
        Print machine-readable JSON.
    """
    try:
        with make_workdir() as workdir:
            report = census(_open_sources([p.expanduser() for p in exports], Path(workdir)))
    except SourceError as exc:
        err.print(f"[red]Error:[/] {exc}")
        return 2
    if json:
        sys.stdout.write(orjson.dumps(report, option=orjson.OPT_INDENT_2).decode() + "\n")
        return 0
    _print_tree(report)
    console.print(f"\nShare this with [bold]convoviz inspect --json[/] in an issue: {ISSUES_URL}")
    return 0


def _print_tree(data: dict[str, object], indent: int = 0) -> None:
    pad = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            console.print(f"{pad}{key}:", style="bold", markup=False)
            _print_tree(value, indent + 1)
        else:
            console.print(f"{pad}{key}: {value}", markup=False)


@config_app.command(name="path")
def config_show_path() -> None:
    """Print where the config file is (or would be)."""
    path = config_path()
    console.print(f"{path}{'' if path.exists() else '  (not created yet)'}")


@config_app.command(name="init")
def config_init(*, force: Annotated[bool, Parameter(negative="")] = False) -> int:
    """Create a commented config file with every setting.

    Parameters
    ----------
    force
        Replace an existing config file.
    """
    path = config_path()
    if path.exists() and not force:
        err.print(f"[yellow]{path} already exists.[/] Use --force to replace it.")
        return 1
    write_default_config(path)
    console.print(f"Created {path}")
    return 0


# --------------------------------------------------------------------------- run


def _setup_logging(verbose: int) -> None:
    # By default nothing is logged: the end-of-run summary reports warnings and errors.
    level = {0: logging.CRITICAL + 1, 1: logging.INFO}.get(verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=err, show_time=False, show_path=False)],
        force=True,
    )


def _open_sources(paths: list[Path], workdir: Path) -> list[ExportSource]:
    """One source per export (split parts together), newest export first."""
    groups = sorted(group_inputs(paths), key=lambda g: _mtime(g[0]), reverse=True)
    return [ExportSource.open_many(g, workdir / f"export-{i}") for i, g in enumerate(groups)]


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _resolve_inputs(exports: list[Path] | None, *, assume_yes: bool, quiet: bool) -> list[Path]:
    if exports:
        return [p.expanduser() for p in exports]
    downloads = Path(platformdirs.user_downloads_dir())
    found = find_recent_exports(downloads)
    if not found:
        raise UsageError(
            f"No ChatGPT export found in {downloads}. Pass the path to the export ZIP, e.g.\n"
            "  convoviz ~/Downloads/chatgpt-export.zip\n"
            "Request an export in ChatGPT: Settings → Data controls → Export data."
        )
    newest = found[0]
    paths = with_sibling_parts(newest)
    if not quiet:
        age = _age(newest)
        names = ", ".join(p.name for p in paths)
        console.print(f"Found export: [bold]{names}[/] ({age})")
    if not assume_yes and sys.stdin.isatty() and not Confirm.ask("Use it?", default=True):
        raise UsageError("Cancelled. Pass the export path explicitly.")
    return [newest]


def _age(path: Path) -> str:
    seconds = max(0.0, time.time() - path.stat().st_mtime)
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} days ago"


def _run(paths: list[Path], settings: Config, *, force: bool, quiet: bool) -> int:
    diagnostics = Diagnostics()
    started = time.monotonic()
    with make_workdir() as workdir:
        sources = _open_sources(paths, Path(workdir))
        for source in sources:
            if not source.has_conversation_data():
                raise UsageError(
                    f"{source.label} doesn't contain ChatGPT conversations (no conversations.json)."
                )
        with nullcontext() if quiet else console.status("Reading export…"):
            export = load_export(sources, diagnostics)
        writer = Writer(
            settings.output,
            settings.render,
            diagnostics,
            layout=settings.layout,
            force=force,
        )
        progress = Progress(
            TextColumn("Writing notes"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=quiet,
            transient=True,
        )
        with progress:
            task = progress.add_task("notes", total=len(export.conversations))
            report = writer.write(export, progress=lambda _: progress.advance(task))
        if settings.insights:
            stats = compute(export.conversations, settings.render.timezone)
            (settings.output / "insights.html").write_text(render_insights(stats), encoding="utf-8")
    if not quiet:
        _summary(report, diagnostics, settings, time.monotonic() - started)
    elif diagnostics.errors:
        for message in diagnostics.errors[:10]:
            err.print(f"[red]error:[/] {message}")
    return 1 if diagnostics.errors else 0


def _summary(
    report: WriteReport, diagnostics: Diagnostics, settings: Config, seconds: float
) -> None:
    say = lambda text: console.print(text, soft_wrap=True)  # noqa: E731 - local shorthand
    noun = "conversation" if report.total_notes == 1 else "conversations"
    where = f"[bold]{settings.output}[/]"
    say(f"[bold green]✓[/] {report.total_notes} {noun} in {where} ({seconds:.1f}s)")
    counts = [
        (report.created, "new"),
        (report.updated, "updated"),
        (report.moved, "renamed"),
        (report.unchanged, "unchanged"),
    ]
    say(
        "  " + " · ".join(f"{n} {label}" for n, label in counts if n)
        if any(n for n, _ in counts)
        else "  nothing to write"
    )
    if report.attachments_copied or report.attachments_missing:
        line = f"  {report.attachments_copied} attachment(s) copied"
        if report.attachments_missing:
            line += f", {report.attachments_missing} not included in the export"
        say(line)
    if report.kept_edits:
        kept = len(report.kept_edits)
        say(f"  [yellow]Kept {kept} note(s) you edited[/] (--force overwrites them):")
        for path in report.kept_edits[:5]:
            say(f"    {path}")
    charts = ", or insights.html for charts" if settings.insights else ""
    say(f"  Start with index.md{charts}.")
    if diagnostics.skipped:
        say(f"  [dim]Skipped {len(diagnostics.skipped)} empty or unusable record(s).[/]")
    for message in diagnostics.warnings[:5]:
        say(f"  [yellow]warning:[/] {message}")
    for message in diagnostics.errors[:10]:
        say(f"  [red]error:[/] {message}")
    if diagnostics.unknown or diagnostics.errors:
        kinds = ", ".join(sorted(diagnostics.unknown)[:6])
        if kinds:
            say(f"  [dim]Content convoviz doesn't know yet ({kinds}) was shown as plain text.[/]")
        say(
            "  [dim]If something looks wrong, run `convoviz inspect --json <export>` and share "
            f"the output (it contains no personal data) at {ISSUES_URL}[/]"
        )


def main() -> None:
    """Entry point for the ``convoviz`` script."""
    result = app()
    sys.exit(result if isinstance(result, int) else 0)
