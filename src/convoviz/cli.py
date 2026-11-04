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
from rich.markup import escape
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
    from collections.abc import Callable

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
    force: Annotated[bool, Parameter(negative="", show_default=False)] = False,
    yes: Annotated[bool, Parameter(name=["--yes", "-y"], negative="", show_default=False)] = False,
    quiet: Annotated[
        bool, Parameter(name=["--quiet", "-q"], negative="", show_default=False)
    ] = False,
    verbose: Annotated[
        int, Parameter(name=["--verbose", "-v"], count=True, show_default=False)
    ] = 0,
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
        Overwrite notes and files you edited since the last run, and recreate
        notes you deleted.
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
        _check_output(settings.output)
        paths = _resolve_inputs(exports, assume_yes=yes, quiet=quiet)
        return _run(paths, settings, force=force, quiet=quiet)
    except (UsageError, SourceError, ConfigError) as exc:
        err.print(f"[red]Error:[/] {escape(str(exc))}")
        return 2
    except OSError as exc:
        # Permissions, a full disk, a vanished folder: say what failed, no traceback.
        where = f" ({exc.filename})" if exc.filename else ""
        err.print(f"[red]Error:[/] {escape(f'{exc.strerror or exc}{where}')}")
        return 1
    except KeyboardInterrupt:
        err.print("\nInterrupted. Run the same command again to finish; nothing is lost.")
        return 130


@app.command(name="inspect")
def inspect_export(
    exports: Annotated[list[Path], Parameter(name="EXPORT")],
    /,
    *,
    json: Annotated[bool, Parameter(negative="", show_default=False)] = False,
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
        err.print(f"[red]Error:[/] {escape(str(exc))}")
        return 2
    except OSError as exc:
        where = f" ({exc.filename})" if exc.filename else ""
        err.print(f"[red]Error:[/] {escape(f'{exc.strerror or exc}{where}')}")
        return 1
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
    console.print(f"{path}{'' if path.exists() else '  (not created yet)'}", markup=False)


@config_app.command(name="init")
def config_init(
    *, force: Annotated[bool, Parameter(negative="", show_default=False)] = False
) -> int:
    """Create a commented config file with every setting.

    Parameters
    ----------
    force
        Replace an existing config file (it is kept as ``config.toml.bak``).
    """
    path = config_path()
    if path.exists() and not force:
        err.print(f"[yellow]{escape(str(path))} already exists.[/] Use --force to replace it.")
        return 1
    if path.exists():
        backup = path.with_name(path.name + ".bak")
        path.replace(backup)
        console.print(f"Kept the old file as {backup}", markup=False)
    write_default_config(path)
    console.print(f"Created {path}", markup=False)
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
        names = escape(", ".join(p.name for p in paths))
        console.print(f"Found export: [bold]{names}[/] ({age})")
    if not assume_yes and sys.stdin.isatty() and not Confirm.ask("Use it?", default=True):
        raise UsageError("Cancelled. Pass the export path explicitly.")
    return [newest]


def _check_output(output: Path) -> None:
    if output.exists() and not output.is_dir():
        raise UsageError(f"The output {output} is a file; pass a folder.")


def _age(path: Path) -> str:
    seconds = max(0.0, time.time() - path.stat().st_mtime)
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{round(seconds / 60)} min ago"
    if seconds < 86400:
        return f"{_plural(int(seconds // 3600), 'hour')} ago"
    days = int(seconds // 86400)
    return "yesterday" if days == 1 else f"{days} days ago"


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def _display(path: Path) -> str:
    """``path`` with the home folder shortened to ``~``."""
    try:
        return f"~/{path.resolve().relative_to(Path.home().resolve()).as_posix()}"
    except ValueError:
        return str(path)


def _file_link(path: Path, label: str) -> str:
    """A clickable file name in terminals that support links."""
    return f"[link={path.resolve().as_uri()}]{label}[/link]"


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
        extra: dict[str, bytes] = {}
        if settings.insights:
            stats = compute(export.conversations, settings.render.timezone, settings.ignored_words)
            extra["insights.html"] = render_insights(stats).encode()
        with progress:
            task = progress.add_task("notes", total=len(export.conversations))
            report = writer.write(
                export, progress=lambda _: progress.advance(task), extra_files=extra
            )
    if not quiet:
        _summary(report, diagnostics, settings, time.monotonic() - started)
    else:
        _messages(err.print, "[red]error:[/]", diagnostics.errors, 10)
    return 1 if diagnostics.errors else 0


def _summary(
    report: WriteReport, diagnostics: Diagnostics, settings: Config, seconds: float
) -> None:
    say = lambda text: console.print(text, soft_wrap=True)  # noqa: E731 - local shorthand
    where = f"[bold]{escape(_display(settings.output))}[/]"
    total = _plural(report.total_notes, "conversation")
    say(f"[bold green]✓[/] {total} in {where} ({seconds:.1f}s)")
    counts = [
        (report.created, "new"),
        (report.updated, "updated"),
        (report.moved, "renamed"),
        (report.unchanged, "unchanged"),
    ]
    say(
        "  " + " · ".join(f"{n:,} {label}" for n, label in counts if n)
        if any(n for n, _ in counts)
        else "  nothing to write"
    )
    files = []
    if copied := report.attachments_copied:
        files.append(f"{_plural(copied, 'attachment')} copied")
    if missing := report.attachments_missing:
        files.append(f"{_plural(missing, 'attachment')} not in the export")
    if files:
        say("  " + " · ".join(files))
    if report.kept_edits:
        say(
            f"  [yellow]{_plural(len(report.kept_edits), 'note')} you edited "
            f"{'has' if len(report.kept_edits) == 1 else 'have'} newer messages in this export.[/] "
            "Your version was kept (--force replaces it):"
        )
        for path in report.kept_edits[:5]:
            say(f"    {escape(path)}")
        if len(report.kept_edits) > 5:
            say(f"    … and {len(report.kept_edits) - 5} more")
    if report.edited_current:
        n = report.edited_current
        say(f"  [dim]{_plural(n, 'note')} you edited {'was' if n == 1 else 'were'} left as is.[/]")
    if report.deleted_skipped:
        say(
            f"  [dim]{_plural(report.deleted_skipped, 'note')} you deleted "
            f"{'was' if report.deleted_skipped == 1 else 'were'} not recreated "
            "(--force brings them back).[/]"
        )
    if report.kept_files:
        names = escape(", ".join(report.kept_files))
        say(
            f"  [yellow]Left {names} alone[/]: changed since convoviz wrote it, or not "
            "convoviz's. Delete or rename it to get a fresh copy."
        )
    index = _file_link(settings.output / "index.md", "index.md")
    if settings.insights:
        charts = _file_link(settings.output / "insights.html", "insights.html")
        say(f"  Open {charts} for your charts, or browse {index}.")
    else:
        say(f"  Start with {index}.")
    if report.created and report.created == report.total_notes:
        say("  [dim]To read it in Obsidian: Open folder as vault, then choose this folder.[/]")
    if diagnostics.skipped:
        n = len(diagnostics.skipped)
        say(f"  [dim]Skipped {n:,} empty or unusable record{'' if n == 1 else 's'}.[/]")
    _messages(say, "  [yellow]warning:[/]", diagnostics.warnings, 5)
    _messages(say, "  [red]error:[/]", diagnostics.errors, 10)
    if diagnostics.unknown or diagnostics.errors:
        kinds = escape(", ".join(sorted(diagnostics.unknown)[:6]))
        if kinds:
            say(f"  [dim]Content convoviz doesn't know yet ({kinds}) was shown as plain text.[/]")
        say(
            "  [dim]If something looks wrong, run `convoviz inspect --json <export>` and share "
            f"the output (it contains no personal data) at {ISSUES_URL}[/]"
        )


def _messages(say: Callable[[str], None], label: str, messages: list[str], limit: int) -> None:
    for message in messages[:limit]:
        say(f"{label} {escape(message)}")
    if len(messages) > limit:
        say(f"{label} … and {len(messages) - limit:,} more (run with -v to see them all)")


def main() -> None:
    """Entry point for the ``convoviz`` script."""
    result = app()
    sys.exit(result if isinstance(result, int) else 0)
