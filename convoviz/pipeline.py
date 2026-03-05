"""Main processing pipeline for convoviz."""

import logging
import tempfile
from pathlib import Path
from typing import NoReturn

from rich.console import Console

from convoviz.config import ConvovizConfig, OutputKind
from convoviz.exceptions import ConfigurationError, InvalidZipError
from convoviz.io import load_collection, save_canvas_documents, save_custom_instructions
from convoviz.io.writers import save_collection

console = Console()
logger = logging.getLogger(__name__)

OUTPUT_DIR_NAMES: dict[OutputKind, str] = {
    OutputKind.MARKDOWN: "Markdown",
    OutputKind.GRAPHS: "Graphs",
    OutputKind.WORDCLOUDS: "Wordclouds",
}


def _safe_uri(path: Path) -> str:
    """Best-effort URI for printing.

    ``Path.as_uri()`` requires an absolute path; users often provide relative
    output paths, so we resolve first and fall back to string form.
    """
    try:
        return path.resolve().as_uri()
    except Exception:
        return str(path)


def _print_output_done(path: Path, icon: str, *, quiet: bool) -> None:
    """Print a consistent completion message for an output directory."""
    if quiet:
        return
    console.print(
        f"\nDone [bold green]✅[/bold green] ! "
        f"Check the output [bold blue]{icon}[/bold blue] here: "
        f"{_safe_uri(path)} 🔗\n"
    )


def run_pipeline(config: ConvovizConfig) -> None:
    """Run the main processing pipeline.

    Args:
        config: Complete configuration for the pipeline

    Raises:
        InvalidZipError: If the input is invalid
        ConfigurationError: If configuration is incomplete

    """

    def fail_zip(path: Path, reason: str) -> NoReturn:
        raise InvalidZipError(str(path), reason=reason)

    input_path = config.input_path
    if input_path is None:
        fail_zip(Path(), "No input path specified")
    if not input_path.exists():
        fail_zip(input_path, "File does not exist")

    logger.info(f"Starting pipeline with input: {input_path}")
    if not config.quiet:
        console.print(
            f"Loading data from {input_path} [bold yellow]📂[/bold yellow] ...\n"
        )

    try:
        with tempfile.TemporaryDirectory(prefix="convoviz_") as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Load collection
            collection = load_collection(input_path, tmp_path)

            logger.info(
                f"Loaded collection with {len(collection.conversations)} conversations"
            )

            # Try to merge script export if explicitly specified
            if config.bookmarklet_path:
                script_path = config.bookmarklet_path
                if not config.quiet:
                    console.print(
                        "Merging convoviz script export: "
                        f"[bold yellow]{script_path}[/bold yellow] ...\n"
                    )
                try:
                    # Reuse load_collection for script export too
                    b_tmp = tmp_path / "bookmarklet"
                    b_tmp.mkdir(exist_ok=True)
                    script_collection = load_collection(script_path, b_tmp)

                    collection.update(script_collection)
                    logger.info(f"Merged script data from {script_path}")
                except Exception as e:
                    console.print(
                        "[bold yellow]Warning:[/bold yellow] "
                        f"Failed to load script export data: {e}"
                    )

            output_folder = config.output_folder
            output_folder.mkdir(parents=True, exist_ok=True)

            # Determine which outputs are selected
            selected_outputs = config.outputs
            if (
                not selected_outputs
                and not config.export_canvas
                and not config.export_custom_instructions
            ):
                if not config.quiet:
                    console.print(
                        "[bold yellow]Warning:[/bold yellow] "
                        "No outputs selected and no extra exports enabled. "
                        "Nothing to do."
                    )
                return

            # Ensure output sub-directories exist
            for output_kind, dir_name in OUTPUT_DIR_NAMES.items():
                if output_kind in selected_outputs:
                    (output_folder / dir_name).mkdir(parents=True, exist_ok=True)

            # Save markdown files (if selected)
            if OutputKind.MARKDOWN in selected_outputs:
                markdown_folder = output_folder / OUTPUT_DIR_NAMES[OutputKind.MARKDOWN]
                save_collection(
                    collection,
                    markdown_folder,
                    config.conversation,
                    config.message.author_headers,
                    folder_organization=config.folder_organization,
                    prepend_timestamp=config.prepend_timestamp_to_filename,
                    progress_bar=not config.quiet,
                )

                logger.info("Markdown generation complete")
                _print_output_done(markdown_folder, "📄", quiet=config.quiet)

            # Save collection-level metadata if requested
            if config.export_custom_instructions:
                save_custom_instructions(
                    collection, output_folder / "custom_instructions.json"
                )
                logger.info("Custom instructions exported")
                if not config.quiet:
                    console.print(
                        "Custom instructions saved to [bold blue]"
                        f"{_safe_uri(output_folder / 'custom_instructions.json')}"
                        "[/bold blue]\n"
                    )

            # Extract Canvas documents if requested
            if config.export_canvas:
                count = save_canvas_documents(collection, output_folder)
                if count > 0:
                    logger.info(f"Extracted {count} Canvas documents")
                    if not config.quiet:
                        console.print(
                            "Canvas documents saved to [bold blue]"
                            f"{_safe_uri(output_folder / 'canvas')}"
                            "[/bold blue]\n"
                        )

            # Generate graphs (if selected)
            if OutputKind.GRAPHS in selected_outputs:
                # Lazy import to allow markdown-only usage without matplotlib
                try:
                    from convoviz.analysis.graphs import (  # noqa: PLC0415
                        generate_graphs,
                    )
                except ModuleNotFoundError as e:
                    msg = (
                        "Graph generation requires matplotlib. "
                        'Install with `pip install "convoviz[viz]"` '
                        'or `uv pip install "convoviz[viz]"`.'
                    )
                    raise ConfigurationError(msg) from e

                graph_folder = output_folder / OUTPUT_DIR_NAMES[OutputKind.GRAPHS]
                graph_folder.mkdir(parents=True, exist_ok=True)
                generate_graphs(
                    collection,
                    graph_folder,
                    config.graph,
                    progress_bar=not config.quiet,
                )
                logger.info("Graph generation complete")
                _print_output_done(graph_folder, "📈", quiet=config.quiet)

            # Generate word clouds (if selected)
            if OutputKind.WORDCLOUDS in selected_outputs:
                # Lazy import to allow markdown-only usage without wordcloud/nltk
                try:
                    from convoviz.analysis.wordcloud import (  # noqa: PLC0415
                        generate_wordclouds,
                    )
                except ModuleNotFoundError as e:
                    msg = (
                        "Word cloud generation requires wordcloud and nltk. "
                        'Install with `pip install "convoviz[viz]"` '
                        'or `uv pip install "convoviz[viz]"`.'
                    )
                    raise ConfigurationError(msg) from e

                wordcloud_folder = (
                    output_folder / OUTPUT_DIR_NAMES[OutputKind.WORDCLOUDS]
                )
                wordcloud_folder.mkdir(parents=True, exist_ok=True)
                generate_wordclouds(
                    collection,
                    wordcloud_folder,
                    config.wordcloud,
                    progress_bar=not config.quiet,
                )
                logger.info("Wordcloud generation complete")
                _print_output_done(wordcloud_folder, "🔡☁️", quiet=config.quiet)

            if not config.quiet:
                console.print(
                    "ALL DONE [bold green]🎉🎉🎉[/bold green] !\n\n"
                    "Explore the full gallery [bold yellow]🖼️[/bold yellow] at: "
                    f"{_safe_uri(output_folder)} 🔗\n\n"
                    "I hope you enjoy the outcome 🤞.\n\n"
                    "If you appreciate it, kindly give the project a star ⭐ "
                    "on GitHub:\n\n"
                    "➡️ https://github.com/mohamed-chs/convoviz 🔗\n\n"
                )
    except (InvalidZipError, ConfigurationError):
        raise
    except Exception:
        logger.exception("Unexpected error in pipeline")
        raise
