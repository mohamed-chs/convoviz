"""Loading functions for conversations and collections."""

import logging
import re
import shutil
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from orjson import loads

from convoviz.exceptions import InvalidZipError
from convoviz.models import ConversationCollection

logger = logging.getLogger(__name__)

_SPLIT_FILE_RE = re.compile(r"^conversations-\d+\.json$")


def _is_safe_zip_member_name(name: str) -> bool:
    r"""Return True if a ZIP entry name is safe to extract.

    This is intentionally OS-agnostic: it treats both ``/`` and ``\\`` as path
    separators and rejects absolute paths, drive-letter paths, and ``..`` parts.
    """
    normalized = name.replace("\\", "/")
    member_path = PurePosixPath(normalized)

    # Absolute paths (e.g. "/etc/passwd") or empty names
    if not normalized or member_path.is_absolute():
        return False

    # Windows drive letters / UNC-style prefixes stored in the archive
    first = member_path.parts[0] if member_path.parts else ""
    if first.endswith(":") or first.startswith(("//", "\\\\")):
        return False

    return ".." not in member_path.parts


def extract_archive(filepath: Path, target_dir: Path) -> Path:
    """Extract a ZIP file to the target folder.

    Includes safety checks to prevent Path Traversal (Zip-Slip).

    Args:
        filepath: Path to the ZIP file
        target_dir: Path where to extract

    Returns:
        Path to the extracted folder (target_dir)

    Raises:
        InvalidZipError: If extraction fails or a security risk is detected

    """
    logger.info(f"Extracting archive: {filepath} to {target_dir}")
    root = target_dir.resolve()

    with ZipFile(filepath) as zf:
        for member in zf.infolist():
            if not _is_safe_zip_member_name(member.filename):
                raise InvalidZipError(
                    str(filepath), reason=f"Malicious path in ZIP: {member.filename}"
                )

            # Reject symlink entries to prevent extraction-then-traversal attacks.
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise InvalidZipError(
                    str(filepath),
                    reason=f"Symlinks are not allowed in ZIP: {member.filename}",
                )

            # Additional check using resolved paths
            normalized = member.filename.replace("\\", "/")
            target_path = (target_dir / normalized).resolve()
            if not target_path.is_relative_to(root):
                raise InvalidZipError(
                    str(filepath), reason=f"Malicious path in ZIP: {member.filename}"
                )

            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return target_dir


def _find_conversation_files(directory: Path) -> list[Path]:
    """Find conversation JSON files in a directory.

    Checks for single-file format (``conversations.json``) first, then falls
    back to the split format (``conversations-000.json``, etc.).

    Returns:
        Sorted list of conversation file paths, or empty list if none found.

    """
    single = directory / "conversations.json"
    if single.exists():
        return [single]

    return sorted(
        p for p in directory.iterdir() if p.is_file() and _SPLIT_FILE_RE.match(p.name)
    )


def _has_conversation_entries(namelist: list[str]) -> bool:
    """Check whether a ZIP namelist contains conversation data files."""
    if "conversations.json" in namelist:
        return True
    return any(_SPLIT_FILE_RE.match(name) for name in namelist)


def validate_zip(filepath: Path) -> bool:
    """Check if a ZIP file contains conversation data.

    Accepts both single-file (``conversations.json``) and split-file
    (``conversations-NNN.json``) formats.

    Args:
        filepath: Path to the ZIP file

    Returns:
        True if valid, False otherwise

    """
    if not filepath.is_file() or filepath.suffix.lower() != ".zip":
        return False
    try:
        with ZipFile(filepath) as zf:
            return _has_conversation_entries(zf.namelist())
    except Exception:
        return False


def load_collection_from_json(filepath: Path | str) -> ConversationCollection:
    """Load a conversation collection from a JSON file.

    The JSON file should contain an array of conversation objects,
    or an object with a "conversations" key.

    Args:
        filepath: Path to the JSON file

    Returns:
        Loaded ConversationCollection object

    """
    filepath = Path(filepath)
    logger.debug(f"Loading collection from JSON: {filepath}")
    with filepath.open(encoding="utf-8") as f:
        data = loads(f.read())

    # Handle case where export is wrapped in a top-level object
    if isinstance(data, dict) and "conversations" in data:
        data = data["conversations"]

    return ConversationCollection(conversations=data, source_paths=[filepath.parent])


def _load_from_conversation_files(directory: Path) -> ConversationCollection:
    """Load and merge conversation files from a directory.

    Args:
        directory: Directory containing conversation JSON file(s)

    Returns:
        Merged ConversationCollection

    Raises:
        InvalidZipError: If no conversation files are found

    """
    files = _find_conversation_files(directory)
    if not files:
        raise InvalidZipError(str(directory), reason="missing conversation data")

    collection = load_collection_from_json(files[0])
    for extra in files[1:]:
        collection.update(load_collection_from_json(extra))
    return collection


def load_collection_from_zip(
    filepath: Path | str, target_dir: Path
) -> ConversationCollection:
    """Load a conversation collection from a ChatGPT export ZIP file.

    Supports both single-file and split-file conversation formats.

    Args:
        filepath: Path to the ZIP file
        target_dir: Directory to extract the ZIP into

    Returns:
        Loaded ConversationCollection object

    Raises:
        InvalidZipError: If the ZIP file is invalid or missing conversation data

    """
    filepath = Path(filepath)

    if not validate_zip(filepath):
        raise InvalidZipError(str(filepath))

    extract_archive(filepath, target_dir)
    return _load_from_conversation_files(target_dir)


def load_collection(input_path: Path, tmp_path: Path) -> ConversationCollection:
    """Load a conversation collection from a directory, JSON, or ZIP.

    Args:
        input_path: Path to the input (directory, JSON, or ZIP)
        tmp_path: Temporary directory for ZIP extraction

    Returns:
        Loaded ConversationCollection object

    """
    if input_path.is_dir():
        return _load_from_conversation_files(input_path)

    if input_path.suffix.lower() == ".json":
        return load_collection_from_json(input_path)

    return load_collection_from_zip(input_path, tmp_path)


def find_latest_valid_zip(directory: Path | None = None) -> Path | None:
    """Find the most recent valid ChatGPT export ZIP in a directory.

    A valid ZIP is one that contains conversation data (either a single
    ``conversations.json`` or split ``conversations-NNN.json`` files).

    Args:
        directory: Directory to search (defaults to ~/Downloads)

    Returns:
        Path to the most recent valid ZIP, or None if none found

    """
    if directory is None:
        directory = Path.home() / "Downloads"

    zip_files = [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".zip"
    ]
    if not zip_files:
        return None

    valid = [p for p in zip_files if validate_zip(p)]
    if not valid:
        return None

    return max(valid, key=lambda p: p.stat().st_mtime)


def find_script_export(directory: Path | None = None) -> Path | None:
    """Find the most recent script-generated export in a directory.

    Looks for files starting with "convoviz_export" and using .json/.zip.

    Args:
        directory: Directory to search (defaults to ~/Downloads)

    Returns:
        Path to the most recent export, or None if none found

    """
    if directory is None:
        directory = Path.home() / "Downloads"

    export_files = [
        f
        for f in directory.iterdir()
        if f.is_file()
        and f.name.lower().startswith("convoviz_export")
        and f.suffix.lower() in (".json", ".zip")
    ]

    if not export_files:
        return None

    return max(export_files, key=lambda p: p.stat().st_mtime)
