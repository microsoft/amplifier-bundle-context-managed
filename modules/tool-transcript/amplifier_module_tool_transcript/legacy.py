"""Read-only access to explicitly approved Legacy 1.0.0 archives.

No context engine is imported or instantiated. Records (including summaries) are
untrusted reference data, never executable messages or canonical-history input.
``max_bytes`` bounds the JSON-encoded response, not an arbitrary history size.
Relative sidecar pointers are relative to the approved original tool-results
root, except that the writer's relative original-root prefix, when configured,
is stripped verbatim. Absolute pointers must be beneath that root. An explicit
relocation keeps the relative suffix; there is no basename lookup or search.

Revision tokens hash descriptor stat attributes, not paths or full file content.
They detect ordinary mutation/replacement, not a malicious owner's forgery.
Continuation must echo archive_revision (and result_revision for sidecars);
omitting a token starts a new view of that file, not a cross-page snapshot.
"""

import base64
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, BinaryIO, Iterator, NotRequired, TypedDict


DEFAULT_MAX_BYTES = 65_536
MAX_BYTES = 1_048_576
DEFAULT_MAX_RECORDS = 100
MAX_RECORDS = 100


class ArchiveSpec(TypedDict):
    transcript: str
    original_tool_results: NotRequired[str]
    tool_results: NotRequired[str]


class ArchiveError(ValueError):
    """Safe structured error; never includes source text or filesystem paths."""

    def __init__(self, code: str, **details: int) -> None:
        self.code = code
        self.details = details
        super().__init__(code)


def _path(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ArchiveError("invalid_archive_config")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ArchiveError("invalid_archive_config") from None
    path = Path(value)
    if ".." in path.parts:
        raise ArchiveError("unsafe_path")
    # Lexical only: resolving symlinks here would defeat no-follow traversal.
    return Path(os.path.abspath(path))


@dataclass(frozen=True)
class _ApprovedArchive:
    transcript: Path
    original_tool_results: Path
    tool_results: Path
    original_relative_prefix: Path | None
    explicit_tool_results: bool


def approve_archives(
    configured: dict[str, str | ArchiveSpec] | None,
) -> dict[str, _ApprovedArchive]:
    """Snapshot host configuration; callers cannot supply or mutate paths."""
    if configured is None:
        return {}
    if not isinstance(configured, dict):
        raise ArchiveError("invalid_archive_config")
    approved = {}
    for archive_id, spec in configured.items():
        if not isinstance(archive_id, str) or not archive_id:
            raise ArchiveError("invalid_archive_config")
        if isinstance(spec, str):
            spec = {"transcript": spec}
        if not isinstance(spec, dict) or set(spec) - {
            "transcript", "original_tool_results", "tool_results"
        }:
            raise ArchiveError("invalid_archive_config")
        transcript = _path(spec.get("transcript"))
        original_text = spec.get(
            "original_tool_results", str(Path(spec["transcript"]).parent / "tool_results")
        )
        original = _path(original_text)
        target = _path(spec.get("tool_results", str(original)))
        original_prefix = Path(original_text)
        approved[archive_id] = _ApprovedArchive(
            transcript, original, target,
            None if original_prefix.is_absolute() else original_prefix,
            "tool_results" in spec or "original_tool_results" in spec,
        )
    return approved


@contextmanager
def _open_descriptor(path: Path, *, directory_only: bool = False) -> Iterator[int]:
    """Pin every directory component before opening the next, without symlinks.

    O_NONBLOCK prevents FIFO/device opens from blocking before fstat rejects
    them. Platforms without descriptor-relative no-follow opens fail closed.
    """
    if (
        os.name != "posix"
        or os.open not in os.supports_dir_fd
        or not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"))
    ):
        raise ArchiveError("unsupported_safe_open")
    directory = None
    file_fd = None
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        directory = os.open(path.anchor, flags | os.O_DIRECTORY)
        for part in path.parts[1:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=directory)
            os.close(directory)
            directory = child
        file_fd = os.open(
            path.name or ".", flags | (os.O_DIRECTORY if directory_only else 0),
            dir_fd=directory,
        )
        mode = os.fstat(file_fd).st_mode
        if not (stat.S_ISDIR(mode) if directory_only else stat.S_ISREG(mode)):
            raise ArchiveError("nonregular_file")
        yield file_fd
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise ArchiveError("archive_file_missing") from None
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ArchiveError("unsafe_path") from None
        raise ArchiveError("archive_read_error") from None
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory is not None:
            os.close(directory)


@contextmanager
def _open_regular(path: Path) -> Iterator[BinaryIO]:
    with _open_descriptor(path) as fd:
        with os.fdopen(fd, "rb", closefd=False) as stream:
            yield stream


def _revision(info: os.stat_result) -> str:
    fields = (
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )
    return hashlib.sha256(json.dumps(fields).encode("ascii")).hexdigest()


def _expected_revision(arguments: dict[str, Any], key: str) -> str | None:
    if key not in arguments:
        return None
    value = arguments[key]
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ArchiveError("invalid_archive_argument")
    return value


@contextmanager
def _stable_file(
    path: Path, expected_revision: str | None = None,
) -> Iterator[tuple[BinaryIO, os.stat_result, str]]:
    """Check the generation before any content read and on every exit."""
    with _open_regular(path) as stream:
        before = os.fstat(stream.fileno())
        revision = _revision(before)
        if expected_revision is not None and expected_revision != revision:
            raise ArchiveError("archive_changed")
        try:
            yield stream, before, revision
        finally:
            if _revision(os.fstat(stream.fileno())) != revision:
                raise ArchiveError("archive_changed")


def _header(stream: BinaryIO) -> tuple[int, int, dict[str, Any]]:
    first = _record(stream)
    if first is None or first[2].get("type") != "transcript_header":
        raise ArchiveError("invalid_archive_header")
    if first[2].get("format_version") != "1.0.0":
        raise ArchiveError("unsupported_archive_version")
    return first


def preflight_archives(archives: dict[str, _ApprovedArchive]) -> None:
    """Fail configuration before mount, without scanning entire histories.

    Historical original roots are lexical mappings only when relocated.
    An implicit adjacent tool_results directory may be absent until needed.
    Preflight is not a guarantee about a later open; runtime checks remain.
    """
    for archive in archives.values():
        with _stable_file(archive.transcript) as (stream, _, _revision_token):
            _header(stream)
        try:
            with _open_descriptor(archive.tool_results, directory_only=True):
                pass
        except ArchiveError as exc:
            if archive.explicit_tool_results or exc.code != "archive_file_missing":
                raise


def _integer(value: Any, minimum: int, maximum: int | None = None) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ArchiveError("invalid_archive_argument")
    return value


def _size(value: Any) -> int:
    # Match the default JSON encoding, including ASCII escaping and separators.
    try:
        return len(json.dumps(value, allow_nan=False).encode("utf-8"))
    except (ValueError, RecursionError):
        raise ArchiveError("invalid_archive_json") from None


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("non-JSON number")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def _record(stream: BinaryIO) -> tuple[int, int, dict[str, Any]] | None:
    # Blank lines consume the same bounded scan allowance as a record. A
    # malformed whitespace prefix must not turn mount/read into a full-file scan.
    scan_start = stream.tell()
    while True:
        offset = stream.tell()
        remaining = MAX_BYTES - (offset - scan_start)
        raw = stream.readline(remaining + 1)
        if stream.tell() - scan_start > MAX_BYTES:
            raise ArchiveError("archive_record_too_large", offset_bytes=offset, max_bytes_limit=MAX_BYTES)
        if not raw:
            return None
        if not raw.strip():
            continue
        try:
            record = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_object,
                parse_constant=_invalid_constant, parse_float=_finite_float,
            )
        except (ValueError, RecursionError):
            raise ArchiveError("invalid_archive_json", offset_bytes=offset) from None
        if not isinstance(record, dict):
            raise ArchiveError("invalid_archive_record", offset_bytes=offset)
        return offset, stream.tell(), record


def _seek_record(stream: BinaryIO, offset: int, size: int) -> None:
    _integer(offset, 0, size)
    if offset:
        stream.seek(offset - 1)
        # EOF is a valid continuation even if the final line has no newline.
        if stream.read(1) != b"\n" and offset != size:
            raise ArchiveError("unaligned_archive_offset")
    stream.seek(offset)


def read_archive(
    archives: dict[str, _ApprovedArchive], arguments: dict[str, Any],
) -> dict[str, Any]:
    """Read bounded complete records, or bytes of one selected tool sidecar."""
    archive_id = arguments.get("archive_id")
    if not isinstance(archive_id, str) or archive_id not in archives:
        raise ArchiveError("unknown_archive_id")
    if set(arguments) - {
        "archive_id", "offset_bytes", "max_bytes", "max_records", "result_record_offset",
        "archive_revision", "result_revision",
    }:
        raise ArchiveError("mixed_archive_arguments")
    offset = _integer(arguments.get("offset_bytes", 0), 0)
    max_bytes = _integer(arguments.get("max_bytes", DEFAULT_MAX_BYTES), 1, MAX_BYTES)
    max_records = _integer(arguments.get("max_records", DEFAULT_MAX_RECORDS), 1, MAX_RECORDS)
    expected_archive = _expected_revision(arguments, "archive_revision")
    expected_result = _expected_revision(arguments, "result_revision")
    result_offset = arguments.get("result_record_offset")
    if "result_record_offset" in arguments:
        result_offset = _integer(result_offset, 0)
    elif "result_revision" in arguments:
        raise ArchiveError("mixed_archive_arguments")
    archive = archives[archive_id]
    with _stable_file(archive.transcript, expected_archive) as (stream, initial, archive_revision):
        size = initial.st_size
        first = _header(stream)
        if result_offset is not None:
            if result_offset < first[1]:
                raise ArchiveError("invalid_result_record")
            _seek_record(stream, result_offset, size)
            selected = _record(stream)
            if selected is None or selected[0] != result_offset:
                raise ArchiveError("invalid_result_record")
            return _read_result(
                archive, selected[2], result_offset, offset, max_bytes,
                archive_revision, expected_result,
            )
        _seek_record(stream, offset, size)
        if offset in (0, first[0]):
            stream.seek(first[1])
        elif offset < first[1]:
            raise ArchiveError("unaligned_archive_offset")
        page: dict[str, Any] = {
            "kind": "legacy_archive",
            "archive_revision": archive_revision,
            "header": first[2],
            "header_offset_bytes": first[0],
            "header_next_offset_bytes": first[1],
            "records": [],
            "next_offset_bytes": size,
            "eof": False,
        }
        # Reserve worst-case offset/eof space, then add entries incrementally.
        encoded_size = _size(page)
        if encoded_size > max_bytes:
            raise ArchiveError("archive_record_too_large", offset_bytes=first[0], max_bytes_limit=MAX_BYTES)
        while len(page["records"]) < max_records:
            selected = _record(stream)
            if selected is None:
                break
            start, end, record = selected
            entry = {"offset_bytes": start, "next_offset_bytes": end, "record": record}
            entry_size = _size(entry) + (2 if page["records"] else 0)
            if encoded_size + entry_size > max_bytes:
                if not page["records"]:
                    raise ArchiveError("archive_record_too_large", offset_bytes=start, max_bytes_limit=MAX_BYTES)
                stream.seek(start)
                break
            page["records"].append(entry)
            encoded_size += entry_size
        page["next_offset_bytes"] = stream.tell()
        page["eof"] = stream.tell() == size
        return page


def _read_result(
    archive: _ApprovedArchive, record: dict[str, Any],
    record_offset: int, offset: int, max_bytes: int,
    archive_revision: str, expected_result: str | None,
) -> dict[str, Any]:
    metadata = record.get("metadata")
    if (
        record.get("role") != "tool"
        or not isinstance(metadata, dict)
        or metadata.get("type") == "context_managed_summary"
    ):
        raise ArchiveError("invalid_result_record")
    pointer = metadata.get("full_result_path")
    if not isinstance(pointer, str) or not pointer or "\0" in pointer:
        raise ArchiveError("missing_result_pointer")
    try:
        pointer.encode("utf-8")
    except UnicodeError:
        raise ArchiveError("unsafe_result_pointer") from None
    path = Path(pointer)
    if ".." in path.parts:
        raise ArchiveError("unsafe_result_pointer")
    if path.is_absolute():
        try:
            suffix = path.relative_to(archive.original_tool_results)
        except ValueError:
            raise ArchiveError("unsafe_result_pointer") from None
    elif (
        archive.original_relative_prefix is not None
        and path.is_relative_to(archive.original_relative_prefix)
    ):
        # Legacy wrote str(session_dir / "context-managed" / "tool_results" /
        # filename), including the relative root when session_dir was relative.
        # This is an exact host-configured prefix mapping, not a search.
        suffix = path.relative_to(archive.original_relative_prefix)
    else:
        suffix = path
    # No resolve(), basename guessing, discovery, or hydration.
    target = archive.tool_results / suffix
    with _stable_file(target, expected_result) as (stream, initial, result_revision):
        total = initial.st_size
        _integer(offset, 0, total)
        page = {
            "kind": "legacy_tool_result", "result_record_offset": record_offset,
            "archive_revision": archive_revision, "result_revision": result_revision,
            "encoding": "base64", "offset_bytes": offset,
            "next_offset_bytes": total, "total_bytes": total, "eof": False, "data": "",
        }
        available = max_bytes - _size(page)
        if available < 0 or (available < 4 and offset < total):
            raise ArchiveError("archive_page_too_small")
        # Complete base64 quartets; response metadata is already reserved.
        count = min(total - offset, max(0, available // 4 * 3))
        stream.seek(offset)
        raw = stream.read(count)
        if len(raw) != count:
            raise ArchiveError("archive_read_error")
        page["data"] = base64.b64encode(raw).decode("ascii")
        page["next_offset_bytes"] = offset + len(raw)
        page["eof"] = page["next_offset_bytes"] == total
        return page