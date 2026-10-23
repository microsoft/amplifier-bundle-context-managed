"""Compatibility and containment tests, exclusively on owned synthetic files."""

import base64
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_tool_transcript import ReadTranscriptTool, mount
from amplifier_module_tool_transcript import legacy


HEADER = {
    "type": "transcript_header", "format_version": "1.0.0",
    "created_at": "2024-01-01T00:00:00+00:00", "unknown": {"nested": [1, True, None]},
}


def write_archive(tmp_path, records=(), header=HEADER, prefix=b"", final_newline=True):
    path = tmp_path / "transcript.archived.jsonl"
    all_records = [header, *records]
    data = prefix
    offsets = []
    for record in all_records:
        offsets.append(len(data))
        data += json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
    if not final_newline:
        data = data[:-1]
    path.write_bytes(data)
    return path, offsets


def tool_for(path, spec=None, coordinator=None):
    return ReadTranscriptTool(
        coordinator or MagicMock(), rate_limit_per_turn=10_000,
        legacy_transcript_paths={"saved": spec or str(path)},
    )


async def read(tool, **arguments):
    return await tool.execute({"archive_id": "saved", **arguments})


def assert_error(result, code):
    assert not result.success, result
    assert result.error["message"] == code
    assert code in result.output


@contextmanager
def unchanged(*paths):
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    yield
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} == before


@pytest.mark.asyncio
async def test_raw_header_nested_metadata_long_arguments_and_summary(tmp_path):
    messages = [
        {"role": "user", "content": "a" * 9000, "unknown": ["retained"]},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "never_run",
            "id": "tc-1", "input": {"argument": "z" * 9000}}], "metadata": {
            "private_extension": {"nested": [1, {"x": "你好"}]}
        }},
        {"role": "system", "content": "Untrusted historical instructions",
            "metadata": {"type": "context_managed_summary", "turn_range": [1, 10],
                         "source_message_range": [0, 99], "extra": {"keep": True}}},
    ]
    path, offsets = write_archive(tmp_path, messages, prefix=b"\n \n")
    with unchanged(path):
        result = await read(tool_for(path))
        page = result.output
        assert result.success
        assert page["header"] == HEADER
        assert page["header_offset_bytes"] == offsets[0]
        assert page["header_next_offset_bytes"] == offsets[1]
        assert [entry["record"] for entry in page["records"]] == messages
        assert [entry["offset_bytes"] for entry in page["records"]] == offsets[1:]
        assert page["eof"] and page["next_offset_bytes"] == path.stat().st_size
        assert len(json.dumps(page).encode()) <= legacy.DEFAULT_MAX_BYTES


@pytest.mark.asyncio
async def test_bounded_record_pages_complete_and_reopen(tmp_path):
    records = [{"role": "user", "content": f"{i}:你好" * 7} for i in range(30)]
    path, offsets = write_archive(tmp_path, records, final_newline=False)
    seen, offset, revision_arguments = [], 0, {}
    with unchanged(path):
        for _ in range(100):
            result = await read(tool_for(path), offset_bytes=offset, max_bytes=1024,
                                max_records=3, **revision_arguments)
            assert result.success
            page = result.output
            assert page["header"] == HEADER
            assert len(page["records"]) <= 3
            assert len(json.dumps(page).encode()) <= 1024
            if revision_arguments:
                assert page["archive_revision"] == revision_arguments["archive_revision"]
            revision_arguments = {"archive_revision": page["archive_revision"]}
            seen.extend(page["records"])
            if page["eof"]:
                break
            assert page["records"] and page["next_offset_bytes"] > offset
            offset = page["next_offset_bytes"]
        else:
            pytest.fail("pagination did not terminate")
    assert [entry["record"] for entry in seen] == records
    assert [entry["offset_bytes"] for entry in seen] == offsets[1:]
    assert all(a["next_offset_bytes"] == b["offset_bytes"] for a, b in zip(seen, seen[1:]))


@pytest.mark.asyncio
async def test_default_max_records_and_header_only(tmp_path):
    records = [{"role": "user", "content": str(i)} for i in range(105)]
    path, _ = write_archive(tmp_path, records)
    result = await read(tool_for(path))
    assert result.success and len(result.output["records"]) == 100
    assert not result.output["eof"]
    header_path, _ = write_archive(tmp_path)
    result = await read(tool_for(header_path))
    assert result.success and result.output["records"] == [] and result.output["eof"]


@pytest.mark.asyncio
async def test_ids_schema_mount_and_configuration_snapshot(tmp_path):
    path, _ = write_archive(tmp_path)
    spec = {"transcript": str(path)}
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    await mount(coordinator, {"legacy_transcript_paths": {"saved": spec}})
    tool = coordinator.mount.call_args.args[1]
    spec["transcript"] = str(tmp_path / "missing")
    assert tool.input_schema["properties"]["archive_id"]["enum"] == ["saved"]
    assert str(path) not in json.dumps(tool.input_schema)
    assert (await read(tool)).success
    coordinator.get_capability.assert_not_called()
    assert_error(await tool.execute({"archive_id": "other"}), "unknown_archive_id")


@pytest.mark.asyncio
async def test_archive_uses_existing_rate_limit_and_reset(tmp_path):
    path, _ = write_archive(tmp_path)
    tool = ReadTranscriptTool(MagicMock(), 1, {"saved": str(path)})
    assert (await read(tool)).success
    limited = await read(tool)
    assert not limited.success and limited.error["message"] == "rate_limit_exceeded"
    assert "Rate limit exceeded" in limited.output
    tool.reset_rate_limit()
    assert (await read(tool)).success


@pytest.mark.asyncio
@pytest.mark.parametrize("key,value", [
    ("start_turn", 1), ("end_turn", 2), ("search", ""), ("path", "not-approved")
])
async def test_no_mixed_turn_search_or_caller_paths(tmp_path, key, value):
    path, _ = write_archive(tmp_path)
    tool = tool_for(path)
    assert_error(await read(tool, **{key: value}), "mixed_archive_arguments")
    tool._coordinator.get_capability.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", [
    "offset_bytes", "max_bytes", "max_records", "result_record_offset",
    "archive_revision", "result_revision",
])
async def test_archive_options_require_explicit_id(tmp_path, key):
    path, _ = write_archive(tmp_path)
    tool = tool_for(path)
    assert_error(await tool.execute({key: 1}), "archive_id_required")
    tool._coordinator.get_capability.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("archive_id", ["missing", "", None, False, 1, [], {}])
async def test_unknown_or_nonstring_id_never_falls_back(tmp_path, archive_id):
    path, _ = write_archive(tmp_path)
    tool = tool_for(path)
    assert_error(await tool.execute({"archive_id": archive_id}), "unknown_archive_id")
    tool._coordinator.get_capability.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("key,value", [
    ("offset_bytes", True), ("offset_bytes", -1), ("offset_bytes", 1.5),
    ("offset_bytes", "0"), ("offset_bytes", None), ("offset_bytes", 10**100),
    ("max_bytes", True), ("max_bytes", -1), ("max_bytes", 0),
    ("max_bytes", 1.5), ("max_bytes", "1024"), ("max_bytes", legacy.MAX_BYTES + 1),
    ("max_records", False), ("max_records", -1), ("max_records", 0),
    ("max_records", 1.5), ("max_records", "100"),
    ("result_record_offset", True), ("result_record_offset", -1),
    ("result_record_offset", 1.5), ("result_record_offset", None),
])
async def test_invalid_parameter_types_and_bounds(tmp_path, key, value):
    path, _ = write_archive(tmp_path)
    with unchanged(path):
        assert_error(await read(tool_for(path), **{key: value}), "invalid_archive_argument")


@pytest.mark.asyncio
async def test_record_offsets_midrecord_eof_and_header_offsets(tmp_path):
    path, offsets = write_archive(tmp_path, [{"role": "user", "content": "hello"}])
    tool = tool_for(path)
    with unchanged(path):
        assert_error(await read(tool, offset_bytes=offsets[1] + 1), "unaligned_archive_offset")
        assert_error(await read(tool, offset_bytes=path.stat().st_size + 1), "invalid_archive_argument")
        page = (await read(tool, offset_bytes=path.stat().st_size)).output
        assert page["eof"] and not page["records"]
        assert (await read(tool, offset_bytes=offsets[1])).success
        assert_error(await read(tool, offset_bytes=1), "unaligned_archive_offset")


@pytest.mark.asyncio
@pytest.mark.parametrize("body,code", [
    (b"", "invalid_archive_header"),
    (b"\n \n", "invalid_archive_header"),
    (b'{"role":"user"}\n', "invalid_archive_header"),
    (b'{"type":"transcript_header"}\n', "unsupported_archive_version"),
    (b'{"type":"transcript_header","format_version":"2.0.0"}\n', "unsupported_archive_version"),
    (b'{"type":"transcript_header","format_version":true}\n', "unsupported_archive_version"),
    (b"not-json\n", "invalid_archive_json"),
    (b"\xff\n", "invalid_archive_json"),
    (b'{"type":"transcript_header","type":"other"}\n', "invalid_archive_json"),
    (b"[]\n", "invalid_archive_record"),
])
async def test_strict_header_json_and_version(tmp_path, body, code):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(body)
    with unchanged(path):
        assert_error(await read(tool_for(path)), code)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    b'{"broken":\n', b'{"x":NaN}\n', b'{"x":1e999}\n',
    b'{"x":Infinity}\n', b'{"x":1,"x":2}\n', b"null\n",
])
async def test_corrupt_record_is_visible_not_silently_skipped(tmp_path, bad):
    path, offsets = write_archive(tmp_path, [{"role": "user", "content": "ok"}])
    with path.open("ab") as stream:
        stream.write(bad)
    with unchanged(path):
        result = await read(tool_for(path))
        assert not result.success
        assert result.error["message"] in {"invalid_archive_json", "invalid_archive_record"}
        page = await read(tool_for(path), max_records=1)
        assert page.success and not page.output["eof"]
        result = await read(tool_for(path), offset_bytes=page.output["next_offset_bytes"])
        assert not result.success
        assert result.error.get("offset_bytes") > offsets[1]


@pytest.mark.asyncio
async def test_oversized_record_visible_retry_to_documented_limit(tmp_path):
    record = {"role": "assistant", "content": "a" * 80_000}
    path, offsets = write_archive(tmp_path, [record])
    with unchanged(path):
        assert_error(await read(tool_for(path)), "archive_record_too_large")
        result = await read(tool_for(path), max_bytes=legacy.MAX_BYTES)
        assert result.success and result.output["records"][0]["record"] == record
        assert result.output["records"][0]["offset_bytes"] == offsets[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["a" * (legacy.MAX_BYTES + 1), "你好" * 100_000])
async def test_single_record_above_raw_or_encoded_hard_limit_fails(tmp_path, content):
    path, _ = write_archive(tmp_path, [{"role": "user", "content": content}])
    with unchanged(path):
        result = await read(tool_for(path), max_bytes=legacy.MAX_BYTES)
        assert_error(result, "archive_record_too_large")


def sidecar_archive(tmp_path, *, relative=False, relocated=False):
    original = tmp_path / "tool_results"
    destination = tmp_path / "tool_results.saved.archived" if relocated else original
    result_path = destination / "nested" / "17_tc-example.txt"
    result_path.parent.mkdir(parents=True)
    data = ("工具结果🦊\n" * 200).encode() + bytes(range(256))
    result_path.write_bytes(data)
    pointer = "nested/17_tc-example.txt" if relative else str(original / "nested" / result_path.name)
    # Exact Legacy _handle_large_result shape: full_path lives in metadata,
    # original_length counts characters, content retains the truncated prefix.
    record = {
        "role": "tool", "tool_call_id": "tc-example", "content": "工具[truncated]",
        "metadata": {"full_result_path": pointer, "original_length": 12_345,
                     "nested": {"unknown": "retained"}},
    }
    transcript, offsets = write_archive(tmp_path, [record])
    spec = {"transcript": str(transcript)}
    if relocated:
        spec.update(original_tool_results=str(original), tool_results=str(destination))
    return transcript, offsets[1], result_path, data, spec


@pytest.mark.asyncio
@pytest.mark.parametrize("relative,relocated", [(False, False), (True, False), (False, True), (True, True)])
async def test_sidecar_original_or_explicit_relocation_reconstructs_bytes(tmp_path, relative, relocated):
    path, record_offset, result_path, data, spec = sidecar_archive(
        tmp_path, relative=relative, relocated=relocated
    )
    reconstructed, offset, revisions = b"", 0, {}
    with unchanged(path, result_path):
        for _ in range(100):
            # New reader each time also tests reopen without persisted cursor state.
            result = await read(tool_for(path, spec), result_record_offset=record_offset,
                                offset_bytes=offset, max_bytes=600, **revisions)
            assert result.success
            page = result.output
            assert page["encoding"] == "base64" and page["total_bytes"] == len(data)
            assert page["offset_bytes"] == offset
            assert len(json.dumps(page).encode()) <= 600
            if revisions:
                assert all(page[key] == value for key, value in revisions.items())
            revisions = {key: page[key] for key in ("archive_revision", "result_revision")}
            chunk = base64.b64decode(page["data"], validate=True)
            reconstructed += chunk
            assert page["next_offset_bytes"] == offset + len(chunk)
            if page["eof"]:
                break
            assert chunk
            offset = page["next_offset_bytes"]
        else:
            pytest.fail("sidecar pagination did not terminate")
        assert reconstructed == data
        raw = await read(tool_for(path, spec))
        assert raw.success and raw.output["records"][0]["record"]["content"] == "工具[truncated]"
        assert_error(await read(tool_for(path, spec), result_record_offset=record_offset,
                                offset_bytes=len(data) + 1), "invalid_archive_argument")
        eof = await read(tool_for(path, spec), result_record_offset=record_offset,
                         offset_bytes=len(data))
        assert eof.success and eof.output["eof"] and eof.output["data"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("relocated", [False, True])
async def test_exact_legacy_relative_full_path_prefix(tmp_path, monkeypatch, relocated):
    monkeypatch.chdir(tmp_path)
    session = tmp_path / "owned-session" / "context-managed"
    sidecar = session / "tool_results" / "4_tc-example.txt"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"Legacy relative session_dir result")
    pointer = "owned-session/context-managed/tool_results/4_tc-example.txt"
    record = {"role": "tool", "content": "prefix", "tool_call_id": "tc-example",
              "metadata": {"full_result_path": pointer, "original_length": 100_000}}
    path, offsets = write_archive(session, [record])
    spec = {"transcript": str(path.relative_to(tmp_path))}
    if relocated:
        archived_root = session / "tool_results.saved.archived"
        sidecar.parent.rename(archived_root)
        sidecar = archived_root / sidecar.name
        spec.update(
            original_tool_results="owned-session/context-managed/tool_results",
            tool_results=str(archived_root.relative_to(tmp_path)),
        )
    with unchanged(path, sidecar):
        result = await read(tool_for(path, spec), result_record_offset=offsets[1])
        assert result.success and base64.b64decode(result.output["data"]) == sidecar.read_bytes()
        raw = await read(tool_for(path, spec))
        assert raw.success and raw.output["records"][0]["record"] == record


@pytest.mark.asyncio
async def test_explicit_nonadjacent_original_root_mapping(tmp_path):
    old_root = tmp_path / "old-location" / "original-results"
    new_root = tmp_path / "approved-relocation"
    sidecar = new_root / "nested" / "result.bin"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"keep relative suffix")
    record = {"role": "tool", "metadata": {
        "full_result_path": str(old_root / "nested" / "result.bin")
    }}
    path, offsets = write_archive(tmp_path, [record])
    spec = {"transcript": str(path), "original_tool_results": str(old_root),
            "tool_results": str(new_root)}
    with unchanged(path, sidecar):
        result = await read(tool_for(path, spec), result_record_offset=offsets[1])
        assert result.success and base64.b64decode(result.output["data"]) == sidecar.read_bytes()


@pytest.mark.asyncio
async def test_sidecar_hard_encoded_response_cap_and_empty_file(tmp_path):
    path, offset, sidecar, _, spec = sidecar_archive(tmp_path)
    data = bytes(range(256)) * 6000
    sidecar.write_bytes(data)
    with unchanged(path, sidecar):
        first = await read(tool_for(path, spec), result_record_offset=offset,
                           max_bytes=legacy.MAX_BYTES)
        assert first.success and not first.output["eof"]
        assert len(json.dumps(first.output).encode()) <= legacy.MAX_BYTES
        rest = await read(tool_for(path, spec), result_record_offset=offset,
                          offset_bytes=first.output["next_offset_bytes"], max_bytes=legacy.MAX_BYTES)
        assert rest.success and rest.output["eof"]
        assert len(json.dumps(rest.output).encode()) <= legacy.MAX_BYTES
        assert base64.b64decode(first.output["data"]) + base64.b64decode(rest.output["data"]) == data
    sidecar.write_bytes(b"")
    with unchanged(path, sidecar):
        result = await read(tool_for(path, spec), result_record_offset=offset)
        assert result.success and result.output["eof"] and result.output["total_bytes"] == 0
        assert result.output["data"] == "" and result.output["next_offset_bytes"] == 0


@pytest.mark.asyncio
async def test_large_header_preserved_and_too_small_page_fails(tmp_path):
    header = {**HEADER, "extension": {"large": "a" * 80_000}}
    path, _ = write_archive(tmp_path, header=header)
    with unchanged(path):
        assert_error(await read(tool_for(path)), "archive_record_too_large")
        result = await read(tool_for(path), max_bytes=legacy.MAX_BYTES)
        assert result.success and result.output["header"] == header
        assert result.output["eof"] and result.output["records"] == []


@pytest.mark.asyncio
async def test_no_relocation_or_basename_guessing(tmp_path):
    path, record_offset, result_path, _, spec = sidecar_archive(tmp_path, relocated=True)
    with unchanged(path, result_path):
        assert_error(await read(tool_for(path), result_record_offset=record_offset), "archive_file_missing")
        # Even an explicit relocation must preserve the nested relative suffix.
        wrong_root = result_path.parent
        spec["tool_results"] = str(wrong_root)
        assert_error(await read(tool_for(path, spec), result_record_offset=record_offset), "archive_file_missing")
        assert (await read(tool_for(path))).success


@pytest.mark.asyncio
@pytest.mark.parametrize("record,code", [
    ({"role": "tool", "content": "prefix"}, "invalid_result_record"),
    ({"role": "tool", "metadata": {}}, "missing_result_pointer"),
    ({"role": "tool", "metadata": {"full_result_path": False}}, "missing_result_pointer"),
    ({"role": "tool", "metadata": {"full_result_path": ""}}, "missing_result_pointer"),
    ({"role": "user", "metadata": {"full_result_path": "result.txt"}}, "invalid_result_record"),
    ({"role": "tool", "metadata": {"type": "context_managed_summary",
                                  "full_result_path": "result.txt"}}, "invalid_result_record"),
])
async def test_sidecar_requires_selected_tool_record_and_pointer(tmp_path, record, code):
    path, offsets = write_archive(tmp_path, [record])
    with unchanged(path):
        assert_error(await read(tool_for(path), result_record_offset=offsets[1]), code)
        raw = await read(tool_for(path))
        assert raw.success and raw.output["records"][0]["record"] == record


@pytest.mark.asyncio
async def test_sidecar_result_offsets_header_eof_midrecord_and_small_page(tmp_path):
    path, offset, result_path, _, spec = sidecar_archive(tmp_path)
    tool = tool_for(path, spec)
    with unchanged(path, result_path):
        assert_error(await read(tool, result_record_offset=0), "invalid_result_record")
        assert_error(await read(tool, result_record_offset=path.stat().st_size), "invalid_result_record")
        assert_error(await read(tool, result_record_offset=path.stat().st_size + 1), "invalid_archive_argument")
        assert_error(await read(tool, result_record_offset=offset + 1), "unaligned_archive_offset")
        assert_error(await read(tool, result_record_offset=offset, max_bytes=1), "archive_page_too_small")


@pytest.mark.asyncio
@pytest.mark.parametrize("pointer", ["../outside.txt", "nested/../../outside.txt", "/outside.txt"])
async def test_sidecar_rejects_traversal_or_unapproved_absolute_path(tmp_path, pointer):
    path, offsets = write_archive(tmp_path, [{"role": "tool", "metadata": {"full_result_path": pointer}}])
    with unchanged(path):
        assert_error(await read(tool_for(path), result_record_offset=offsets[1]), "unsafe_result_pointer")


@pytest.mark.asyncio
async def test_sibling_root_prefix_is_not_containment(tmp_path):
    outside = tmp_path / "tool_results-evil" / "result.txt"
    outside.parent.mkdir()
    outside.write_bytes(b"not approved")
    path, offsets = write_archive(tmp_path, [{"role": "tool", "metadata": {"full_result_path": str(outside)}}])
    with unchanged(path, outside):
        assert_error(await read(tool_for(path), result_record_offset=offsets[1]), "unsafe_result_pointer")


@pytest.mark.asyncio
@pytest.mark.parametrize("which", ["transcript_leaf", "transcript_parent", "sidecar_leaf", "sidecar_parent", "sidecar_root"])
async def test_symlink_leaf_and_parent_components_rejected(tmp_path, which):
    path, offset, result_path, _, spec = sidecar_archive(tmp_path)
    if which == "transcript_leaf":
        link = tmp_path / "alias.jsonl"
        link.symlink_to(path)
        spec["transcript"] = str(link)
    elif which == "transcript_parent":
        link = tmp_path / "alias-parent"
        link.symlink_to(tmp_path, target_is_directory=True)
        spec["transcript"] = str(link / path.name)
    elif which == "sidecar_leaf":
        real = result_path.with_suffix(".original")
        result_path.rename(real)
        result_path.symlink_to(real)
    else:
        directory = result_path.parent if which == "sidecar_parent" else tmp_path / "tool_results"
        real = directory.with_name(directory.name + "-original")
        directory.rename(real)
        directory.symlink_to(real, target_is_directory=True)
    with unchanged(path, result_path):
        result = await read(tool_for(path, spec), **(
            {} if which.startswith("transcript") else {"result_record_offset": offset}
        ))
        assert_error(result, "unsafe_path")


@pytest.mark.asyncio
@pytest.mark.parametrize("sidecar,is_fifo", [(False, False), (False, True), (True, False), (True, True)])
async def test_nonregular_transcript_and_sidecar_fail_without_blocking(tmp_path, sidecar, is_fifo):
    target = tmp_path / "not-regular"
    if is_fifo:
        os.mkfifo(target)
    else:
        target.mkdir()
    if sidecar:
        path, offsets = write_archive(tmp_path, [{"role": "tool", "metadata": {"full_result_path": "not-regular"}}])
        tool = tool_for(path, {"transcript": str(path), "tool_results": str(tmp_path)})
        result = await read(tool, result_record_offset=offsets[1])
    else:
        result = await read(tool_for(target))
    assert_error(result, "nonregular_file")


@pytest.mark.asyncio
async def test_missing_transcript_and_read_error_are_visible_and_safe(tmp_path, monkeypatch, capsys, caplog):
    path, _ = write_archive(tmp_path)
    assert_error(await read(tool_for(tmp_path / "missing")), "archive_file_missing")
    original_open = os.open

    def fail_leaf(name, flags, *args, **kwargs):
        if name == path.name:
            raise OSError(errno.EIO, "SECRET-SOURCE-CONTENT", str(path))
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(legacy.os, "open", fail_leaf)
    monkeypatch.setattr(legacy.os, "supports_dir_fd", {fail_leaf})
    with unchanged(path):
        result = await read(tool_for(path))
        assert_error(result, "archive_read_error")
        assert str(path) not in str(result)
        assert "SECRET-SOURCE-CONTENT" not in str(result)
    captured = capsys.readouterr()
    assert not captured.out and not captured.err and not caplog.text


@pytest.mark.asyncio
async def test_unsupported_safe_opens_fail_closed(tmp_path, monkeypatch):
    path, _ = write_archive(tmp_path)
    monkeypatch.setattr(legacy.os, "supports_dir_fd", set())
    with unchanged(path):
        assert_error(await read(tool_for(path)), "unsupported_safe_open")


@pytest.mark.asyncio
async def test_descriptor_chain_resists_parent_swap_before_leaf_open(tmp_path, monkeypatch):
    path, offset, result_path, data, spec = sidecar_archive(tmp_path)
    original_open = os.open
    pinned = result_path.parent.with_name("pinned-original")
    outside = tmp_path / "unapproved"
    outside.mkdir()
    (outside / result_path.name).write_bytes(b"must not read")
    swapped = False

    def swap_parent(name, flags, *args, **kwargs):
        nonlocal swapped
        assert flags & os.O_NOFOLLOW
        if name == result_path.name and not swapped:
            assert kwargs.get("dir_fd") is not None
            result_path.parent.rename(pinned)
            result_path.parent.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(legacy.os, "open", swap_parent)
    monkeypatch.setattr(legacy.os, "supports_dir_fd", {swap_parent})
    result = await read(tool_for(path, spec), result_record_offset=offset)
    assert result.success and base64.b64decode(result.output["data"]) == data
    assert (pinned / result_path.name).read_bytes() == data
    assert_error(await read(tool_for(path, spec), result_record_offset=offset), "unsafe_path")


def test_default_schema_has_no_empty_enum_and_accepts_ordinary_history():
    schema = ReadTranscriptTool(MagicMock()).input_schema
    # Stdlib-only check of the schema subset this tool emits. jsonschema is not
    # installed; this is not a claim of full provider/Draft validation.
    for spec in schema["properties"].values():
        assert "enum" not in spec or spec["enum"]
    assert set(schema["properties"]) == {"start_turn", "end_turn", "search"}
    for arguments in ({}, {"start_turn": 1, "end_turn": 2}, {"search": "example"}):
        assert not schema.get("required")
        for key, value in arguments.items():
            expected = schema["properties"][key]["type"]
            assert isinstance(value, {"integer": int, "string": str}[expected])


@pytest.mark.asyncio
async def test_default_schema_omission_does_not_allow_unknown_archive_fallback():
    tool = ReadTranscriptTool(MagicMock())
    assert_error(await read(tool), "unknown_archive_id")
    tool._coordinator.get_capability.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [101, 10**100, True])
async def test_review_max_records_independent_hard_cap(tmp_path, value):
    path, _ = write_archive(tmp_path)
    tool = tool_for(path)
    assert tool.input_schema["properties"]["max_records"]["maximum"] == 100
    with unchanged(path):
        assert_error(await read(tool, max_records=value), "invalid_archive_argument")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [
    "missing", "symlink_leaf", "symlink_parent", "directory", "fifo", "bad_header",
    "unsupported", "missing_result_root", "symlink_result_root", "file_result_root",
    "symlink_result_parent",
])
async def test_review_invalid_mount_preflight_has_no_side_effects(tmp_path, monkeypatch, kind):
    path, _ = write_archive(tmp_path)
    spec = {"transcript": str(path)}
    if kind == "missing":
        spec["transcript"] = str(tmp_path / "missing")
    elif kind == "symlink_leaf":
        link = tmp_path / "linked-transcript"
        link.symlink_to(path)
        spec["transcript"] = str(link)
    elif kind == "symlink_parent":
        link = tmp_path / "linked-parent"
        link.symlink_to(tmp_path, target_is_directory=True)
        spec["transcript"] = str(link / path.name)
    elif kind == "directory":
        spec["transcript"] = str(tmp_path)
    elif kind == "fifo":
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        spec["transcript"] = str(fifo)
    elif kind == "bad_header":
        path.write_bytes(b'{"role": "user"}\n')
    elif kind == "unsupported":
        monkeypatch.setattr(legacy.os, "supports_dir_fd", set())
    elif kind == "missing_result_root":
        spec["tool_results"] = str(tmp_path / "missing-results")
    elif kind == "symlink_result_root":
        link = tmp_path / "linked-results"
        link.symlink_to(tmp_path, target_is_directory=True)
        spec["tool_results"] = str(link)
    elif kind == "file_result_root":
        spec["tool_results"] = str(path)
    elif kind == "symlink_result_parent":
        parent = tmp_path / "real"
        (parent / "results").mkdir(parents=True)
        link = tmp_path / "linked-parent"
        link.symlink_to(parent, target_is_directory=True)
        spec["tool_results"] = str(link / "results")
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    with unchanged(path):
        with pytest.raises(legacy.ArchiveError):
            await mount(coordinator, {"legacy_transcript_paths": {"saved": spec}})
    coordinator.mount.assert_not_called()
    coordinator.hooks.register.assert_not_called()


@pytest.mark.asyncio
async def test_review_mount_allows_missing_implicit_results_and_missing_historical_root(tmp_path):
    path, _ = write_archive(tmp_path)
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    await mount(coordinator, {"legacy_transcript_paths": {"saved": str(path)}})
    tool = coordinator.mount.call_args.args[1]
    assert (await read(tool)).success
    actual = tmp_path / "approved-results"
    actual.mkdir()
    # Historical pointer roots are lexical mappings, not runtime prerequisites.
    await mount(coordinator, {"legacy_transcript_paths": {"saved": {
        "transcript": str(path), "original_tool_results": str(tmp_path / "gone"),
        "tool_results": str(actual),
    }}})
    assert coordinator.mount.call_count == 2


@pytest.mark.asyncio
async def test_review_mount_checks_only_bounded_header_not_all_records(tmp_path):
    path, _ = write_archive(tmp_path)
    with path.open("ab") as stream:
        stream.write(b"not-valid-JSON-after-valid-header\n")
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    with unchanged(path):
        await mount(coordinator, {"legacy_transcript_paths": {"saved": str(path)}})
        assert coordinator.mount.call_count == 1
        assert_error(await read(coordinator.mount.call_args.args[1]), "invalid_archive_json")


def mutate_owned_fixture(path, mutation):
    """Simulate another writer; never used outside pytest-owned temp paths."""
    before = path.stat()
    if mutation == "append":
        with path.open("ab") as stream:
            stream.write(b"\n")
    elif mutation == "truncate":
        with path.open("r+b") as stream:
            stream.truncate(0)
    elif mutation == "overwrite":
        data = path.read_bytes()
        with path.open("r+b") as stream:
            stream.write(b" " * len(data))
        assert path.stat().st_ino == before.st_ino
        assert path.stat().st_size == before.st_size
    else:
        replacement = path.with_name(path.name + ".replacement")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        assert path.stat().st_ino != before.st_ino
    # Ensure a deterministic timestamp change even on a fast/coarse filesystem.
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["append", "truncate", "overwrite"])
async def test_review_archive_mutation_at_old_eof_fails_changed(tmp_path, monkeypatch, mutation):
    path, _ = write_archive(tmp_path, [{"role": "user", "content": "original"}])
    tool = tool_for(path)
    original_record = legacy._record
    mutated = False

    def after_old_eof(stream):
        nonlocal mutated
        record = original_record(stream)
        if record is None and not mutated:
            mutate_owned_fixture(path, mutation)
            mutated = True
        return record

    monkeypatch.setattr(legacy, "_record", after_old_eof)
    result = await read(tool)
    assert mutated
    assert_error(result, "archive_changed")
    assert str(path) not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("which", ["archive", "result"])
@pytest.mark.parametrize("mutation", ["append", "truncate", "overwrite"])
@pytest.mark.parametrize("at_eof", [False, True])
async def test_review_sidecar_or_pointer_mutation_before_return_fails_changed(
    tmp_path, monkeypatch, which, mutation, at_eof,
):
    path, offset, sidecar, _, spec = sidecar_archive(tmp_path)
    tool = tool_for(path, spec)
    original_encode = legacy.base64.b64encode

    def mutate_after_read(raw):
        mutate_owned_fixture(path if which == "archive" else sidecar, mutation)
        return original_encode(raw)

    monkeypatch.setattr(legacy.base64, "b64encode", mutate_after_read)
    result = await read(tool, result_record_offset=offset,
                        offset_bytes=sidecar.stat().st_size if at_eof else 0)
    assert_error(result, "archive_changed")
    assert str(path) not in str(result) and str(sidecar) not in str(result)


@pytest.mark.asyncio
async def test_review_raw_continuation_revision_success_and_replacement(tmp_path):
    path, _ = write_archive(tmp_path, [
        {"role": "user", "content": "first"}, {"role": "assistant", "content": "second"},
    ])
    tool = tool_for(path)
    first = await read(tool, max_records=1)
    assert first.success
    revision = first.output["archive_revision"]
    assert isinstance(revision, str) and len(revision) == 64
    continuation = {"offset_bytes": first.output["next_offset_bytes"], "archive_revision": revision}
    with unchanged(path):
        second = await read(tool_for(path), **continuation)
        assert second.success and second.output["archive_revision"] == revision
    mutate_owned_fixture(path, "replace")
    with unchanged(path):
        assert_error(await read(tool_for(path), **continuation), "archive_changed")
        fresh = await read(tool_for(path))
        assert fresh.success and fresh.output["archive_revision"] != revision


@pytest.mark.asyncio
@pytest.mark.parametrize("which", ["archive", "result"])
async def test_review_sidecar_continuation_detects_either_replaced_generation(tmp_path, which):
    path, offset, sidecar, _, spec = sidecar_archive(tmp_path)
    first = await read(tool_for(path, spec), result_record_offset=offset, max_bytes=1024)
    assert first.success and not first.output["eof"]
    continuation = {
        "result_record_offset": offset, "offset_bytes": first.output["next_offset_bytes"],
        "archive_revision": first.output["archive_revision"],
        "result_revision": first.output["result_revision"],
    }
    with unchanged(path, sidecar):
        valid = await read(tool_for(path, spec), **continuation)
        assert valid.success
    mutate_owned_fixture(path if which == "archive" else sidecar, "replace")
    with unchanged(path, sidecar):
        assert_error(await read(tool_for(path, spec), **continuation), "archive_changed")
        assert (await read(tool_for(path, spec), result_record_offset=offset)).success


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["archive_revision", "result_revision"])
@pytest.mark.parametrize("value", [None, False, 1, "", [], {}, "a" * 63, "z" * 64, "A" * 64])
async def test_review_invalid_revision_types_and_token_shape(tmp_path, key, value):
    path, offset, _, _, spec = sidecar_archive(tmp_path)
    assert_error(await read(tool_for(path, spec), result_record_offset=offset,
                            **{key: value}), "invalid_archive_argument")


@pytest.mark.asyncio
async def test_review_revision_mismatch_rejected_before_content_read(tmp_path, monkeypatch):
    path, _ = write_archive(tmp_path)
    tool = tool_for(path)

    def unexpected_read(stream):
        pytest.fail("revision mismatch must fail before reading content")

    monkeypatch.setattr(legacy, "_record", unexpected_read)
    assert_error(await read(tool, archive_revision="0" * 64), "archive_changed")


@pytest.mark.asyncio
async def test_review_revision_overhead_respects_byte_budget_and_progress(tmp_path):
    path, offset, sidecar, data, spec = sidecar_archive(tmp_path)
    tool = tool_for(path, spec)
    first = await read(tool, result_record_offset=offset, max_bytes=1024)
    assert first.success
    assert len(json.dumps(first.output).encode()) <= 1024
    tokens = {key: first.output[key] for key in ("archive_revision", "result_revision")}
    empty = {**first.output, "data": "", "eof": False}
    empty["next_offset_bytes"] = len(data)
    exact_overhead = len(json.dumps(empty).encode())
    with unchanged(path, sidecar):
        assert_error(await read(tool, result_record_offset=offset, max_bytes=exact_overhead,
                                **tokens), "archive_page_too_small")
        progress = await read(tool, result_record_offset=offset, max_bytes=exact_overhead + 4, **tokens)
        assert progress.success and base64.b64decode(progress.output["data"])
        assert len(json.dumps(progress.output).encode()) <= exact_overhead + 4
        raw = await read(tool, max_bytes=1024, archive_revision=tokens["archive_revision"])
        assert raw.success and len(json.dumps(raw.output).encode()) <= 1024


@pytest.mark.asyncio
async def test_review_sidecar_revision_mismatch_rejected_before_sidecar_read(tmp_path, monkeypatch):
    path, offset, sidecar, _, spec = sidecar_archive(tmp_path)
    tool = tool_for(path, spec)
    first = await read(tool, result_record_offset=offset)
    original_open = legacy._open_regular

    @contextmanager
    def forbid_sidecar_read(target):
        with original_open(target) as stream:
            if target == sidecar:
                class NoContentReads:
                    def fileno(self):
                        return stream.fileno()

                    def read(self, *_args):
                        pytest.fail("result revision mismatch must precede sidecar read")

                    def seek(self, *_args):
                        pytest.fail("result revision mismatch must precede sidecar seek")

                yield NoContentReads()
            else:
                yield stream

    monkeypatch.setattr(legacy, "_open_regular", forbid_sidecar_read)
    with unchanged(path, sidecar):
        assert_error(await read(tool, result_record_offset=offset,
                                archive_revision=first.output["archive_revision"],
                                result_revision="0" * 64), "archive_changed")


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["append", "truncate", "overwrite"])
async def test_review_mutation_after_initial_snapshot_before_header_read(tmp_path, monkeypatch, mutation):
    path, _ = write_archive(tmp_path)
    tool = tool_for(path)
    original_revision = legacy._revision
    changed = False

    def mutate_after_snapshot(info):
        nonlocal changed
        result = original_revision(info)
        if not changed:
            mutate_owned_fixture(path, mutation)
            changed = True
        return result

    monkeypatch.setattr(legacy, "_revision", mutate_after_snapshot)
    assert_error(await read(tool), "archive_changed")


@pytest.mark.asyncio
async def test_review_stat_token_not_path_or_raw_stat_export(tmp_path):
    path, _ = write_archive(tmp_path)
    with unchanged(path):
        result = await read(tool_for(path))
        info = path.stat()
        expected = hashlib.sha256(json.dumps((
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
        )).encode("ascii")).hexdigest()
        assert result.success and result.output["archive_revision"] == expected
        assert str(path) not in json.dumps(result.output)
        assert "st_ino" not in result.output and "st_dev" not in result.output


@pytest.mark.asyncio
async def test_review_result_revision_without_result_selection_is_rejected(tmp_path):
    path, _ = write_archive(tmp_path)
    assert_error(await read(tool_for(path), result_revision="a" * 64), "mixed_archive_arguments")


@pytest.mark.asyncio
async def test_review_mount_validates_all_archives_before_first_side_effect(tmp_path):
    path, _ = write_archive(tmp_path)
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    with unchanged(path):
        with pytest.raises(legacy.ArchiveError, match="archive_file_missing"):
            await mount(coordinator, {"legacy_transcript_paths": {
                "valid": str(path), "invalid": str(tmp_path / "missing"),
            }})
    coordinator.mount.assert_not_called()
    coordinator.hooks.register.assert_not_called()


@pytest.mark.asyncio
async def test_review_mount_does_not_assume_later_archive_stability(tmp_path):
    path, _ = write_archive(tmp_path)
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    await mount(coordinator, {"legacy_transcript_paths": {"saved": str(path)}})
    tool = coordinator.mount.call_args.args[1]
    first = await read(tool)
    mutate_owned_fixture(path, "replace")
    assert_error(await read(tool, archive_revision=first.output["archive_revision"]), "archive_changed")
    path.unlink()
    assert_error(await read(tool), "archive_file_missing")


@pytest.mark.asyncio
async def test_review_explicit_original_root_requires_actual_directory_without_relocation(tmp_path):
    path, _ = write_archive(tmp_path)
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    with unchanged(path):
        with pytest.raises(legacy.ArchiveError, match="archive_file_missing"):
            await mount(coordinator, {"legacy_transcript_paths": {"saved": {
                "transcript": str(path), "original_tool_results": str(tmp_path / "gone"),
            }}})
    coordinator.mount.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("has_header", [False, True])
async def test_review_mount_bounds_cumulative_blank_prefix(tmp_path, has_header):
    prefix = b"\n" * (legacy.MAX_BYTES + 1)
    path, _ = write_archive(tmp_path, prefix=prefix)
    if not has_header:
        path.write_bytes(prefix)
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    with unchanged(path):
        with pytest.raises(legacy.ArchiveError, match="archive_record_too_large"):
            await mount(coordinator, {"legacy_transcript_paths": {"saved": str(path)}})
    coordinator.mount.assert_not_called()
    coordinator.hooks.register.assert_not_called()


@pytest.mark.asyncio
async def test_review_runtime_bounds_blank_lines_between_records(tmp_path):
    path, offsets = write_archive(tmp_path)
    path.write_bytes(
        path.read_bytes() + b" \n" * (legacy.MAX_BYTES // 2 + 1)
        + b'{"role":"user","content":"not silently reached"}\n'
    )
    with unchanged(path):
        assert_error(await read(tool_for(path), offset_bytes=offsets[0]), "archive_record_too_large")


def test_review_record_scan_consumes_at_most_one_extra_byte():
    import io

    stream = io.BytesIO(b"\n" * (legacy.MAX_BYTES + 100))
    with pytest.raises(legacy.ArchiveError, match="archive_record_too_large"):
        legacy._record(stream)
    assert stream.tell() == legacy.MAX_BYTES + 1