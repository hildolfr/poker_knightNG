#!/usr/bin/env python3
"""CPU-only verifier for the committed CUDA release-qualification record."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

MAX_BYTES = 131_072
GIT_TIMEOUT_SECONDS = 10
GIT_STDERR_BYTES = 4_096
GIT_DRAIN_SECONDS = 0.25
GIT_CLEANUP_SECONDS = 1.0
SHA_RE = re.compile(r"[0-9a-f]{64}")
DEVICE_RE = re.compile(r"cuda-uuid:[0-9a-f]{32}")
SOURCE_SHA = "7fb617b900c06102caafe240ff95afe7fef2aa58"
SOURCE_DIGEST = "8da8349bed65e782a18d29f83de884341b3838f40c1e83904d07860c2c4ade5a"
EVIDENCE_SHA256 = "295fb7629dc53d956f933b2dbc2cd37a1142d52d0c4bf71762e534d79272132d"
QUALIFICATION_ID = "poker-knight-ng/cuda-release/7fb617b900c06102/8da8349bed65e782"
RECORD_RELATIVE = "validation/holdem/v1/cuda_release_qualification.json"
MANIFEST_RELATIVE = "validation/holdem/v1/manifests/cuda_release_qualification.sha256"
SOURCE_BINDINGS = (
    "pyproject.toml",
    "src/poker_knight_ng/_cuda_runtime.py",
    "src/poker_knight_ng/cuda-sources/cards.cuh",
    "src/poker_knight_ng/cuda-sources/dealer.cuh",
    "src/poker_knight_ng/cuda-sources/deterministic_kernels.cu",
    "src/poker_knight_ng/cuda-sources/evaluator.cuh",
    "src/poker_knight_ng/cuda-sources/philox.cuh",
    "src/poker_knight_ng/cuda-sources/reduce.cuh",
    "src/poker_knight_ng/cuda-sources/simulate.cuh",
    "tests/cuda/test_gpu_qualification_harness.py",
    "tools/qualify_gpu.py",
    "uv.lock",
    "validation/holdem/v1/manifests/rng_seed_bank.sha256",
    "validation/holdem/v1/rng_seed_bank.json",
)
CUDA_SOURCES = (
    "philox.cuh", "dealer.cuh", "cards.cuh", "evaluator.cuh",
    "simulate.cuh", "reduce.cuh", "deterministic_kernels.cu",
)
MANIFEST_PATHS = tuple(sorted(set(SOURCE_BINDINGS) | {
    "ARCHITECTURE.md",
    "README.md",
    "TODO.md",
    "src/poker_knight_ng/README.md",
    "tests/cuda/test_cuda_release_qualification.py",
    "tools/verify_cuda_release_qualification.py",
    "validation/holdem/v1/QUALIFICATION.md",
    "validation/holdem/v1/SPEC.md",
    RECORD_RELATIVE,
}))
ARTIFACTS = {
    "junit": "pytest.xml",
    "rng_manifest": "rng_seed_bank.sha256",
    "rng_seed_bank": "rng_seed_bank.json",
    "sanitizer_initcheck": "initcheck.log",
    "sanitizer_memcheck": "memcheck.log",
    "sanitizer_racecheck": "racecheck.log",
    "sanitizer_synccheck": "synccheck.log",
    "sdist": "poker_knight_ng-0.1.0.tar.gz",
    "uv_lock": "uv.lock",
    "wheel": "poker_knight_ng-0.1.0-py3-none-any.whl",
}
SANITIZER_SUMMARIES = {
    "initcheck": "ERROR SUMMARY: 0 errors",
    "memcheck": "ERROR SUMMARY: 0 errors",
    "racecheck": "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)",
    "synccheck": "ERROR SUMMARY: 0 errors",
}
COMMAND = (
    "RUN_CUDA_QUALIFICATION=1 python tools/qualify_gpu.py qualify "
    "--run-id phase6c-opt-7fb617b-5b3 "
    "--target-sha 7fb617b900c06102caafe240ff95afe7fef2aa58 "
    "--branch revival/phase-6c-optimize-evaluator "
    "--wheel poker_knight_ng-0.1.0-py3-none-any.whl "
    "--sdist poker_knight_ng-0.1.0.tar.gz"
)


class VerificationError(Exception):
    pass


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError("duplicate key")
        result[key] = value
    return result


def strict_json(raw: bytes) -> Any:
    def reject(_value: str) -> Any:
        raise VerificationError("JSON numbers are forbidden")

    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_pairs,
            parse_int=reject,
            parse_float=reject,
            parse_constant=reject,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError("invalid JSON") from exc


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _path_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_size


def _open_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (*_path_identity(info), info.st_mtime_ns, info.st_ctime_ns)


def read_regular(path: Path, limit: int = MAX_BYTES) -> bytes:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise VerificationError("invalid file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or _path_identity(opened) != _path_identity(before):
                raise VerificationError("file identity changed")
            chunks: list[bytes] = []
            remaining = limit + 1
            while remaining:
                chunk = os.read(descriptor, min(65_536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise VerificationError("unreadable file") from exc
    data = b"".join(chunks)
    if _open_identity(after) != _open_identity(opened):
        raise VerificationError("file changed while reading")
    if len(data) != opened.st_size or len(data) > limit:
        raise VerificationError("invalid file size")
    return data


def sha256(path: Path) -> str:
    return hashlib.sha256(read_regular(path)).hexdigest()


def closed(value: Any, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise VerificationError("closed schema violation")
    return value


def decimal(value: Any) -> int:
    if type(value) is not str or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise VerificationError("noncanonical decimal")
    return int(value)


def digest(value: Any) -> str:
    if type(value) is not str or SHA_RE.fullmatch(value) is None:
        raise VerificationError("invalid SHA-256")
    return value


def _all_strings(value: Any):
    if type(value) is str:
        yield value
    elif type(value) is dict:
        for key, item in value.items():
            yield key
            yield from _all_strings(item)
    elif type(value) is list:
        for item in value:
            yield from _all_strings(item)


def _verify_manifest(root: Path) -> None:
    raw = read_regular(root / MANIFEST_RELATIVE)
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as exc:
        raise VerificationError("invalid manifest") from exc
    rows: list[tuple[str, str]] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_./-]+)", line)
        if match is None:
            raise VerificationError("invalid manifest")
        rows.append((match.group(2), match.group(1)))
    paths = tuple(path for path, _value in rows)
    if paths != MANIFEST_PATHS or len(paths) != len(set(paths)):
        raise VerificationError("manifest path set mismatch")
    for path, expected in rows:
        if sha256(root / path) != expected:
            raise VerificationError("manifest hash mismatch")


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


def _read_capped_stream(
    stream, limit: int, state: dict[str, Any], process: subprocess.Popen[bytes],
) -> None:
    try:
        while True:
            chunk = stream.read(65_536)
            if not chunk:
                return
            remaining = limit - len(state["data"])
            if len(chunk) > remaining:
                if remaining > 0:
                    state["data"].extend(chunk[:remaining])
                state["overflow"] = True
                _kill_process_group(process)
                return
            state["data"].extend(chunk)
    except BaseException as exc:
        state["error"] = exc
        _kill_process_group(process)


def _wait_leader(process: subprocess.Popen[bytes], timeout: float) -> int:
    return process.wait(timeout=timeout)


def _join_readers(threads: tuple[threading.Thread, ...], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    return not any(thread.is_alive() for thread in threads)


def _cleanup_process(
    process: subprocess.Popen[bytes],
    threads: tuple[threading.Thread, ...],
    streams: tuple[Any, ...],
) -> None:
    deadline = time.monotonic() + GIT_CLEANUP_SECONDS
    _kill_process_group(process)
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except (OSError, subprocess.TimeoutExpired):
        pass
    if _join_readers(threads, max(0.0, deadline - time.monotonic())):
        return
    for stream in streams:
        try:
            os.close(stream.fileno())
        except (OSError, ValueError):
            pass
    _join_readers(threads, max(0.0, deadline - time.monotonic()))


def _git(root: Path, arguments: list[str], *, capture: bool = True) -> bytes:
    try:
        process = subprocess.Popen(
            ["git", "-C", str(root), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise VerificationError("git source lookup failed") from exc
    streams: tuple[Any, ...] = ()
    started: list[threading.Thread] = []
    try:
        assert process.stdout is not None and process.stderr is not None
        streams = (process.stdout, process.stderr)
        stdout_state: dict[str, Any] = {"data": bytearray(), "overflow": False, "error": None}
        stderr_state: dict[str, Any] = {"data": bytearray(), "overflow": False, "error": None}
        states = (stdout_state, stderr_state)
        threads = (
            threading.Thread(
                target=_read_capped_stream,
                args=(process.stdout, MAX_BYTES, stdout_state, process),
                daemon=True,
            ),
            threading.Thread(
                target=_read_capped_stream,
                args=(process.stderr, GIT_STDERR_BYTES, stderr_state, process),
                daemon=True,
            ),
        )
        for thread in threads:
            thread.start()
            started.append(thread)
        active = tuple(started)
        try:
            returncode = _wait_leader(process, GIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            _cleanup_process(process, active, streams)
            raise VerificationError("git source lookup timed out") from exc
        if not _join_readers(active, GIT_DRAIN_SECONDS):
            _cleanup_process(process, active, streams)
            raise VerificationError("git source lookup inherited pipes")
        for state in states:
            error = state["error"]
            if error is not None:
                if isinstance(error, Exception):
                    raise VerificationError("git output reader failed") from error
                raise error
        if any(state["overflow"] for state in states):
            _cleanup_process(process, active, streams)
            raise VerificationError("git output limit exceeded")
        if returncode != 0:
            raise VerificationError("git source lookup failed")
        return bytes(stdout_state["data"]) if capture else b""
    except BaseException:
        active = tuple(started)
        if process.poll() is None or any(thread.is_alive() for thread in active):
            _cleanup_process(process, active, streams)
        raise
    finally:
        for stream in streams:
            try:
                stream.close()
            except OSError:
                pass


def _source_blobs(root: Path, git_sha: str) -> dict[str, bytes]:
    _git(root, ["merge-base", "--is-ancestor", git_sha, "HEAD"], capture=False)
    return {path: _git(root, ["show", f"{git_sha}:{path}"]) for path in SOURCE_BINDINGS}


def _cuda_digest(blobs: dict[str, bytes]) -> str:
    payload = b"".join(
        name.encode("ascii") + b"\0" + blobs[f"src/poker_knight_ng/cuda-sources/{name}"]
        for name in CUDA_SOURCES
    )
    return hashlib.sha256(payload).hexdigest()


def _verify_record(record: dict[str, Any], root: Path) -> None:
    record = closed(record, {
        "device", "environment", "evidence", "format_version", "qualification_id",
        "sanitizers", "source", "status", "tests", "workers",
    })
    if record["format_version"] != "1" or record["qualification_id"] != QUALIFICATION_ID or record["status"] != "passed":
        raise VerificationError("record identity mismatch")

    source = closed(record["source"], {
        "bindings", "cuda_source_sha256", "executed_branch", "git_sha", "published_branch",
    })
    if source["git_sha"] != SOURCE_SHA or source["cuda_source_sha256"] != SOURCE_DIGEST:
        raise VerificationError("source identity mismatch")
    if source["executed_branch"] != "revival/phase-6c-optimize-evaluator" or source["published_branch"] != "revival/phase-6c-optimize-evaluator":
        raise VerificationError("branch identity mismatch")
    bindings = closed(source["bindings"], set(SOURCE_BINDINGS))
    source_blobs = _source_blobs(root, source["git_sha"])
    for path in SOURCE_BINDINGS:
        if digest(bindings[path]) != hashlib.sha256(source_blobs[path]).hexdigest():
            raise VerificationError("source binding mismatch")
    if _cuda_digest(source_blobs) != SOURCE_DIGEST:
        raise VerificationError("CUDA closure mismatch")

    evidence = closed(record["evidence"], {"artifacts", "command", "qualification_sha256", "run_id"})
    if evidence["run_id"] != "phase6c-opt-7fb617b-5b3" or evidence["command"] != COMMAND:
        raise VerificationError("evidence identity mismatch")
    if digest(evidence["qualification_sha256"]) != EVIDENCE_SHA256:
        raise VerificationError("evidence hash mismatch")
    artifacts = closed(evidence["artifacts"], set(ARTIFACTS))
    for name, basename in ARTIFACTS.items():
        row = closed(artifacts[name], {"basename", "sha256", "size"})
        if row["basename"] != basename or "/" in row["basename"] or "\\" in row["basename"]:
            raise VerificationError("artifact basename mismatch")
        digest(row["sha256"])
        if decimal(row["size"]) <= 0:
            raise VerificationError("artifact size mismatch")
    if artifacts["uv_lock"]["sha256"] != bindings["uv.lock"]:
        raise VerificationError("lock artifact mismatch")
    if artifacts["rng_seed_bank"]["sha256"] != bindings["validation/holdem/v1/rng_seed_bank.json"]:
        raise VerificationError("seed-bank artifact mismatch")
    if artifacts["rng_manifest"]["sha256"] != bindings["validation/holdem/v1/manifests/rng_seed_bank.sha256"]:
        raise VerificationError("seed-manifest artifact mismatch")

    environment = closed(record["environment"], {
        "cpp_compiler", "cuda_driver_version", "cuda_runtime_version", "cupy_version",
        "nvcc_version", "poker_knight_ng_version", "python_version",
    })
    if environment["cupy_version"] != "14.1.1" or environment["poker_knight_ng_version"] != "0.1.0" or environment["python_version"] != "3.13.15":
        raise VerificationError("environment version mismatch")
    decimal(environment["cuda_driver_version"])
    decimal(environment["cuda_runtime_version"])
    for key in ("cpp_compiler", "nvcc_version"):
        if type(environment[key]) is not str or not environment[key].isascii() or len(environment[key]) > 160:
            raise VerificationError("invalid environment text")

    device = closed(record["device"], {
        "compute_capability", "device_id", "minimum_required_bytes",
    })
    if device["compute_capability"] != "12.0" or type(device["device_id"]) is not str or DEVICE_RE.fullmatch(device["device_id"]) is None:
        raise VerificationError("device identity mismatch")
    minimum = decimal(device["minimum_required_bytes"])
    if minimum != 2 * 1024**3:
        raise VerificationError("device admission mismatch")

    tests = closed(record["tests"], {"errors", "failures", "passed", "skipped", "total"})
    counts = {key: decimal(value) for key, value in tests.items()}
    if counts != {"errors": 0, "failures": 0, "passed": 791, "skipped": 8, "total": 799}:
        raise VerificationError("test summary mismatch")

    workers = closed(record["workers"], {"cold", "cpu_cuda_frozen_equal", "exact_vector_count", "force_ptx_jit", "warm"})
    if workers != {"cold": "passed", "cpu_cuda_frozen_equal": True, "exact_vector_count": "3", "force_ptx_jit": "passed", "warm": "passed"}:
        raise VerificationError("worker summary mismatch")

    sanitizers = closed(record["sanitizers"], set(SANITIZER_SUMMARIES))
    for tool, summary in SANITIZER_SUMMARIES.items():
        row = closed(sanitizers[tool], {"log_sha256", "status", "summary"})
        if row["status"] != "passed" or row["summary"] != summary:
            raise VerificationError("sanitizer summary mismatch")
        if digest(row["log_sha256"]) != artifacts["sanitizer_" + tool]["sha256"]:
            raise VerificationError("sanitizer artifact mismatch")

    forbidden = ("/home/", "/tmp/", "desktop-drizzt", "RTX", "NVIDIA", "compute_applications", "power_draw")
    if any(any(token in value for token in forbidden) for value in _all_strings(record)):
        raise VerificationError("private or unstable evidence leaked")


def verify(record_path: Path, root: Path) -> int:
    try:
        root = root.resolve(strict=True)
        record_path = record_path.resolve(strict=True)
        if record_path != root / RECORD_RELATIVE:
            raise VerificationError("record path mismatch")
        raw = read_regular(record_path)
        record = strict_json(raw)
        if raw != canonical(record):
            raise VerificationError("record is not canonical")
        _verify_record(record, root)
        _verify_manifest(root)
        return 0
    except (Exception, OSError, UnicodeError, ValueError):
        return 1


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    argument_parser.add_argument("--record", type=Path)
    return argument_parser


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    root = arguments.root
    record = arguments.record or root / RECORD_RELATIVE
    return verify(record, root)


if __name__ == "__main__":
    raise SystemExit(main())
