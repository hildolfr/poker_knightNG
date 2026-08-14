#!/usr/bin/env python3.13
"""Exact-SHA, fail-closed CUDA hardware qualification and evidence verifier.

The orchestrator imports only the standard library. CuPy and poker_knight_ng are
loaded only by the private worker subprocess after the host admission gate.
"""
from __future__ import annotations

import argparse
import base64
import csv
from email.parser import BytesParser
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

FORMAT_VERSION = "1"
CUPY_VERSION = "14.1.1"
MIN_FREE_BYTES = 2 * 1024**3
MAX_ARTIFACT_BYTES = 512 * 1024**2
MAX_JSON_BYTES = 16 * 1024**2
MAX_LOG_BYTES = 64 * 1024**2
MAX_PROCESS_OUTPUT_BYTES = 8 * 1024**2
PROCESS_TIMEOUT_SECONDS = 1800
PIPE_DRAIN_TIMEOUT_SECONDS = 5
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)\Z")
DEVICE_ID = re.compile(r"cuda-uuid:[0-9a-f]{32}\Z")
KERNEL_ID = re.compile(r"cuda-source-sha256:[0-9a-f]{64}\Z")
SOURCE_NAMES = (
    "philox.cuh", "dealer.cuh", "cards.cuh", "evaluator.cuh",
    "simulate.cuh", "reduce.cuh", "deterministic_kernels.cu",
)
SANITIZERS = ("memcheck", "racecheck", "initcheck", "synccheck")
SANITIZER_ZERO_MARKERS = {
    "memcheck": "ERROR SUMMARY: 0 errors",
    "racecheck": "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)",
    "initcheck": "ERROR SUMMARY: 0 errors",
    "synccheck": "ERROR SUMMARY: 0 errors",
}
ERROR_CODES = {
    "ADMISSION", "ARTIFACT", "CHECKOUT", "ENVIRONMENT", "INTERNAL",
    "JUNIT", "MANIFEST", "OUTPUT", "SANITIZER", "SOURCE", "SUBPROCESS", "VERIFY",
}
TOP_KEYS = {
    "artifacts", "environment", "error_codes", "format_version", "gates",
    "gpu", "pytest", "run_id", "sanitizers", "source", "status", "workers",
}


class QualificationError(Exception):
    def __init__(self, code: str) -> None:
        if code not in ERROR_CODES:
            code = "INTERNAL"
        self.code = code
        super().__init__(code)


class VerificationError(Exception):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical(value: object) -> bytes:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise VerificationError("JSON") from exc
    if not text.isascii():
        raise VerificationError("ASCII")
    return (text + "\n").encode("ascii")


def _pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in rows:
        if key in result:
            raise VerificationError("DUPLICATE")
        result[key] = value
    return result


def strict_json(raw: bytes) -> Any:
    if len(raw) > MAX_JSON_BYTES:
        raise VerificationError("JSON")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or raw.count(b"\n") != 1 or not raw.endswith(b"\n"):
        raise VerificationError("CANONICAL")
    try:
        value = json.loads(
            raw.decode("ascii"), object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(VerificationError("NUMBER")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError("JSON") from exc
    def ascii_only(item: object) -> bool:
        if type(item) is str:
            return item.isascii()
        if type(item) is list:
            return all(ascii_only(value) for value in item)
        if type(item) is dict:
            return all(type(key) is str and key.isascii() and ascii_only(value) for key, value in item.items())
        return item is None or type(item) in {bool, int, float}
    if not ascii_only(value):
        raise VerificationError("ASCII")
    if canonical(value) != raw:
        raise VerificationError("CANONICAL")
    return value


def closed(value: object, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise VerificationError("SCHEMA")
    return value


def decimal(value: object) -> str:
    if type(value) is not str or DECIMAL.fullmatch(value) is None:
        raise VerificationError("DECIMAL")
    return value


def _fsync_parent(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(destination: Path, data: bytes) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination.name)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_parent(destination.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def regular(path: Path, code: str = "ARTIFACT") -> None:
    if path.is_symlink() or not path.is_file():
        raise QualificationError(code)


def read_limited(path: Path, limit: int, code: str) -> bytes:
    regular(path, code)
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise QualificationError(code)
    return data


def file_record(path: Path) -> dict[str, str]:
    regular(path)
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(block)
            if total > MAX_ARTIFACT_BYTES:
                raise QualificationError("ARTIFACT")
            digest.update(block)
    return {"basename": path.name, "sha256": digest.hexdigest(), "size": str(total)}


def copy_artifact(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise QualificationError("ARTIFACT")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source, os.O_RDONLY | nofollow)
    except OSError as exc:
        raise QualificationError("ARTIFACT") from exc
    try:
        admitted = os.fstat(source_descriptor)
        if not stat.S_ISREG(admitted.st_mode) or not 0 < admitted.st_size <= MAX_ARTIFACT_BYTES:
            raise QualificationError("ARTIFACT")
        try:
            destination_descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o644)
        except OSError as exc:
            raise QualificationError("ARTIFACT") from exc
        try:
            copied = 0
            while True:
                block = os.read(source_descriptor, 1024 * 1024)
                if not block:
                    break
                copied += len(block)
                view = memoryview(block)
                while view:
                    written = os.write(destination_descriptor, view)
                    if written <= 0:
                        raise QualificationError("ARTIFACT")
                    view = view[written:]
            os.fsync(destination_descriptor)
        finally:
            os.close(destination_descriptor)
        final = os.fstat(source_descriptor)
        stable_identity = lambda row: (
            row.st_dev, row.st_ino, row.st_mode, row.st_size, row.st_mtime_ns, row.st_ctime_ns,
        )
        if copied != admitted.st_size or stable_identity(final) != stable_identity(admitted):
            raise QualificationError("ARTIFACT")
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(source_descriptor)
    _fsync_parent(destination.parent)


def _run(
    argv: list[str], *, cwd: Path, code: str, env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        process = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
    except OSError as exc:
        raise QualificationError(code) from exc
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = threading.Event()
    reader_errors: list[BaseException] = []

    def kill_group() -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def drain(name: str, stream: Any) -> None:
        try:
            while True:
                block = stream.read(64 * 1024)
                if not block:
                    break
                if len(buffers[name]) + len(block) > MAX_PROCESS_OUTPUT_BYTES:
                    overflow.set()
                    kill_group()
                    continue
                buffers[name].extend(block)
        except BaseException as exc:
            reader_errors.append(exc)
            kill_group()

    threads = [
        threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()

    def bounded_cleanup() -> None:
        kill_group()
        try:
            process.wait(timeout=PIPE_DRAIN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.monotonic() + PIPE_DRAIN_TIMEOUT_SECONDS
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            for stream in (process.stdout, process.stderr):
                if stream is None:
                    continue
                try:
                    stream.close()
                except (AttributeError, OSError):
                    pass
            for thread in threads:
                thread.join(max(0.0, deadline - time.monotonic()))

    try:
        return_code = process.wait(timeout=PROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        bounded_cleanup()
        raise QualificationError(code) from exc
    except BaseException:
        bounded_cleanup()
        raise
    deadline = time.monotonic() + PIPE_DRAIN_TIMEOUT_SECONDS
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        bounded_cleanup()
        raise QualificationError(code)
    if reader_errors or overflow.is_set():
        raise QualificationError(code)
    try:
        stdout = bytes(buffers["stdout"]).decode("utf-8", errors="strict")
        stderr = bytes(buffers["stderr"]).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise QualificationError(code) from exc
    if return_code != 0:
        raise QualificationError(code)
    return subprocess.CompletedProcess(argv, return_code, stdout, stderr)


def checkout_identity(root: Path, target_sha: str, expected_branch: str) -> tuple[str, str]:
    if (HEX40.fullmatch(target_sha) is None
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", expected_branch) is None
            or ".." in expected_branch):
        raise QualificationError("CHECKOUT")
    def git(*arguments: str) -> str:
        return _run(["git", *arguments], cwd=root, code="CHECKOUT").stdout.strip()

    target = git("rev-parse", "HEAD")
    branch = git("symbolic-ref", "--quiet", "--short", "HEAD")
    branch_target = git("rev-parse", f"refs/heads/{expected_branch}")
    if (target != target_sha or branch != expected_branch or branch_target != target_sha
            or git("status", "--porcelain", "--untracked-files=all")):
        raise QualificationError("CHECKOUT")
    return target, branch


def qualification_namespace(root: Path, *, create: bool) -> Path:
    root = root.resolve(strict=True)
    current = root
    for name in ("artifacts", "qualification"):
        current = current / name
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise QualificationError("OUTPUT")
        if create and not current.exists():
            current.mkdir()
            _fsync_parent(current.parent)
    expected = root / "artifacts/qualification"
    if not current.exists() or current.resolve(strict=True) != expected:
        raise QualificationError("OUTPUT")
    return expected


def verify_seed_manifest(root: Path) -> tuple[Path, Path]:
    manifest = root / "validation/holdem/v1/manifests/rng_seed_bank.sha256"
    seed_bank = root / "validation/holdem/v1/rng_seed_bank.json"
    regular(manifest, "MANIFEST")
    regular(seed_bank, "MANIFEST")
    wanted: str | None = None
    try:
        lines = read_limited(manifest, MAX_JSON_BYTES, "MANIFEST").decode("ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise QualificationError("MANIFEST") from exc
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[1] == "validation/holdem/v1/rng_seed_bank.json":
            if wanted is not None:
                raise QualificationError("MANIFEST")
            wanted = parts[0]
    if wanted is None or HEX64.fullmatch(wanted) is None or file_record(seed_bank)["sha256"] != wanted:
        raise QualificationError("MANIFEST")
    return manifest, seed_bank


def source_digest(root: Path) -> str:
    bundle = bytearray()
    source_directory = root / "src/poker_knight_ng/cuda-sources"
    for name in SOURCE_NAMES:
        path = source_directory / name
        regular(path, "SOURCE")
        bundle.extend(name.encode("ascii") + b"\0" + read_limited(path, 64 * 1024**2, "SOURCE"))
    return hashlib.sha256(bundle).hexdigest()


def _bounded_version(argv: list[str], root: Path, *, last: bool = False) -> str:
    lines = [line.strip() for line in _run(argv, cwd=root, code="ENVIRONMENT").stdout.replace("\r", "").splitlines() if line.strip()]
    output = lines[-1] if last and lines else lines[0] if lines else ""
    if not output or not output.isascii() or len(output) > 512:
        raise QualificationError("ENVIRONMENT")
    return output


def _nvidia_rows(query: str, root: Path) -> list[list[str]]:
    result = _run(
        ["nvidia-smi", f"--query-{query.split(':', 1)[0]}={query.split(':', 1)[1]}", "--format=csv,noheader,nounits"],
        cwd=root, code="ADMISSION",
    )
    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            rows.append([part.strip() for part in line.split(",")])
    return rows


def gpu_admission(root: Path) -> dict[str, Any]:
    rows = _nvidia_rows("gpu:name,uuid,compute_cap,memory.total,memory.free,power.draw,clocks.sm", root)
    if len(rows) != 1 or len(rows[0]) != 7:
        raise QualificationError("ADMISSION")
    name, uuid, cc, total_mib, free_mib, power_w, clock_mhz = rows[0]
    if not name or not uuid.startswith("GPU-") or not re.fullmatch(r"[0-9]+", total_mib) or not re.fullmatch(r"[0-9]+", free_mib):
        raise QualificationError("ADMISSION")
    applications: list[dict[str, str]] = []
    try:
        app_rows = _nvidia_rows("compute-apps:pid,used_gpu_memory", root)
    except QualificationError:
        app_rows = []
    for row in app_rows:
        if len(row) != 2 or not re.fullmatch(r"[0-9]+", row[0]) or not re.fullmatch(r"[0-9]+", row[1]):
            raise QualificationError("ADMISSION")
        if len(applications) >= 64:
            raise QualificationError("ADMISSION")
        applications.append({"pid": row[0], "used_memory_mib": row[1]})
    free_bytes = int(free_mib) * 1024 * 1024
    inventory = {
        "clock_sm_mhz": clock_mhz,
        "compute_applications": applications,
        "compute_capability": cc,
        "memory_free_before_bytes": str(free_bytes),
        "memory_total_bytes": str(int(total_mib) * 1024 * 1024),
        "name": name,
        "nvidia_uuid": uuid,
        "power_draw_w": power_w,
    }
    if free_bytes < MIN_FREE_BYTES:
        raise QualificationError("ADMISSION")
    return inventory


def child_environment(cache: Path, *, force_ptx: bool = False) -> dict[str, str]:
    environment: dict[str, str] = {}
    for name in ("PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(name)
        if value is not None:
            environment[name] = value
    environment.update({
        "CUPY_CACHE_DIR": str(cache),
        "CUDA_CACHE_PATH": str(cache / "driver"),
        "CUDA_CACHE_DISABLE": "0",
        "PKNG_QUALIFICATION_WORKER": "1",
        "PYTHONNOUSERSITE": "1",
    })
    if force_ptx:
        environment["CUDA_FORCE_PTX_JIT"] = "1"
    return environment


def parse_worker(stdout: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    if len(lines) != 1:
        raise QualificationError("SUBPROCESS")
    try:
        value = strict_json((lines[0] + "\n").encode("ascii"))
    except (UnicodeError, VerificationError) as exc:
        raise QualificationError("SUBPROCESS") from exc
    if type(value) is not dict:
        raise QualificationError("SUBPROCESS")
    return value


def run_worker(
    root: Path, seed_bank: Path, cache: Path, *, mode: str,
    force_ptx: bool = False, wheel: Path | None = None,
) -> dict[str, Any]:
    environment = child_environment(cache, force_ptx=force_ptx)
    started = time.monotonic_ns()
    argv = [sys.executable, str(Path(__file__).resolve()), "worker", "--mode", mode, "--seed-bank", str(seed_bank)]
    if wheel is not None:
        argv.extend(["--wheel", str(wheel)])
    completed = _run(
        argv,
        cwd=root, code="SUBPROCESS", env=environment,
    )
    wall = time.monotonic_ns() - started
    result = parse_worker(completed.stdout)
    result["wall_duration_ns"] = str(wall)
    return result


def run_sanitizer(root: Path, run: Path, seed_bank: Path, cache: Path, tool: str) -> tuple[dict[str, Any], dict[str, str]]:
    expected_marker = SANITIZER_ZERO_MARKERS.get(tool)
    if tool not in SANITIZERS or expected_marker is None:
        raise QualificationError("SANITIZER")
    log = run / f"{tool}.log"
    environment = child_environment(cache)
    completed = _run(
        [
            "compute-sanitizer", "--tool", tool, "--error-exitcode", "91", "--log-file", str(log),
            sys.executable, str(Path(__file__).resolve()), "worker", "--mode", "smoke", "--seed-bank", str(seed_bank),
        ],
        cwd=root, code="SANITIZER", env=environment,
    )
    worker = parse_worker(completed.stdout)
    regular(log, "SANITIZER")
    try:
        log_raw = read_limited(log, MAX_LOG_BYTES, "SANITIZER")
        log_text = log_raw.decode("utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise QualificationError("SANITIZER") from exc
    if expected_marker not in log_text:
        raise QualificationError("SANITIZER")
    return worker, file_record(log)


def result_projection(result: Any, rejection_count: int) -> dict[str, Any]:
    return {
        "case_hash": result.case_hash,
        "completed_trials": str(result.completed_trials),
        "equity_share_units": str(result.equity_share_units),
        "hero_category_counts": {name: str(value) for name, value in result.hero_category_counts},
        "losses": str(result.losses),
        "requested_trials": str(result.requested_trials),
        "rejection_count": str(rejection_count),
        "seed": f"0x{result.seed:016x}",
        "tie_by_other_winners": {str(index + 1): str(value) for index, value in enumerate(result.tie_by_other_winners)},
        "ties": str(result.ties),
        "unique_wins": str(result.unique_wins),
    }


def expected_projection(case: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    return {
        "case_hash": case["canonical_case_hash_hex"],
        "completed_trials": expected["completed_trials"],
        "equity_share_units": expected["equity_share_units"],
        "hero_category_counts": expected["hero_category_counts"],
        "losses": expected["losses"],
        "requested_trials": case["requested_trials"],
        "rejection_count": expected["rejection_count"],
        "seed": case["seed"],
        "tie_by_other_winners": expected["tie_by_other_winners"],
        "ties": str(sum(int(value) for value in expected["tie_by_other_winners"].values())),
        "unique_wins": expected["unique_wins"],
    }


def _device_uuid(properties: dict[Any, Any]) -> str:
    raw = properties.get("uuid", properties.get(b"uuid"))
    if not isinstance(raw, (bytes, bytearray)) or len(raw) != 16:
        raise RuntimeError("invalid device UUID")
    return "cuda-uuid:" + bytes(raw).hex()


def _verify_installed_wheel(distribution: importlib.metadata.Distribution, wheel_path: Path) -> dict[str, str]:
    if wheel_path.is_symlink() or not wheel_path.is_file() or not wheel_path.name.endswith(".whl"):
        raise RuntimeError("invalid wheel artifact")
    digest = file_record(wheel_path)["sha256"]
    with zipfile.ZipFile(wheel_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if (not names or len(names) != len(set(names)) or len(names) > 1000
                or any(info.file_size > 64 * 1024**2 or info.flag_bits & 1 for info in infos)
                or sum(info.file_size for info in infos) > 256 * 1024**2):
            raise RuntimeError("invalid wheel archive")
        version = distribution.version
        expected_basename = f"poker_knight_ng-{version}-py3-none-any.whl"
        dist_info = f"poker_knight_ng-{version}.dist-info"
        if wheel_path.name != expected_basename:
            raise RuntimeError("wheel filename does not match installed distribution")
        member_names = {name for name in names if not name.endswith("/")}
        required = {
            "poker_knight_ng/__init__.py", f"{dist_info}/METADATA", f"{dist_info}/WHEEL",
            f"{dist_info}/top_level.txt", f"{dist_info}/RECORD",
        }
        if not required <= member_names or any(
            not (name.startswith("poker_knight_ng/") or name.startswith(f"{dist_info}/"))
            for name in member_names
        ):
            raise RuntimeError("wheel closure is invalid")
        metadata = BytesParser().parsebytes(archive.read(f"{dist_info}/METADATA"))
        if metadata.get("Name") != "poker-knight-ng" or metadata.get("Version") != version:
            raise RuntimeError("wheel metadata does not match installed distribution")
        record_rows = list(csv.reader(archive.read(f"{dist_info}/RECORD").decode("utf-8").splitlines()))
        if any(len(row) != 3 for row in record_rows) or len(record_rows) != len(member_names):
            raise RuntimeError("wheel RECORD has invalid shape")
        record = {row[0]: (row[1], row[2]) for row in record_rows}
        if len(record) != len(record_rows) or set(record) != member_names:
            raise RuntimeError("wheel RECORD is not closed")
        for name in member_names:
            pure = Path(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise RuntimeError("unsafe wheel member")
            if name == f"{dist_info}/RECORD":
                if record[name] != ("", ""):
                    raise RuntimeError("wheel RECORD self-entry is invalid")
                continue
            data = archive.read(name)
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
            if record[name] != (f"sha256={encoded}", str(len(data))):
                raise RuntimeError("wheel RECORD digest mismatch")
            installed = Path(str(distribution.locate_file(name)))
            if installed.is_symlink() or not installed.is_file() or installed.read_bytes() != data:
                raise RuntimeError("installed distribution differs from wheel")
        package_root = Path(str(distribution.locate_file("poker_knight_ng")))
        installed_package: set[str] = set()
        for installed in package_root.rglob("*"):
            relative = installed.relative_to(package_root)
            if "__pycache__" in relative.parts or installed.suffix == ".pyc" or installed.is_dir():
                continue
            if installed.is_symlink() or not installed.is_file():
                raise RuntimeError("installed package closure is unsafe")
            installed_package.add("poker_knight_ng/" + relative.as_posix())
        archived_package = {name for name in member_names if name.startswith("poker_knight_ng/")}
        if installed_package != archived_package:
            raise RuntimeError("installed package has unverified members")
    return {"wheel_basename": wheel_path.name, "wheel_contents_verified": "true", "wheel_sha256": digest}


def worker(mode: str, seed_bank_path: Path, wheel_path: Path | None = None) -> None:
    if os.environ.get("PKNG_QUALIFICATION_WORKER") != "1":
        raise RuntimeError("qualification worker is private")
    import cupy as cp  # type: ignore
    from poker_knight_ng import EquityRequest
    from poker_knight_ng import _cuda_runtime
    from poker_knight_ng.contract.canonical import card_id
    from poker_knight_ng.engine import CPUReferenceEngine, CUDAEngine
    from poker_knight_ng.engine.result import ENGINE_BUILD_ID
    from poker_knight_ng.reference.monte_carlo import run_cpu_monte_carlo

    distribution = importlib.metadata.distribution("poker-knight-ng")
    if mode == "inventory" and wheel_path is None:
        raise RuntimeError("inventory requires wheel artifact")

    device = int(cp.cuda.runtime.getDevice())
    properties = cp.cuda.runtime.getDeviceProperties(device)
    uuid = _device_uuid(properties)
    major = properties.get("major", properties.get(b"major"))
    minor = properties.get("minor", properties.get(b"minor"))
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    inventory = {
        "build": {
            "contract_version": "v1", "engine_build_id": ENGINE_BUILD_ID,
            "rng_algorithm_id": "poker-knight-ng/philox4x32-10", "rng_algorithm_version": "1",
        },
        "cuda": {
            "compiler_options": list(_cuda_runtime.compiler_cache_key(cp)[2]),
            "driver_version": str(cp.cuda.runtime.driverGetVersion()),
            "runtime_version": str(cp.cuda.runtime.runtimeGetVersion()),
            "source_sha256": _cuda_runtime.APPROVED_SOURCE_SHA256,
        },
        "cupy_version": cp.__version__,
        "device": {
            "compute_capability": f"{int(major)}.{int(minor)}", "device_id": uuid,
            "memory_free_bytes": str(int(free_bytes)), "memory_total_bytes": str(int(total_bytes)),
        },
        "installation": _verify_installed_wheel(distribution, wheel_path) if wheel_path is not None else {},
        "poker_knight_ng_version": importlib.metadata.version("poker-knight-ng"),
        "python_version": platform.python_version(),
    }
    if mode == "inventory":
        print(canonical(inventory).decode("ascii"), end="")
        return
    if mode != "smoke":
        raise RuntimeError("invalid worker mode")
    raw = read_limited(seed_bank_path, MAX_JSON_BYTES, "MANIFEST")
    bank = strict_json(raw)
    cases = bank.get("exact_vectors") if type(bank) is dict else None
    if type(cases) is not list or not cases:
        raise RuntimeError("invalid seed bank")
    deck = {card_id(rank + suit): rank + suit for suit in "shdc" for rank in "23456789TJQKA"}
    observations: dict[str, Any] = {}
    provenance: tuple[str, str, str | None, str | None] | None = None
    native_runtime = _cuda_runtime.CupyDeterministicRuntime()

    class RecordingRuntime:
        def __init__(self) -> None:
            self.last: Any = None

        def run(self, **kwargs: Any) -> Any:
            self.last = native_runtime.run(**kwargs)
            return self.last

        def provenance(self) -> tuple[str, str]:
            return native_runtime.provenance()

    recording_runtime = RecordingRuntime()
    for case in cases:
        if type(case) is not dict or type(case.get("id")) is not str:
            raise RuntimeError("invalid seed case")
        common = {
            "hero_cards": tuple(deck[value] for value in case["hero_card_ids"]),
            "board_cards": tuple(deck[value] for value in case["board_card_ids"]),
            "opponent_count": case["opponent_count"],
            "requested_trials": int(case["requested_trials"]),
            "seed": int(case["seed"], 16),
        }
        cpu = CPUReferenceEngine().solve(EquityRequest(backend="cpu_reference", **common))
        cpu_raw = run_cpu_monte_carlo(
            seed=common["seed"], hero_card_ids=tuple(case["hero_card_ids"]),
            board_card_ids=tuple(case["board_card_ids"]), opponent_count=common["opponent_count"],
            requested_trials=common["requested_trials"], replay_case_hash=bytes.fromhex(case["canonical_case_hash_hex"]),
        )
        cuda = CUDAEngine(runtime=recording_runtime).solve(EquityRequest(backend="cuda", **common))
        cp.cuda.runtime.deviceSynchronize()
        if recording_runtime.last is None:
            raise RuntimeError("CUDA runtime result missing")
        cpu_projection = result_projection(cpu, cpu_raw.rejection_count)
        cuda_projection = result_projection(cuda, recording_runtime.last.rejection_count)
        expected = expected_projection(case)
        if cpu_projection != expected or cuda_projection != expected:
            raise RuntimeError("qualification aggregate mismatch")
        if provenance is None:
            provenance = cuda.provenance
        elif cuda.provenance != provenance:
            raise RuntimeError("qualification provenance changed")
        observations[case["id"]] = {"aggregate": cuda_projection, "engine_duration_ns": str(cuda.timing)}
    if provenance is None or provenance[1] != "cuda-deterministic-v1" or not isinstance(provenance[2], str) or not isinstance(provenance[3], str):
        raise RuntimeError("invalid CUDA provenance")
    if DEVICE_ID.fullmatch(provenance[2]) is None or provenance[2] != uuid or provenance[3] != f"cuda-source-sha256:{_cuda_runtime.APPROVED_SOURCE_SHA256}":
        raise RuntimeError("CUDA provenance mismatch")
    output = {
        "cases": observations,
        "device_id": provenance[2],
        "kernel_id": provenance[3],
        "qualification": provenance[1],
        "source_sha256": _cuda_runtime.APPROVED_SOURCE_SHA256,
    }
    print(canonical(output).decode("ascii"), end="")


def empty_evidence(run_id: str) -> dict[str, Any]:
    return {
        "artifacts": {}, "environment": {}, "error_codes": [], "format_version": FORMAT_VERSION,
        "gates": {}, "gpu": {}, "pytest": {}, "run_id": run_id, "sanitizers": {},
        "source": {}, "status": "failed", "workers": {},
    }


def _same_observation(left: dict[str, Any], right: dict[str, Any]) -> bool:
    ignored = {"wall_duration_ns"}
    clean_left = {key: value for key, value in left.items() if key not in ignored}
    clean_right = {key: value for key, value in right.items() if key not in ignored}
    if "cases" in clean_left and "cases" in clean_right:
        clean_left = dict(clean_left)
        clean_right = dict(clean_right)
        clean_left["cases"] = {
            case: {key: value for key, value in row.items() if key != "engine_duration_ns"}
            for case, row in clean_left["cases"].items()
        }
        clean_right["cases"] = {
            case: {key: value for key, value in row.items() if key != "engine_duration_ns"}
            for case, row in clean_right["cases"].items()
        }
    return clean_left == clean_right


def qualify(arguments: argparse.Namespace) -> int:
    if os.environ.get("RUN_CUDA_QUALIFICATION") != "1" or RUN_ID.fullmatch(arguments.run_id) is None:
        return 2
    root = repo_root().resolve()
    try:
        namespace = qualification_namespace(root, create=True)
    except QualificationError:
        return 2
    requested = Path(arguments.output_root).absolute() if arguments.output_root else namespace
    if requested != namespace:
        return 2
    wheel_input = Path(arguments.wheel)
    sdist_input = Path(arguments.sdist)
    if wheel_input.is_symlink() or sdist_input.is_symlink():
        return 2
    wheel = wheel_input.resolve()
    sdist = sdist_input.resolve()
    if not wheel.name.endswith(".whl") or not (sdist.name.endswith(".tar.gz") or sdist.name.endswith(".zip")):
        return 2
    try:
        regular(wheel)
        regular(sdist)
    except QualificationError:
        return 2
    run = namespace / arguments.run_id
    if run.exists() or run.is_symlink():
        return 2
    run.mkdir(mode=0o755)
    _fsync_parent(namespace)
    evidence = empty_evidence(arguments.run_id)
    destination = run / "qualification.json"
    try:
        target, branch = checkout_identity(root, arguments.target_sha, arguments.branch)
        manifest, seed_bank = verify_seed_manifest(root)
        uv_lock = root / "uv.lock"
        regular(uv_lock)
        files = run / "files"
        files.mkdir()
        if wheel.name == sdist.name:
            raise QualificationError("ARTIFACT")
        copied_wheel = files / wheel.name
        copied_sdist = files / sdist.name
        copy_artifact(wheel, copied_wheel)
        copy_artifact(sdist, copied_sdist)
        evidence["artifacts"] = {
            "rng_manifest": file_record(manifest), "rng_seed_bank": file_record(seed_bank),
            "sdist": file_record(copied_sdist), "uv_lock": file_record(uv_lock), "wheel": file_record(copied_wheel),
        }
        evidence["source"] = {"branch": branch, "clean": "true", "git_sha": target}
        evidence["gates"] = {"artifacts": "passed", "checkout": "passed", "manifest": "passed"}
        gpu = gpu_admission(root)
        evidence["gpu"] = gpu
        evidence["gates"]["admission"] = "passed"
        inventory_cache = run / "inventory-cache"
        inventory_cache.mkdir()
        inventory = run_worker(root, seed_bank, inventory_cache, mode="inventory", wheel=copied_wheel)
        cache = run / "cupy-cache"
        cache.mkdir()
        if any(cache.iterdir()):
            raise QualificationError("ENVIRONMENT")
        if inventory.get("cupy_version") != CUPY_VERSION:
            raise QualificationError("ENVIRONMENT")
        cuda = closed(inventory.get("cuda"), {"compiler_options", "driver_version", "runtime_version", "source_sha256"})
        build = closed(inventory.get("build"), {"contract_version", "engine_build_id", "rng_algorithm_id", "rng_algorithm_version"})
        device = closed(inventory.get("device"), {"compute_capability", "device_id", "memory_free_bytes", "memory_total_bytes"})
        installation = closed(inventory.get("installation"), {"wheel_basename", "wheel_contents_verified", "wheel_sha256"})
        local_source = source_digest(root)
        if cuda["source_sha256"] != local_source or device["compute_capability"] != gpu["compute_capability"]:
            raise QualificationError("SOURCE")
        normalized_nvidia_uuid = "cuda-uuid:" + gpu["nvidia_uuid"].removeprefix("GPU-").replace("-", "").lower()
        if device["device_id"] != normalized_nvidia_uuid:
            raise QualificationError("ADMISSION")
        if (installation["wheel_contents_verified"] != "true"
                or installation["wheel_basename"] != copied_wheel.name
                or installation["wheel_sha256"] != evidence["artifacts"]["wheel"]["sha256"]):
            raise QualificationError("ARTIFACT")
        evidence["gpu"].update({
            "cupy_memory_free_before_bytes": decimal(device["memory_free_bytes"]),
            "cupy_memory_total_bytes": decimal(device["memory_total_bytes"]),
        })
        evidence["environment"] = {
            "cpp_compiler": _bounded_version(["c++", "--version"], root),
            "cupy_version": inventory["cupy_version"],
            "cuda_driver_version": cuda["driver_version"],
            "cuda_runtime_version": cuda["runtime_version"],
            "nvcc_version": _bounded_version(["nvcc", "--version"], root, last=True),
            "poker_knight_ng_version": inventory["poker_knight_ng_version"],
            "python_version": inventory["python_version"],
        }
        evidence["source"].update({
            "compiler_options": cuda["compiler_options"], "contract_version": build["contract_version"],
            "cuda_source_sha256": local_source, "engine_build_id": build["engine_build_id"],
            "rng_algorithm_id": build["rng_algorithm_id"], "rng_algorithm_version": build["rng_algorithm_version"],
        })
        cold = run_worker(root, seed_bank, cache, mode="smoke")
        warm = run_worker(root, seed_bank, cache, mode="smoke")
        ptx_cache = run / "force-ptx-cache"
        ptx_cache.mkdir()
        ptx = run_worker(root, seed_bank, ptx_cache, mode="smoke", force_ptx=True)
        if not _same_observation(cold, warm) or not _same_observation(cold, ptx):
            raise QualificationError("VERIFY")
        evidence["workers"] = {"cold": cold, "force_ptx_jit": ptx, "warm": warm}
        junit = run / "pytest.xml"
        _run([sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"], cwd=root, code="JUNIT", env=child_environment(cache))
        regular(junit, "JUNIT")
        try:
            suite = ET.fromstring(read_limited(junit, MAX_JSON_BYTES, "JUNIT"))
        except (ET.ParseError, OSError) as exc:
            raise QualificationError("JUNIT") from exc
        failures = sum(int(node.attrib.get("failures", "0")) for node in suite.iter() if node.tag in {"testsuite", "testsuites"})
        errors = sum(int(node.attrib.get("errors", "0")) for node in suite.iter() if node.tag in {"testsuite", "testsuites"})
        if failures or errors:
            raise QualificationError("JUNIT")
        evidence["artifacts"]["junit"] = file_record(junit)
        evidence["pytest"] = {"errors": "0", "exit_code": "0", "failures": "0", "status": "passed"}
        sanitizer_cache = run / "sanitizer-cache"
        sanitizer_cache.mkdir()
        sanitizer_observations: dict[str, Any] = {}
        for tool in SANITIZERS:
            observation, record = run_sanitizer(root, run, seed_bank, sanitizer_cache, tool)
            if not _same_observation(cold, observation):
                raise QualificationError("SANITIZER")
            evidence["artifacts"][f"sanitizer_{tool}"] = record
            sanitizer_observations[tool] = {"status": "passed", "worker": observation}
        evidence["sanitizers"] = sanitizer_observations
        post = gpu_admission(root)
        evidence["gpu"]["memory_free_after_bytes"] = post["memory_free_before_bytes"]
        evidence["gates"].update({"pytest": "passed", "sanitizers": "passed", "workers": "passed"})
        evidence["status"] = "passed"
    except Exception as exc:
        code = exc.code if isinstance(exc, QualificationError) else "INTERNAL"
        evidence["error_codes"] = [code]
    atomic_write(destination, canonical(evidence))
    return 0 if evidence["status"] == "passed" else 1


def _verify_record(record: object, path: Path) -> None:
    row = closed(record, {"basename", "sha256", "size"})
    if row["basename"] != path.name or HEX64.fullmatch(row["sha256"]) is None or decimal(row["size"]) != str(path.stat().st_size):
        raise VerificationError("ARTIFACT")
    if file_record(path) != row:
        raise VerificationError("HASH")


def _verify_worker_shape(worker_value: object) -> dict[str, Any]:
    worker_row = closed(worker_value, {"cases", "device_id", "kernel_id", "qualification", "source_sha256", "wall_duration_ns"})
    if DEVICE_ID.fullmatch(worker_row["device_id"]) is None or KERNEL_ID.fullmatch(worker_row["kernel_id"]) is None:
        raise VerificationError("PROVENANCE")
    if worker_row["qualification"] != "cuda-deterministic-v1" or HEX64.fullmatch(worker_row["source_sha256"]) is None:
        raise VerificationError("PROVENANCE")
    decimal(worker_row["wall_duration_ns"])
    if type(worker_row["cases"]) is not dict or not worker_row["cases"]:
        raise VerificationError("CASES")
    for case_id, row in worker_row["cases"].items():
        if type(case_id) is not str or not case_id.isascii() or not case_id:
            raise VerificationError("CASES")
        case = closed(row, {"aggregate", "engine_duration_ns"})
        decimal(case["engine_duration_ns"])
        aggregate = closed(case["aggregate"], {
            "case_hash", "completed_trials", "equity_share_units", "hero_category_counts", "losses",
            "rejection_count", "requested_trials", "seed", "tie_by_other_winners", "ties", "unique_wins",
        })
        if HEX64.fullmatch(aggregate["case_hash"]) is None or not re.fullmatch(r"0x[0-9a-f]{16}", aggregate["seed"]):
            raise VerificationError("CASES")
        for key in ("completed_trials", "equity_share_units", "losses", "rejection_count", "requested_trials", "ties", "unique_wins"):
            decimal(aggregate[key])
        for mapping, keys in ((aggregate["tie_by_other_winners"], {str(i) for i in range(1, 7)}),
                              (aggregate["hero_category_counts"], {"high_card", "one_pair", "two_pair", "three_of_a_kind", "straight", "flush", "full_house", "four_of_a_kind", "straight_flush"})):
            row_map = closed(mapping, keys)
            for value in row_map.values():
                decimal(value)
        completed = int(aggregate["completed_trials"])
        wins = int(aggregate["unique_wins"])
        ties = int(aggregate["ties"])
        losses = int(aggregate["losses"])
        bins = aggregate["tie_by_other_winners"]
        categories = aggregate["hero_category_counts"]
        if (completed != int(aggregate["requested_trials"]) or wins + ties + losses != completed
                or ties != sum(int(value) for value in bins.values())
                or completed != sum(int(value) for value in categories.values())
                or int(aggregate["equity_share_units"]) != 420 * wins + sum(420 // (index + 1) * int(bins[str(index)]) for index in range(1, 7))):
            raise VerificationError("CASES")
    return worker_row


def verify(evidence_path: Path, root: Path) -> int:
    try:
        root = root.resolve(strict=True)
        namespace = qualification_namespace(root, create=False)
        if evidence_path.is_symlink():
            raise VerificationError("PATH")
        evidence_path = evidence_path.resolve(strict=True)
        if namespace not in evidence_path.parents or evidence_path.name != "qualification.json" or evidence_path.parent.parent != namespace:
            raise VerificationError("PATH")
        regular(evidence_path)
        evidence = closed(strict_json(read_limited(evidence_path, MAX_JSON_BYTES, "VERIFY")), TOP_KEYS)
        if evidence["format_version"] != FORMAT_VERSION or evidence["run_id"] != evidence_path.parent.name or RUN_ID.fullmatch(evidence["run_id"]) is None:
            raise VerificationError("SCHEMA")
        if evidence["status"] not in {"passed", "failed"} or type(evidence["error_codes"]) is not list or any(code not in ERROR_CODES for code in evidence["error_codes"]):
            raise VerificationError("SCHEMA")
        run = evidence_path.parent
        artifacts = evidence["artifacts"]
        if type(artifacts) is not dict:
            raise VerificationError("SCHEMA")
        fixed_paths = {
            "uv_lock": root / "uv.lock",
            "rng_manifest": root / "validation/holdem/v1/manifests/rng_seed_bank.sha256",
            "rng_seed_bank": root / "validation/holdem/v1/rng_seed_bank.json",
            "junit": run / "pytest.xml",
        }
        for key, record in artifacts.items():
            if key in fixed_paths:
                path = fixed_paths[key]
            elif key == "wheel" or key == "sdist":
                basename = closed(record, {"basename", "sha256", "size"})["basename"]
                if type(basename) is not str or "/" in basename or "\\" in basename or basename.startswith("."):
                    raise VerificationError("PATH")
                path = run / "files" / basename
            elif key.startswith("sanitizer_") and key.removeprefix("sanitizer_") in SANITIZERS:
                path = run / f"{key.removeprefix('sanitizer_')}.log"
            else:
                raise VerificationError("ARTIFACT")
            regular(path)
            _verify_record(record, path)
        if evidence["status"] == "failed":
            if not evidence["error_codes"]:
                raise VerificationError("FAILURE")
            return 0
        if evidence["error_codes"]:
            raise VerificationError("STATUS")
        source_preflight = evidence["source"]
        if type(source_preflight) is not dict:
            raise VerificationError("CHECKOUT")
        target_sha = source_preflight.get("git_sha")
        branch = source_preflight.get("branch")
        if type(target_sha) is not str or type(branch) is not str:
            raise VerificationError("CHECKOUT")
        try:
            identity = checkout_identity(root, target_sha, branch)
        except QualificationError as exc:
            raise VerificationError("CHECKOUT") from exc
        if identity != (target_sha, branch):
            raise VerificationError("CHECKOUT")
        verify_seed_manifest(root)
        required_artifacts = {"uv_lock", "rng_manifest", "rng_seed_bank", "wheel", "sdist", "junit"} | {f"sanitizer_{tool}" for tool in SANITIZERS}
        if set(artifacts) != required_artifacts:
            raise VerificationError("ARTIFACT")
        source = closed(evidence["source"], {
            "branch", "clean", "compiler_options", "contract_version", "cuda_source_sha256", "engine_build_id",
            "git_sha", "rng_algorithm_id", "rng_algorithm_version",
        })
        if (source["clean"] != "true" or HEX64.fullmatch(source["cuda_source_sha256"]) is None
                or source["cuda_source_sha256"] != source_digest(root) or source["contract_version"] != "v1"):
            raise VerificationError("SOURCE")
        if source["rng_algorithm_id"] != "poker-knight-ng/philox4x32-10" or source["rng_algorithm_version"] != "1":
            raise VerificationError("SOURCE")
        if source["engine_build_id"] != f"poker-knight-ng-{evidence['environment'].get('poker_knight_ng_version', '')}":
            raise VerificationError("SOURCE")
        if type(source["compiler_options"]) is not list or source["compiler_options"] != ["-std=c++17", f"--gpu-architecture=compute_{evidence['gpu']['compute_capability'].replace('.', '')}"]:
            raise VerificationError("SOURCE")
        environment = closed(evidence["environment"], {
            "cpp_compiler", "cupy_version", "cuda_driver_version", "cuda_runtime_version", "nvcc_version",
            "poker_knight_ng_version", "python_version",
        })
        if environment["cupy_version"] != CUPY_VERSION or any(type(value) is not str or not value or not value.isascii() for value in environment.values()):
            raise VerificationError("ENVIRONMENT")
        gpu = closed(evidence["gpu"], {
            "clock_sm_mhz", "compute_applications", "compute_capability", "cupy_memory_free_before_bytes",
            "cupy_memory_total_bytes", "memory_free_after_bytes",
            "memory_free_before_bytes", "memory_total_bytes", "name", "nvidia_uuid", "power_draw_w",
        })
        for key in ("cupy_memory_free_before_bytes", "cupy_memory_total_bytes", "memory_free_after_bytes", "memory_free_before_bytes", "memory_total_bytes"):
            decimal(gpu[key])
        if (int(gpu["memory_free_before_bytes"]) < MIN_FREE_BYTES
                or re.fullmatch(r"[0-9]{1,2}\.[0-9]", gpu["compute_capability"]) is None
                or type(gpu["compute_applications"]) is not list or len(gpu["compute_applications"]) > 64
                or any(type(gpu[key]) is not str or not gpu[key].isascii() or len(gpu[key]) > 256
                       for key in ("clock_sm_mhz", "name", "nvidia_uuid", "power_draw_w"))):
            raise VerificationError("ADMISSION")
        for application in gpu["compute_applications"]:
            app = closed(application, {"pid", "used_memory_mib"})
            decimal(app["pid"]); decimal(app["used_memory_mib"])
        gates = closed(evidence["gates"], {"admission", "artifacts", "checkout", "manifest", "pytest", "sanitizers", "workers"})
        if set(gates.values()) != {"passed"}:
            raise VerificationError("GATES")
        pytest_row = closed(evidence["pytest"], {"errors", "exit_code", "failures", "status"})
        if pytest_row != {"errors": "0", "exit_code": "0", "failures": "0", "status": "passed"}:
            raise VerificationError("JUNIT")
        try:
            xml_root = ET.fromstring(read_limited(run / "pytest.xml", MAX_JSON_BYTES, "JUNIT"))
        except (ET.ParseError, OSError) as exc:
            raise VerificationError("JUNIT") from exc
        if any(int(node.attrib.get(name, "0")) for node in xml_root.iter() for name in ("errors", "failures")):
            raise VerificationError("JUNIT")
        workers = closed(evidence["workers"], {"cold", "force_ptx_jit", "warm"})
        cold = _verify_worker_shape(workers["cold"])
        warm = _verify_worker_shape(workers["warm"])
        ptx = _verify_worker_shape(workers["force_ptx_jit"])
        if not _same_observation(cold, warm) or not _same_observation(cold, ptx):
            raise VerificationError("WORKERS")
        if cold["source_sha256"] != source["cuda_source_sha256"] or cold["kernel_id"] != f"cuda-source-sha256:{source['cuda_source_sha256']}":
            raise VerificationError("PROVENANCE")
        normalized_uuid = "cuda-uuid:" + gpu["nvidia_uuid"].removeprefix("GPU-").replace("-", "").lower()
        if cold["device_id"] != normalized_uuid:
            raise VerificationError("PROVENANCE")
        seed_bank = strict_json(read_limited(root / "validation/holdem/v1/rng_seed_bank.json", MAX_JSON_BYTES, "MANIFEST"))
        vectors = seed_bank.get("exact_vectors") if type(seed_bank) is dict else None
        if type(vectors) is not list:
            raise VerificationError("CASES")
        expected_cases = {
            case["id"]: expected_projection(case)
            for case in vectors if type(case) is dict and type(case.get("id")) is str
        }
        if set(cold["cases"]) != set(expected_cases):
            raise VerificationError("CASES")
        for case_id, expected in expected_cases.items():
            if cold["cases"][case_id]["aggregate"] != expected:
                raise VerificationError("CASES")
        sanitizers = closed(evidence["sanitizers"], set(SANITIZERS))
        for tool in SANITIZERS:
            sanitizer = closed(sanitizers[tool], {"status", "worker"})
            if sanitizer["status"] != "passed":
                raise VerificationError("SANITIZER")
            observation = _verify_worker_shape({**sanitizer["worker"], "wall_duration_ns": "0"})
            if not _same_observation(cold, observation):
                raise VerificationError("SANITIZER")
            expected_marker = SANITIZER_ZERO_MARKERS.get(tool)
            log_text = read_limited(run / f"{tool}.log", MAX_LOG_BYTES, "SANITIZER").decode("utf-8")
            if expected_marker is None or expected_marker not in log_text:
                raise VerificationError("SANITIZER")
        return 0
    except (Exception, OSError, UnicodeError, ValueError):
        return 1


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    commands = argument_parser.add_subparsers(dest="command", required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--evidence", required=True)
    verify_parser.add_argument("--root", required=True)
    qualify_parser = commands.add_parser("qualify")
    qualify_parser.add_argument("--run-id", required=True)
    qualify_parser.add_argument("--target-sha", required=True)
    qualify_parser.add_argument("--branch", required=True)
    qualify_parser.add_argument("--wheel", required=True)
    qualify_parser.add_argument("--sdist", required=True)
    qualify_parser.add_argument("--output-root")
    worker_parser = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--mode", choices=("inventory", "smoke"), required=True)
    worker_parser.add_argument("--seed-bank", required=True)
    worker_parser.add_argument("--wheel")
    return argument_parser


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = parser().parse_args(argv)
    except SystemExit:
        return 2
    if arguments.command == "verify":
        return verify(Path(arguments.evidence), Path(arguments.root))
    if arguments.command == "worker":
        worker(arguments.mode, Path(arguments.seed_bank), Path(arguments.wheel) if arguments.wheel else None)
        return 0
    return qualify(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
