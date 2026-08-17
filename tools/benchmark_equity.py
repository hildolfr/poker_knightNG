#!/usr/bin/env python3
"""Closed Phase 6C benchmark controller primitives.

Controller import is stdlib-only and CUDA-inert. Live worker execution is added
only after the preregistered protocol and private timing seam are verified.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from math import ceil
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any


BENCHMARK_ID = "holdem-v1-cuda-baseline-1"
SCENARIO_FORMAT = "phase6c-scenario-v1"
STREETS = ("preflop", "flop", "turn", "river")
BOARD_COUNTS = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
OPPONENT_COUNTS = ("1", "3", "6")
REQUESTED_TRIALS = ("10000", "100000", "500000", "1000000")
SCENARIO_KEYS = {
    "board_cards", "format_version", "hero_cards", "id",
    "opponent_count", "seed", "street",
}
CARD_PATTERN = re.compile(r"(?:[2-9TJQKA][cdhs])\Z", re.ASCII)
HEX_SEED_PATTERN = re.compile(r"0x[0-9a-f]{16}\Z", re.ASCII)
RANKS = "23456789TJQKA"
SUITS = "cdhs"
MAX_SCENARIO_BYTES = 1024
MAX_WORKER_OUTPUT_BYTES = 2 * 1024 * 1024
WORKER_TIMEOUT_SECONDS = 7200
WORKER_MODES = ("inventory", "expected", "cold", "warm", "steady", "stage")
WORKER_MARKER = "PKNG_BENCHMARK_WORKER"
CACHE_SEAL_NAME = ".phase6c-cache-seal.json"
STAGE_BATCH_BLOCKS = 256


class BenchmarkError(ValueError):
    """A stable closed benchmark-contract failure."""


def canonical(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkError("CANONICAL_JSON") from exc
    return (text + "\n").encode("ascii")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BenchmarkError("DUPLICATE_JSON_KEY")
        value[key] = item
    return value


def _strict_json(data: bytes, *, maximum: int, code: str) -> Any:
    if not data or len(data) > maximum or not data.endswith(b"\n") or b"\r" in data:
        raise BenchmarkError(f"{code}_FRAMING")
    try:
        text = data.decode("ascii")
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_int=lambda _text: (_ for _ in ()).throw(BenchmarkError("JSON_NUMBER")),
            parse_float=lambda _text: (_ for _ in ()).throw(BenchmarkError("JSON_NUMBER")),
            parse_constant=lambda _text: (_ for _ in ()).throw(BenchmarkError("JSON_CONSTANT")),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"{code}_JSON") from exc
    if canonical(value) != data:
        raise BenchmarkError(f"{code}_NONCANONICAL")
    return value


def strict_json(data: bytes) -> Any:
    return _strict_json(data, maximum=MAX_SCENARIO_BYTES, code="SCENARIO")


def parse_worker_output(stdout: bytes, stderr: bytes) -> dict[str, Any]:
    if type(stdout) is not bytes or type(stderr) is not bytes or stderr:
        raise BenchmarkError("WORKER_OUTPUT")
    value = _strict_json(stdout, maximum=MAX_WORKER_OUTPUT_BYTES, code="WORKER")
    if type(value) is not dict:
        raise BenchmarkError("WORKER_OUTPUT")
    return value


def worker_environment(cache: Path) -> dict[str, str]:
    if not isinstance(cache, Path) or not cache.is_absolute() or cache.is_symlink():
        raise BenchmarkError("WORKER_CACHE")
    return {
        "CUPY_CACHE_DIR": str(cache),
        "CUDA_CACHE_PATH": str(cache / "driver"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        WORKER_MARKER: "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }


def prepare_cold_cache(cache: Path) -> None:
    if not isinstance(cache, Path) or not cache.is_absolute() or cache.exists() or cache.is_symlink():
        raise BenchmarkError("COLD_CACHE")
    try:
        cache.mkdir(mode=0o700)
    except OSError as exc:
        raise BenchmarkError("COLD_CACHE") from exc
    if cache.is_symlink() or not cache.is_dir():
        raise BenchmarkError("COLD_CACHE")


def _cache_files(cache: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(cache.rglob("*"), key=lambda item: item.relative_to(cache).as_posix()):
        if path.name == CACHE_SEAL_NAME and path.parent == cache:
            continue
        if path.is_symlink():
            raise BenchmarkError("WARM_CACHE")
        if path.is_dir():
            continue
        if not path.is_file():
            raise BenchmarkError("WARM_CACHE")
        data = path.read_bytes()
        rows.append({
            "path": path.relative_to(cache).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": str(len(data)),
        })
    if not rows:
        raise BenchmarkError("WARM_CACHE")
    return rows


def seal_cold_cache(cache: Path, *, cold_result_sha256: str) -> dict[str, Any]:
    if (
        not isinstance(cache, Path)
        or cache.is_symlink()
        or not cache.is_dir()
        or type(cold_result_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", cold_result_sha256) is None
    ):
        raise BenchmarkError("WARM_CACHE")
    manifest = {
        "cold_result_sha256": cold_result_sha256,
        "files": _cache_files(cache),
        "format_version": "phase6c-cache-seal-v1",
    }
    data = canonical(manifest)
    path = cache / CACHE_SEAL_NAME
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            offset = 0
            while offset < len(data):
                written = os.write(descriptor, data[offset:])
                if written < 1:
                    raise OSError("short cache-seal write")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent = os.open(cache, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except OSError as exc:
        raise BenchmarkError("WARM_CACHE") from exc
    return manifest


def verify_warm_cache(cache: Path) -> dict[str, Any]:
    if not isinstance(cache, Path) or cache.is_symlink() or not cache.is_dir():
        raise BenchmarkError("WARM_CACHE")
    marker = cache / CACHE_SEAL_NAME
    if marker.is_symlink() or not marker.is_file():
        raise BenchmarkError("WARM_CACHE")
    try:
        value = _strict_json(marker.read_bytes(), maximum=128 * 1024, code="CACHE")
    except (OSError, BenchmarkError) as exc:
        raise BenchmarkError("WARM_CACHE") from exc
    if (
        type(value) is not dict
        or set(value) != {"cold_result_sha256", "files", "format_version"}
        or value["format_version"] != "phase6c-cache-seal-v1"
        or type(value["cold_result_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", value["cold_result_sha256"]) is None
        or value["files"] != _cache_files(cache)
    ):
        raise BenchmarkError("WARM_CACHE")
    return value


def worker_argv(
    mode: str,
    *,
    scenario_directory: Path,
    wheel: Path,
) -> list[str]:
    if mode not in WORKER_MODES:
        raise BenchmarkError("WORKER_MODE")
    if not all(
        isinstance(path, Path) and path.is_absolute() and not path.is_symlink()
        for path in (scenario_directory, wheel)
    ):
        raise BenchmarkError("WORKER_PATH")
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--mode",
        mode,
        "--scenario-dir",
        str(scenario_directory),
        "--wheel",
        str(wheel),
    ]


def require_private_worker(mode: str) -> None:
    if os.environ.get(WORKER_MARKER) != "1" or mode not in WORKER_MODES:
        raise BenchmarkError("PRIVATE_WORKER")


def _has_symlink_component(path: Path) -> bool:
    if not path.is_absolute():
        return True
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def worker_main(argv: tuple[str, ...] | list[str]) -> dict[str, Any]:
    arguments = tuple(argv)
    if (
        len(arguments) != 7
        or arguments[0] != "--worker"
        or arguments[1] != "--mode"
        or arguments[2] not in WORKER_MODES
        or arguments[3] != "--scenario-dir"
        or not arguments[4]
        or arguments[5] != "--wheel"
        or not arguments[6]
    ):
        raise BenchmarkError("WORKER_MODE")
    mode = arguments[2]
    require_private_worker(mode)
    scenario_directory, wheel = Path(arguments[4]), Path(arguments[6])
    if (
        not scenario_directory.is_absolute() or _has_symlink_component(scenario_directory)
        or not scenario_directory.is_dir() or not wheel.is_absolute()
        or _has_symlink_component(wheel) or not wheel.is_file() or wheel.suffix != ".whl"
    ):
        raise BenchmarkError("WORKER_PATH")
    # This is intentionally the first package/CUDA import in the worker.
    equity_request, solve_cuda, serialize = _worker_public_api()
    cells = expand_matrix(load_scenarios(scenario_directory))

    def normal(cell: dict[str, Any]) -> tuple[str, str]:
        request = equity_request.parse({
            "contract_version": "v1", "hero_cards": cell["hero_cards"],
            "board_cards": cell["board_cards"], "opponent_count": cell["opponent_count"],
            "requested_trials": cell["requested_trials"], "seed": cell["seed"],
            "backend": "cuda", "rng": {
                "algorithm_id": "poker-knight-ng/philox4x32-10", "algorithm_version": "1",
            },
        })
        wire = serialize(solve_cuda(request), request)
        timing = wire.get("timing")
        if type(timing) is not dict or type(timing.get("total_duration_ns")) is not str:
            raise BenchmarkError("WORKER_MEASUREMENT")
        duration = timing["total_duration_ns"]
        if re.fullmatch(r"[1-9][0-9]{0,19}", duration) is None:
            raise BenchmarkError("WORKER_MEASUREMENT")
        analytical = dict(wire)
        analytical.pop("timing", None)
        analytical.pop("provenance", None)
        return duration, hashlib.sha256(canonical(analytical)).hexdigest()

    if mode == "expected":
        return {"mode": mode, "analytical_sha256s": {cell["cell_id"]: normal(cell)[1] for cell in cells}}
    if mode in ("cold", "warm"):
        cell = cells[0]
        duration, digest = normal(cell)
        return {"mode": mode, "duration_ns": duration, "analytical_sha256": digest}
    if mode == "steady":
        output = {}
        for cell in cells:
            warmup_duration, warmup_digest = normal(cell)
            durations, digests = zip(*(normal(cell) for _ in range(30)))
            output[cell["cell_id"]] = {
                "warmup_duration_ns": warmup_duration,
                "warmup_analytical_sha256": warmup_digest,
                "durations_ns": list(durations), "analytical_sha256s": list(digests),
            }
        return {"mode": mode, "cells": output}
    if mode == "stage":
        return _worker_stage(cells, equity_request, serialize)
    if mode == "inventory":
        return _worker_inventory(wheel)
    raise BenchmarkError("WORKER_MODE")


def _worker_public_api() -> tuple[Any, Any, Any]:
    """Lazy public boundary: controller imports remain package/CUDA inert."""
    from poker_knight_ng import EquityRequest, serialize_equity_result, solve_cuda
    return EquityRequest, solve_cuda, serialize_equity_result


def _run_private_stage(
    runtime: Any,
    runtime_module: Any,
    *,
    hero: tuple[int, ...],
    board: tuple[int, ...],
    opponents: int,
    key: tuple[int, int],
    count: int,
    observer: Any,
    planned_batch_blocks: int,
) -> tuple[Any, dict[str, str]]:
    """Benchmark-only CUDA launch loop with private event boundaries."""
    runtime._validate_run_arguments(hero, board, opponents, key, 0, count)
    cp = runtime._cupy()
    available_batch_blocks = runtime._batch_capacity()
    if (
        type(planned_batch_blocks) is not int
        or planned_batch_blocks < 1
        or planned_batch_blocks > available_batch_blocks
    ):
        raise BenchmarkError("STAGE_BATCH_PLAN")
    simulate, reduce = runtime._kernels()
    if runtime._batch_capacity() < planned_batch_blocks:
        raise BenchmarkError("STAGE_BATCH_PLAN")

    observer.prepare_inputs(hero, board)
    observer.boundary("h2d", "start", 0)
    hero_d, board_d = observer.copy_inputs()
    observer.boundary("h2d", "end", 0)
    pending_batches: list[tuple[Any, int]] = []
    offset = 0
    ordinal = 0
    while offset < count:
        if ordinal > 0 and runtime._batch_capacity() < planned_batch_blocks:
            raise BenchmarkError("STAGE_BATCH_PLAN")
        trials = min(count - offset, planned_batch_blocks * runtime_module.THREADS)
        blocks = (trials + runtime_module.THREADS - 1) // runtime_module.THREADS
        partials = cp.empty(blocks * runtime_module.AGGREGATE_BYTES, dtype=cp.uint8)
        final = cp.empty(runtime_module.AGGREGATE_BYTES, dtype=cp.uint8)
        observer.boundary("simulate", "start", ordinal)
        simulate(
            (blocks,),
            (runtime_module.THREADS,),
            (
                hero_d, board_d, cp.uint32(len(board)), cp.uint32(opponents),
                cp.uint32(key[0]), cp.uint32(key[1]), cp.uint64(offset),
                cp.uint64(trials), partials,
            ),
        )
        observer.boundary("simulate", "end", ordinal)
        observer.boundary("reduction", "start", ordinal)
        reduce((1,), (runtime_module.THREADS,), (partials, cp.uint64(blocks), final))
        observer.boundary("reduction", "end", ordinal)
        observer.boundary("d2h", "start", ordinal)
        raw = observer.copy_to_host(final, ordinal)
        observer.boundary("d2h", "end", ordinal)
        pending_batches.append((raw, trials))
        offset += trials
        ordinal += 1

    profile = observer.finish()
    aggregate = runtime_module.empty_aggregate_record()
    for raw, trials in pending_batches:
        batch_result = runtime_module.validated_aggregate(
            raw,
            opponents=opponents,
            requested_trials=trials,
            board_count=len(board),
        )
        aggregate = runtime_module._merge(
            aggregate,
            runtime_module._record_from_result(batch_result),
        )
    result = runtime_module.validated_aggregate(
        aggregate,
        opponents=opponents,
        requested_trials=count,
        board_count=len(board),
    )
    if type(profile) is not dict or any(
        type(name) is not str or type(value) is not str
        for name, value in profile.items()
    ):
        raise BenchmarkError("STAGE_PROFILE")
    return result, profile


def _worker_stage(cells: tuple[dict[str, Any], ...], equity_request: Any, serialize: Any) -> dict[str, Any]:
    import cupy as cp  # type: ignore[import-not-found]
    from poker_knight_ng import _cuda_runtime
    from poker_knight_ng.contract.canonical import card_id
    from poker_knight_ng.engine.result import to_equity_result
    from poker_knight_ng.reference.rng import derive_philox_key

    output = {}
    runtime = _cuda_runtime.CupyDeterministicRuntime()
    plan = STAGE_BATCH_BLOCKS
    if runtime._batch_capacity() < plan:
        raise BenchmarkError("STAGE_BATCH_PLAN")
    batch_counts: dict[str, str] = {}
    for cell in cells:
        request = equity_request.parse({"contract_version": "v1", "hero_cards": cell["hero_cards"], "board_cards": cell["board_cards"], "opponent_count": cell["opponent_count"], "requested_trials": cell["requested_trials"], "seed": cell["seed"], "backend": "cuda", "rng": {"algorithm_id": "poker-knight-ng/philox4x32-10", "algorithm_version": "1"}})
        count = int(cell["requested_trials"])
        batches = ceil(count / (plan * _cuda_runtime.THREADS))
        batch_counts[cell["cell_id"]] = str(batches)
        observer = CupyStageObserver(cp, batches=batches, aggregate_bytes=_cuda_runtime.AGGREGATE_BYTES)
        hero = tuple(sorted(card_id(card) for card in cell["hero_cards"]))
        board = tuple(sorted(card_id(card) for card in cell["board_cards"]))
        from poker_knight_ng.contract import canonical_case_hash
        _digest, key = derive_philox_key(int(cell["seed"], 16), bytes.fromhex(canonical_case_hash(request)))
        result, profile = _run_private_stage(
            runtime,
            _cuda_runtime,
            hero=hero,
            board=board,
            opponents=int(cell["opponent_count"]),
            key=key,
            count=count,
            observer=observer,
            planned_batch_blocks=plan,
        )
        # Validate the private aggregate through the exact production adapter.
        device, kernel = runtime.provenance()
        public = to_equity_result(
            result, request, 0,
            provenance=("cuda-deterministic-v1", device, kernel),
        )
        wire = serialize(public, request)
        analytical = dict(wire); analytical.pop("timing"); analytical.pop("provenance")
        # The observer result itself must also satisfy the public adapter.
        if result is None:
            raise BenchmarkError("STAGE_PROFILE")
        output[cell["cell_id"]] = {"durations": evidence_stage_durations(profile), "analytical_sha256": hashlib.sha256(canonical(analytical)).hexdigest()}
    return {
        "mode": "stage",
        "planned_batch_blocks": str(plan),
        "batch_counts": batch_counts,
        "cells": output,
    }


def _worker_inventory(wheel: Path) -> dict[str, Any]:
    # Qualification's full archive/install byte closure is deliberately reused.
    import importlib.util
    qualification = Path(__file__).with_name("qualify_gpu.py")
    spec = importlib.util.spec_from_file_location("_pkng_qualification", qualification)
    if spec is None or spec.loader is None:
        raise BenchmarkError("INVENTORY")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import importlib.metadata
    import platform
    import cupy as cp  # worker-only environment inventory
    try:
        installation = module._verify_installed_wheel(importlib.metadata.distribution("poker-knight-ng"), wheel)
        device = int(cp.cuda.runtime.getDevice())
        properties = cp.cuda.runtime.getDeviceProperties(device)

        def property_value(name: str) -> Any:
            return properties.get(name, properties.get(name.encode("ascii")))

        raw_name = property_value("name")
        raw_uuid = property_value("uuid")
        major = property_value("major")
        minor = property_value("minor")
        total_memory = property_value("totalGlobalMem")
        if (
            not isinstance(raw_name, (str, bytes))
            or not isinstance(raw_uuid, (bytes, bytearray))
            or len(raw_uuid) != 16
            or type(major) is not int
            or type(minor) is not int
            or type(total_memory) is not int
            or total_memory < 1
        ):
            raise BenchmarkError("INVENTORY")
        name = raw_name.decode("ascii") if isinstance(raw_name, bytes) else raw_name
        uuid = "GPU-" + bytes(raw_uuid).hex()
        runtime = cp.cuda.runtime.runtimeGetVersion()
        driver = cp.cuda.runtime.driverGetVersion()
        environment = {
            "os": platform.system(), "kernel": platform.release(),
            "python_version": platform.python_version(), "cupy_version": cp.__version__,
            "cuda_driver_version": str(driver), "cuda_runtime_version": str(runtime),
            "gpu_name": name, "gpu_uuid": uuid,
            "compute_capability": f"{major}.{minor}",
            "device_memory_bytes": str(total_memory),
        }
    except Exception as exc:
        raise BenchmarkError("INVENTORY") from exc
    return {"mode": "inventory", "installation": installation, "environment": environment}


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    try:
        output = worker_main(arguments)
        data = canonical(output)
        if len(data) > MAX_WORKER_OUTPUT_BYTES:
            raise BenchmarkError("WORKER_OUTPUT")
        sys.stdout.buffer.write(data)
        return 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return 1


def run_worker(
    mode: str,
    *,
    cwd: Path,
    cache_dir: Path,
    scenario_directory: Path,
    wheel: Path,
) -> dict[str, Any]:
    completed = run_bounded(
        worker_argv(mode, scenario_directory=scenario_directory, wheel=wheel),
        cwd=cwd,
        env=worker_environment(cache_dir),
        timeout_seconds=WORKER_TIMEOUT_SECONDS,
        output_limit=MAX_WORKER_OUTPUT_BYTES,
    )
    return parse_worker_output(completed.stdout, completed.stderr)


def _normal_measurement(value: Any) -> tuple[str, bytes]:
    if type(value) is not dict or type(value.get("duration_ns")) is not str:
        raise BenchmarkError("WORKER_MEASUREMENT")
    duration = value["duration_ns"]
    if re.fullmatch(r"[1-9][0-9]{0,19}", duration) is None:
        raise BenchmarkError("WORKER_MEASUREMENT")
    result = value.get("result")
    if type(result) is not dict:
        raise BenchmarkError("WORKER_MEASUREMENT")
    if value.get("stage_profile") is not None:
        raise BenchmarkError("TIMING_CONFLATION")
    return duration, canonical(result)


def collect_steady_samples(run_once: Any) -> tuple[str, ...]:
    if not callable(run_once):
        raise BenchmarkError("WORKER_MEASUREMENT")
    _warmup_duration, expected = _normal_measurement(run_once("steady"))
    samples = []
    for _ordinal in range(30):
        duration, result = _normal_measurement(run_once("steady"))
        if result != expected:
            raise BenchmarkError("ANALYTICAL_MISMATCH")
        samples.append(duration)
    return tuple(samples)


def collect_stage_profile(run_once: Any) -> dict[str, str]:
    if not callable(run_once):
        raise BenchmarkError("WORKER_MEASUREMENT")
    value = run_once("stage")
    if type(value) is not dict or type(value.get("result")) is not dict:
        raise BenchmarkError("WORKER_MEASUREMENT")
    profile = value.get("stage_profile")
    if type(profile) is not dict or not profile:
        raise BenchmarkError("STAGE_PROFILE")
    if any(type(key) is not str or type(item) is not str for key, item in profile.items()):
        raise BenchmarkError("STAGE_PROFILE")
    return dict(profile)


def evidence_stage_durations(profile: dict[str, str]) -> dict[str, str]:
    mapping = {
        "h2d_gpu_ns": "h2d_ns",
        "simulate_gpu_ns": "simulate_ns",
        "reduction_gpu_ns": "reduction_ns",
        "d2h_gpu_ns": "d2h_ns",
    }
    if type(profile) is not dict or set(profile) != set(mapping):
        raise BenchmarkError("STAGE_PROFILE")
    if any(
        type(value) is not str or re.fullmatch(r"(?:0|[1-9][0-9]{0,19})", value) is None
        for value in profile.values()
    ):
        raise BenchmarkError("STAGE_PROFILE")
    return {destination: profile[source] for source, destination in mapping.items()}


def run_bounded(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
    output_limit: int,
    before_group_kill: Any = None,
) -> subprocess.CompletedProcess[bytes]:
    if (
        type(argv) is not list
        or not argv
        or any(type(item) is not str or not item for item in argv)
        or not isinstance(cwd, Path)
        or not cwd.is_dir()
        or type(env) is not dict
        or any(type(key) is not str or type(value) is not str for key, value in env.items())
        or type(timeout_seconds) not in (int, float)
        or timeout_seconds <= 0
        or type(output_limit) is not int
        or output_limit < 1
        or before_group_kill is not None and not callable(before_group_kill)
    ):
        raise BenchmarkError("PROCESS_ARGUMENT")
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise BenchmarkError("PROCESS_START") from exc

    threads: list[threading.Thread] = []
    streams: list[Any] = []
    cleanup_seconds = min(0.5, float(timeout_seconds))
    group_killed = False

    def kill_group() -> None:
        nonlocal group_killed
        if group_killed:
            return
        group_killed = True
        try:
            if before_group_kill is not None:
                before_group_kill(process.pid)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def cleanup() -> None:
        kill_group()
        try:
            process.wait(timeout=cleanup_seconds)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.monotonic() + cleanup_seconds
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        for stream in streams:
            try:
                stream.close()
            except (AttributeError, OSError):
                pass
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))

    try:
        if process.stdout is None or process.stderr is None:
            raise BenchmarkError("PROCESS_PIPE")
        streams.extend((process.stdout, process.stderr))
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        overflow = threading.Event()
        reader_errors: list[BaseException] = []

        def drain(name: str, stream: Any) -> None:
            try:
                while True:
                    block = stream.read(64 * 1024)
                    if not block:
                        return
                    remaining = output_limit - len(buffers[name])
                    if len(block) > remaining:
                        buffers[name].extend(block[:max(0, remaining)])
                        overflow.set()
                        kill_group()
                        return
                    buffers[name].extend(block)
            except BaseException as exc:
                reader_errors.append(exc)
                kill_group()

        threads.extend([
            threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
            threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
        ])
        for thread in threads:
            thread.start()
        try:
            return_code = process.wait(timeout=float(timeout_seconds))
        except subprocess.TimeoutExpired as exc:
            cleanup()
            raise BenchmarkError("PROCESS_TIMEOUT") from exc

        deadline = time.monotonic() + cleanup_seconds
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            cleanup()
            raise BenchmarkError("PROCESS_PIPE")
        if overflow.is_set() or reader_errors:
            cleanup()
            raise BenchmarkError("PROCESS_OUTPUT")
        kill_group()
        if return_code != 0:
            raise BenchmarkError("PROCESS_EXIT")
        return subprocess.CompletedProcess(
            argv,
            return_code,
            bytes(buffers["stdout"]),
            bytes(buffers["stderr"]),
        )
    except BaseException:
        cleanup()
        raise


def _card_id(card: str) -> int:
    if type(card) is not str or CARD_PATTERN.fullmatch(card) is None:
        raise BenchmarkError("SCENARIO_CARD")
    return RANKS.index(card[0]) * 4 + SUITS.index(card[1])


def _validate_scenario(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != SCENARIO_KEYS:
        raise BenchmarkError("SCENARIO_KEYS")
    if value["format_version"] != SCENARIO_FORMAT:
        raise BenchmarkError("SCENARIO_VERSION")
    street = value["street"]
    opponents = value["opponent_count"]
    if street not in STREETS or opponents not in OPPONENT_COUNTS:
        raise BenchmarkError("SCENARIO_DOMAIN")
    if value["id"] != f"v1-{street}-o{opponents}":
        raise BenchmarkError("SCENARIO_ID")
    if type(value["seed"]) is not str or HEX_SEED_PATTERN.fullmatch(value["seed"]) is None:
        raise BenchmarkError("SCENARIO_SEED")
    hero = value["hero_cards"]
    board = value["board_cards"]
    if type(hero) is not list or type(board) is not list:
        raise BenchmarkError("SCENARIO_CARDS")
    if len(hero) != 2 or len(board) != BOARD_COUNTS[street]:
        raise BenchmarkError("SCENARIO_TOPOLOGY")
    cards = hero + board
    ids = [_card_id(card) for card in cards]
    if len(set(ids)) != len(ids):
        raise BenchmarkError("SCENARIO_DUPLICATE_CARD")
    normalized = dict(value)
    normalized["hero_cards"] = [card for _, card in sorted(zip(ids[:2], hero))]
    board_ids = ids[2:]
    normalized["board_cards"] = [card for _, card in sorted(zip(board_ids, board))]
    return normalized


def load_scenario(path: Path) -> dict[str, Any]:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise BenchmarkError("SCENARIO_PATH")
    data = path.read_bytes()
    value = _validate_scenario(strict_json(data))
    if canonical(value) != data:
        raise BenchmarkError("SCENARIO_CARD_ORDER")
    if path.name != value["id"].removeprefix("v1-") + ".json":
        raise BenchmarkError("SCENARIO_FILENAME")
    return value


def load_scenarios(directory: Path) -> tuple[dict[str, Any], ...]:
    if not isinstance(directory, Path) or directory.is_symlink() or not directory.is_dir():
        raise BenchmarkError("SCENARIO_DIRECTORY")
    values = tuple(load_scenario(path) for path in directory.glob("*.json"))
    ordered = tuple(sorted(values, key=lambda row: (STREETS.index(row["street"]), int(row["opponent_count"]))))
    expected = tuple((street, opponents) for street in STREETS for opponents in OPPONENT_COUNTS)
    if tuple((row["street"], row["opponent_count"]) for row in ordered) != expected:
        raise BenchmarkError("SCENARIO_MATRIX")
    return ordered


def scenario_manifest(directory: Path) -> dict[str, Any]:
    load_scenarios(directory)
    rows = []
    for path in sorted(directory.glob("*.json"), key=lambda item: item.name):
        value = load_scenario(path)
        data = path.read_bytes()
        rows.append({
            "id": value["id"],
            "path": path.name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": str(len(data)),
        })
    return {
        "format_version": "phase6c-scenario-manifest-v1",
        "scenarios": rows,
    }


def scenario_manifest_sha256(manifest: dict[str, Any]) -> str:
    if type(manifest) is not dict or set(manifest) != {"format_version", "scenarios"}:
        raise BenchmarkError("SCENARIO_MANIFEST")
    return hashlib.sha256(canonical(manifest)).hexdigest()


def expand_matrix(scenarios: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    cells: list[dict[str, Any]] = []
    for scenario in scenarios:
        for trials in REQUESTED_TRIALS:
            cells.append({
                **scenario,
                "backend": "cuda",
                "cell_id": f"{scenario['id']}-n{trials}",
                "requested_trials": trials,
            })
    return tuple(cells)


def preregistration_record() -> dict[str, Any]:
    return {
        "analytical_reference": "separate_process_isolated_cache_public_cuda_repeatability_reference_not_independent_correctness_proof",
        "benchmark_id": BENCHMARK_ID,
        "cache_modes": ["cold", "warm", "steady"],
        "end_to_end_scope": "one_normal_explicit_solve_cuda_host_monotonic",
        "format_version": "phase6c-benchmark-preregistration-v1",
        "outlier_policy": "none_all_valid_samples_retained",
        "percentile_method": "nearest_rank_ceil_p_times_n_div_100_minus_1",
        "percentiles": ["5", "50", "95"],
        "steady_repetitions": "30",
        "steady_warmups": "1",
        "stage_batch_blocks": str(STAGE_BATCH_BLOCKS),
        "stage_threads_per_block": "128",
        "stage_batch_counts_by_requested_trials": {
            "10000": "1", "100000": "4", "500000": "16", "1000000": "31",
        },
        "streets": list(STREETS),
        "opponent_counts": list(OPPONENT_COUNTS),
        "requested_trials": list(REQUESTED_TRIALS),
        "startup_canary_cell_id": "v1-preflop-o1-n10000",
        "throughput_scale": "0.001",
    }


def nearest_rank(samples: tuple[int, ...], percentile: int) -> int:
    if (
        type(samples) is not tuple
        or not samples
        or type(percentile) is not int
        or not 1 <= percentile <= 100
        or any(type(value) is not int or value <= 0 for value in samples)
    ):
        raise BenchmarkError("SAMPLES")
    ordered = sorted(samples)
    index = ceil(percentile * len(ordered) / 100) - 1
    return ordered[index]


def samples_per_second(requested_trials: int, p50_duration_ns: int) -> str:
    if (
        type(requested_trials) is not int
        or type(p50_duration_ns) is not int
        or requested_trials <= 0
        or p50_duration_ns <= 0
    ):
        raise BenchmarkError("THROUGHPUT")
    value = Decimal(requested_trials) * Decimal(1_000_000_000) / Decimal(p50_duration_ns)
    return format(value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP), "f")


class CupyStageObserver:
    """Benchmark-only CUDA-event observer with preallocated transfer buffers."""

    _STAGES = ("simulate", "reduction", "d2h")

    def __init__(self, cp: Any, *, batches: int, aggregate_bytes: int):
        if type(batches) is not int or batches < 1:
            raise BenchmarkError("OBSERVER_BATCHES")
        if type(aggregate_bytes) is not int or aggregate_bytes < 1:
            raise BenchmarkError("OBSERVER_AGGREGATE")
        import numpy as np  # type: ignore[import-not-found]

        self._cp = cp
        self._np = np
        self._batches = batches
        self._stream = cp.cuda.get_current_stream()
        self._events = {
            "h2d": [(cp.cuda.Event(), cp.cuda.Event())],
            **{
                stage: [(cp.cuda.Event(), cp.cuda.Event()) for _ in range(batches)]
                for stage in self._STAGES
            },
        }
        self._expected = [("h2d", "start", 0), ("h2d", "end", 0)]
        for ordinal in range(batches):
            for stage in self._STAGES:
                self._expected.extend([
                    (stage, "start", ordinal),
                    (stage, "end", ordinal),
                ])
        self._position = 0
        self._outputs = []
        for _ in range(batches):
            memory = cp.cuda.alloc_pinned_memory(aggregate_bytes)
            self._outputs.append(np.frombuffer(memory, dtype=np.uint8, count=aggregate_bytes))
        self._input_devices: tuple[Any, Any] | None = None
        self._input_hosts: tuple[Any, Any] | None = None

    def prepare_inputs(self, hero: tuple[int, int], board: tuple[int, ...]) -> None:
        if self._input_devices is not None:
            raise BenchmarkError("OBSERVER_INPUT_REUSE")
        board_values = board if board else (0,)
        hero_memory = self._cp.cuda.alloc_pinned_memory(len(hero))
        board_memory = self._cp.cuda.alloc_pinned_memory(len(board_values))
        hero_host = self._np.frombuffer(hero_memory, dtype=self._np.uint8, count=len(hero))
        board_host = self._np.frombuffer(
            board_memory,
            dtype=self._np.uint8,
            count=len(board_values),
        )
        hero_host[:] = hero
        board_host[:] = board_values
        hero_device = self._cp.empty(len(hero), dtype=self._cp.uint8)
        board_device = self._cp.empty(len(board_values), dtype=self._cp.uint8)
        self._input_hosts = (hero_host, board_host)
        self._input_devices = (hero_device, board_device)

    def copy_inputs(self) -> tuple[Any, Any]:
        if self._input_hosts is None or self._input_devices is None:
            raise BenchmarkError("OBSERVER_INPUTS")
        for device, host in zip(self._input_devices, self._input_hosts):
            device.set(host, stream=self._stream)
        return self._input_devices

    def boundary(self, stage: str, edge: str, ordinal: int) -> None:
        observed = (stage, edge, ordinal)
        if self._position >= len(self._expected) or observed != self._expected[self._position]:
            raise BenchmarkError("OBSERVER_ORDER")
        event_index = 0 if edge == "start" else 1
        self._events[stage][ordinal][event_index].record(self._stream)
        self._position += 1

    def copy_to_host(self, final: Any, ordinal: int) -> Any:
        if type(ordinal) is not int or not 0 <= ordinal < self._batches:
            raise BenchmarkError("OBSERVER_COPY")
        output = self._outputs[ordinal]
        final.get(out=output, stream=self._stream, blocking=False)
        return output

    def finish(self) -> dict[str, str]:
        if self._position != len(self._expected):
            raise BenchmarkError("OBSERVER_INCOMPLETE")
        self._events["d2h"][-1][1].synchronize()
        values: dict[str, str] = {}
        for stage, pairs in self._events.items():
            milliseconds = sum(
                Decimal(str(self._cp.cuda.get_elapsed_time(start, end)))
                for start, end in pairs
            )
            nanoseconds = (milliseconds * Decimal(1_000_000)).quantize(
                Decimal("1"),
                rounding=ROUND_HALF_UP,
            )
            if nanoseconds < 0:
                raise BenchmarkError("OBSERVER_DURATION")
            values[f"{stage}_gpu_ns"] = str(nanoseconds)
        return dict(sorted(values.items()))


if __name__ == "__main__":
    raise SystemExit(main())
