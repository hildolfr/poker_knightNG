#!/usr/bin/env python3
"""Private Phase 6C controller; deliberately stdlib-only and CUDA/package inert."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

MAX_OUTPUT_BYTES = 2 * 1024 * 1024
WORKER_LIMIT = 2 * 1024 * 1024
WORKER_TIMEOUT = 7200
MODES = ("inventory", "expected", "cold", "warm", "steady", "stage")
STREETS = ("preflop", "flop", "turn", "river")
TRIALS = ("10000", "100000", "500000", "1000000")
STAGE_BATCH_BLOCKS = 256
STAGE_THREADS = 128
QUALIFICATION_FILES = {
    "phase5b_qualification": "validation/holdem/v1/cuda_release_qualification.json",
    "phase5b_manifest": "validation/holdem/v1/manifests/cuda_release_qualification.sha256",
    "phase5c_qualification": "validation/holdem/v1/cuda_statistical_release_qualification.json",
    "phase5c_manifest": "validation/holdem/v1/manifests/cuda_statistical_release_qualification.sha256",
}
HEX = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
UINT = re.compile(r"(?:0|[1-9][0-9]*)\Z", re.ASCII)
POSITIVE = re.compile(r"[1-9][0-9]*\Z", re.ASCII)
GPU_UUID = re.compile(
    r"GPU-(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\Z",
    re.ASCII | re.IGNORECASE,
)


class ControllerError(ValueError):
    """Stable public controller failure; detailed causes are intentionally hidden."""


@dataclass(frozen=True)
class ControllerConfig:
    repo_root: Path
    wheel: Path
    sdist: Path
    lockfile: Path
    scenario_directory: Path
    scenario_manifest: Path
    output: Path


@dataclass(frozen=True)
class WorkerCapture:
    value: dict[str, Any]
    stdout: bytes
    stderr: bytes


def canonical(value: Any) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ControllerError("CONTROLLER") from exc


def _strict(data: bytes) -> dict[str, Any]:
    if type(data) is not bytes or not data or len(data) > WORKER_LIMIT or not data.endswith(b"\n") or b"\r" in data:
        raise ControllerError("CONTROLLER")
    try:
        value = json.loads(data.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ControllerError("CONTROLLER") from exc
    if type(value) is not dict or canonical(value) != data:
        raise ControllerError("CONTROLLER")
    return value


def _load_tool(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ControllerError("CONTROLLER")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _has_symlink_component(path: Path) -> bool:
    if not path.is_absolute():
        return True
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def run_bounded(argv: list[str], *, cwd: Path, env: dict[str, str], timeout_seconds: int, output_limit: int) -> subprocess.CompletedProcess[bytes]:
    try:
        benchmark = _load_tool(Path(__file__).with_name("benchmark_equity.py"), "_phase6c_benchmark_runtime")
        return benchmark.run_bounded(
            argv, cwd=cwd, env=env, timeout_seconds=timeout_seconds,
            output_limit=output_limit,
        )
    except Exception as exc:
        raise ControllerError("CONTROLLER") from exc


def _worker_env(cache: Path) -> dict[str, str]:
    return {"CUPY_CACHE_DIR": str(cache), "CUDA_CACHE_PATH": str(cache / "driver"), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PKNG_BENCHMARK_WORKER": "1", "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1"}


def run_worker_capture(mode: str, *, config: ControllerConfig, cache_dir: Path) -> WorkerCapture:
    if mode not in MODES:
        raise ControllerError("CONTROLLER")
    script = config.repo_root / "tools" / "benchmark_equity.py"
    result = run_bounded([sys.executable, os.fspath(script), "--worker", "--mode", mode, "--scenario-dir", os.fspath(config.scenario_directory), "--wheel", os.fspath(config.wheel)], cwd=cache_dir.parent, env=_worker_env(cache_dir), timeout_seconds=WORKER_TIMEOUT, output_limit=WORKER_LIMIT)
    if result.stderr:
        raise ControllerError("CONTROLLER")
    return WorkerCapture(_strict(result.stdout), result.stdout, result.stderr)


def prepare_cold_cache(path: Path) -> None:
    try:
        _load_tool(Path(__file__).with_name("benchmark_equity.py"), "_phase6c_cache_prepare").prepare_cold_cache(path)
    except Exception as exc:
        raise ControllerError("CONTROLLER") from exc


def seal_cold_cache(path: Path, *, cold_result_sha256: str) -> dict[str, str]:
    try:
        return _load_tool(Path(__file__).with_name("benchmark_equity.py"), "_phase6c_cache_seal").seal_cold_cache(
            path, cold_result_sha256=cold_result_sha256,
        )
    except Exception as exc:
        raise ControllerError("CONTROLLER") from exc


def verify_warm_cache(path: Path) -> dict[str, str]:
    try:
        return _load_tool(Path(__file__).with_name("benchmark_equity.py"), "_phase6c_cache_verify").verify_warm_cache(path)
    except Exception as exc:
        raise ControllerError("CONTROLLER") from exc


def _wheel_closure(path: Path) -> list[dict[str, str]]:
    try:
        verifier = _load_tool(
            Path(__file__).with_name("verify_cuda_benchmark_evidence.py"),
            "_phase6c_binding_verifier",
        )
        return verifier._archive_closure(path)
    except Exception as exc:
        raise ControllerError("CONTROLLER") from exc


def admit_gpu(*, cwd: Path) -> dict[str, Any]:
    base = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": os.defpath}
    try:
        gpu = run_bounded(["/usr/bin/nvidia-smi", "--query-gpu=uuid,memory.free,memory.total", "--format=csv,noheader,nounits"], cwd=cwd, env=base, timeout_seconds=10, output_limit=4096)
        apps = run_bounded(["/usr/bin/nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], cwd=cwd, env=base, timeout_seconds=10, output_limit=4096)
        if gpu.stderr or apps.stderr or apps.stdout != b"": raise ControllerError("ADMISSION")
        rows = gpu.stdout.decode("ascii", "strict").splitlines()
        if len(rows) != 1: raise ControllerError("ADMISSION")
        parts = [x.strip() for x in rows[0].split(",")]
        if len(parts) != 3 or not GPU_UUID.fullmatch(parts[0]) or not all(POSITIVE.fullmatch(x) for x in parts[1:]): raise ControllerError("ADMISSION")
        normalized_uuid = "GPU-" + parts[0][4:].replace("-", "").lower()
        # nvidia-smi returns MiB: controller records bytes and refuses a no-headroom GPU.
        free, total = int(parts[1]) * 1048576, int(parts[2]) * 1048576
        if free < 2 * 1024**3 or free > total: raise ControllerError("ADMISSION")
        return {
            "gpu_uuid": normalized_uuid,
            "free_bytes": str(free),
            "total_bytes": str(total),
            "compute_applications": [],
            "gpu_snapshot_hex": gpu.stdout.hex(),
            "compute_snapshot_hex": apps.stdout.hex(),
            "gpu_snapshot_sha256": hashlib.sha256(gpu.stdout).hexdigest(),
            "compute_snapshot_sha256": hashlib.sha256(apps.stdout).hexdigest(),
        }
    except (ControllerError, UnicodeError, ValueError): raise
    except BaseException as exc: raise ControllerError("ADMISSION") from exc


def _aggregate(trials: str, samples: list[str]) -> dict[str, str]:
    if len(samples) != 30 or any(not POSITIVE.fullmatch(x) for x in samples): raise ControllerError("CONTROLLER")
    ordered = sorted(map(int, samples)); p50 = ordered[14]
    rate = (Decimal(trials) * Decimal(1_000_000_000) / Decimal(p50)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return {"count": "30", "minimum_ns": str(ordered[0]), "p5_ns": str(ordered[1]), "p50_ns": str(p50), "p95_ns": str(ordered[28]), "maximum_ns": str(ordered[-1]), "throughput_per_second": format(rate, "f")}


def _sample(value: dict[str, Any]) -> dict[str, str]:
    digest, duration = value.get("analytical_sha256"), value.get("duration_ns")
    if not isinstance(digest, str) or not HEX.fullmatch(digest) or not isinstance(duration, str) or not POSITIVE.fullmatch(duration): raise ControllerError("CONTROLLER")
    rate = (Decimal(10_000) * Decimal(1_000_000_000) / Decimal(duration)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return {"analytical_sha256": digest, "duration_ns": duration, "throughput_per_second": format(rate, "f")}


def _file(path: Path) -> dict[str, str]:
    if _has_symlink_component(path.absolute()) or not path.is_file():
        raise ControllerError("CONTROLLER")
    try: data = path.read_bytes()
    except OSError as exc: raise ControllerError("CONTROLLER") from exc
    return {"basename": path.name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": str(len(data))}


def _source(config: ControllerConfig) -> dict[str, str]:
    def git(arguments: list[str]) -> str:
        result = run_bounded(["git", "-C", os.fspath(config.repo_root), *arguments], cwd=config.repo_root, env={"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}, timeout_seconds=10, output_limit=4096)
        if result.stderr: raise ControllerError("CONTROLLER")
        return result.stdout.decode("ascii", "strict").strip()
    head, branch, status = git(["rev-parse", "--verify", "HEAD"]), git(["symbolic-ref", "--quiet", "--short", "HEAD"]), git(["status", "--porcelain=v1", "--untracked-files=all"])
    if not re.fullmatch(r"[0-9a-f]{40}", head) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", branch) or status: raise ControllerError("CONTROLLER")
    return {"git_sha":head,"branch":branch,"clean":"true","benchmark_tool_sha256":hashlib.sha256((config.repo_root / "tools/benchmark_equity.py").read_bytes()).hexdigest(),"cuda_runtime_source_sha256":hashlib.sha256((config.repo_root / "src/poker_knight_ng/_cuda_runtime.py").read_bytes()).hexdigest()}


def assemble(
    config: ControllerConfig,
    results: dict[str, WorkerCapture],
    admission_before: dict[str, Any],
    admission_after: dict[str, Any],
    source_identity: dict[str, str],
    cold_seal: dict[str, Any],
    warm_verified_seal: dict[str, Any],
) -> dict[str, Any]:
    payloads = {mode: results[mode].value for mode in MODES}
    if any(payloads[mode].get("mode") != mode for mode in MODES):
        raise ControllerError("CONTROLLER")
    environment = payloads["inventory"].get("environment")
    if (
        type(environment) is not dict
        or admission_before["gpu_uuid"] != admission_after["gpu_uuid"]
        or environment.get("gpu_uuid") != admission_before["gpu_uuid"]
    ):
        raise ControllerError("CONTROLLER")
    expected = payloads["expected"].get("analytical_sha256s")
    steady = payloads["steady"].get("cells")
    stage_payload = payloads["stage"]
    if set(stage_payload) != {"mode", "planned_batch_blocks", "batch_counts", "cells"}:
        raise ControllerError("CONTROLLER")
    plan_text = stage_payload.get("planned_batch_blocks")
    if plan_text != str(STAGE_BATCH_BLOCKS):
        raise ControllerError("CONTROLLER")
    plan = STAGE_BATCH_BLOCKS
    stage = stage_payload.get("cells")
    expected_ids = [
        f"v1-{street}-o{opponents}-n{trials}"
        for street in STREETS
        for opponents in ("1", "3", "6")
        for trials in TRIALS
    ]
    expected_batch_counts = {
        cell_id: str((int(cell_id.rsplit("-n", 1)[1]) + plan * STAGE_THREADS - 1) // (plan * STAGE_THREADS))
        for cell_id in expected_ids
    }
    if (
        type(expected) is not dict
        or type(steady) is not dict
        or type(stage) is not dict
        or set(expected) != set(expected_ids)
        or set(steady) != set(expected_ids)
        or set(stage) != set(expected_ids)
        or stage_payload.get("batch_counts") != expected_batch_counts
    ):
        raise ControllerError("CONTROLLER")
    matrix = []
    for cell_id in expected_ids:
        digest = expected[cell_id]
        match = re.fullmatch(
            r"v1-(preflop|flop|turn|river)-o(1|3|6)-n(10000|100000|500000|1000000)",
            cell_id,
        )
        steady_cell = steady[cell_id]
        stage_cell = stage[cell_id]
        if (
            type(digest) is not str
            or HEX.fullmatch(digest) is None
            or match is None
            or type(steady_cell) is not dict
            or type(stage_cell) is not dict
            or steady_cell.get("warmup_analytical_sha256") != digest
            or any(value != digest for value in steady_cell.get("analytical_sha256s", []))
            or stage_cell.get("analytical_sha256") != digest
            or type(steady_cell.get("durations_ns")) is not list
        ):
            raise ControllerError("CONTROLLER")
        matrix.append({
            "cell_id": cell_id,
            "street": match[1],
            "opponent_count": match[2],
            "requested_trials": match[3],
            "backend": "cuda",
            "seed": "0x0123456789abcdef",
            "expected_analytical_sha256": digest,
            "steady": {
                **steady_cell,
                "aggregate": _aggregate(match[3], steady_cell["durations_ns"]),
            },
            "stage": stage_cell,
        })
    cold_sample = _sample(payloads["cold"])
    warm_sample = _sample(payloads["warm"])
    if (
        cold_sample["analytical_sha256"] != warm_sample["analytical_sha256"]
        or cold_sample["analytical_sha256"] != matrix[0]["expected_analytical_sha256"]
    ):
        raise ControllerError("CONTROLLER")
    cache_classes = {
        "inventory": "inventory",
        "expected": "expected-isolated",
        "cold": "startup-cold",
        "warm": "startup-warm",
        "steady": "steady-isolated",
        "stage": "stage-isolated",
    }
    workers = []
    for ordinal, mode in enumerate(MODES):
        capture = results[mode]
        digest = hashlib.sha256(capture.stdout).hexdigest()
        workers.append({
            "cache_class": cache_classes[mode],
            "mode": mode,
            "ordinal": str(ordinal),
            "fresh_process": "true",
            "exit_code": "0",
            "stdout_bytes": str(len(capture.stdout)),
            "stderr_bytes": str(len(capture.stderr)),
            "stdout_sha256": digest,
            "stderr_sha256": hashlib.sha256(capture.stderr).hexdigest(),
            "output_limit_bytes": str(WORKER_LIMIT),
            "payload": capture.value,
            "payload_sha256": digest,
        })
    return {
        "format_version": "phase6c-private-evidence-v1",
        "benchmark_id": "holdem-v1-cuda-baseline-1",
        "source": source_identity,
        "artifacts": {
            "wheel": _file(config.wheel),
            "sdist": _file(config.sdist),
            "lock": _file(config.lockfile),
            "scenario_manifest": _file(config.scenario_manifest),
            **{
                name: _file(config.repo_root / relative)
                for name, relative in QUALIFICATION_FILES.items()
            },
            "installed_wheel_byte_closure": _wheel_closure(config.wheel),
        },
        "environment": environment,
        "admission": {"before": admission_before, "after": admission_after},
        "workers": workers,
        "startup_cache": {
            "cold_worker_ordinal": "2",
            "warm_worker_ordinal": "3",
            "relationship": "shared_run_owned_0700_prepared_empty_sealed_after_cold_verified_immediately_before_warm",
            "cold_seal": cold_seal,
            "warm_verified_seal": warm_verified_seal,
        },
        "startup": {
            "canary_cell_id": "v1-preflop-o1-n10000",
            "cold": cold_sample,
            "warm": warm_sample,
        },
        "matrix": matrix,
        "gates": {
            key: "passed"
            for key in (
                "checkout", "artifacts", "environment", "admission",
                "worker", "matrix", "statistics",
            )
        },
    }


def write_evidence_exclusive(path: Path, record: dict[str, Any]) -> None:
    data = canonical(record)
    if len(data) > MAX_OUTPUT_BYTES or path.exists() or path.is_symlink(): raise ControllerError("OUTPUT")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + next(tempfile._get_candidate_names()))
    published = False
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            remaining = memoryview(data)
            while remaining:
                written = os.write(fd, remaining)
                if written < 1:
                    raise OSError("short evidence write")
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(temporary, path)
        published = True
        os.unlink(temporary)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        if published:
            path.unlink(missing_ok=True)
        raise


def _validate_config(config: ControllerConfig) -> None:
    if type(config) is not ControllerConfig:
        raise ControllerError("CONFIG")
    paths = (
        config.repo_root, config.wheel, config.sdist, config.lockfile,
        config.scenario_directory, config.scenario_manifest, config.output,
    )
    if any(not isinstance(path, Path) or not path.is_absolute() for path in paths):
        raise ControllerError("CONFIG")
    if any(_has_symlink_component(path) for path in paths):
        raise ControllerError("CONFIG")
    if (
        config.repo_root.is_symlink()
        or not config.repo_root.is_dir()
        or config.scenario_directory.is_symlink()
        or not config.scenario_directory.is_dir()
        or config.output.exists()
        or config.output.is_symlink()
        or config.output.parent.is_symlink()
        or not config.output.parent.is_dir()
    ):
        raise ControllerError("CONFIG")
    for path in (config.wheel, config.sdist, config.lockfile, config.scenario_manifest):
        if path.is_symlink() or not path.is_file():
            raise ControllerError("CONFIG")
    try:
        root = config.repo_root.resolve(strict=True)
        for path in (
            config.wheel,
            config.sdist,
            config.lockfile,
            config.scenario_directory,
            config.scenario_manifest,
        ):
            path.resolve(strict=True).relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ControllerError("CONFIG") from exc


def _verify_final_record(config: ControllerConfig, record: dict[str, Any]) -> None:
    try:
        verifier = _load_tool(
            Path(__file__).with_name("verify_cuda_benchmark_evidence.py"),
            "_phase6c_final_verifier",
        )
        context = verifier.VerificationContext(
            repo_root=config.repo_root,
            wheel=config.wheel,
            sdist=config.sdist,
            lockfile=config.lockfile,
            scenario_directory=config.scenario_directory,
            scenario_manifest=config.scenario_manifest,
        )
        verifier.verify_bound_record(record, verifier._load_schema(verifier.SCHEMA_PATH), context)
    except Exception as exc:
        raise ControllerError("VERIFICATION") from exc


def run(config: ControllerConfig) -> dict[str, Any]:
    _validate_config(config)
    caches = Path(tempfile.mkdtemp(prefix="phase6c-", dir=config.output.parent))
    try:
        source_before = _source(config)
        before = admit_gpu(cwd=config.repo_root); results={}
        cold_seal = None
        warm_verified_seal = None
        for mode in MODES:
            cache = caches / ("cold-warm" if mode in ("cold","warm") else mode)
            if mode == "cold": prepare_cold_cache(cache)
            if mode == "warm":
                warm_verified_seal = verify_warm_cache(cache)
                if warm_verified_seal != cold_seal:
                    raise ControllerError("CACHE_CHANGED")
            if not cache.exists(): cache.mkdir(mode=0o700)
            results[mode]=run_worker_capture(mode, config=config, cache_dir=cache)
            if mode == "cold":
                cold_seal = seal_cold_cache(
                    cache,
                    cold_result_sha256=results[mode].value.get("analytical_sha256", ""),
                )
        after = admit_gpu(cwd=config.repo_root)
        source_after = _source(config)
        if source_after != source_before:
            raise ControllerError("SOURCE_CHANGED")
        if type(cold_seal) is not dict or type(warm_verified_seal) is not dict:
            raise ControllerError("CACHE_EVIDENCE")
        record = assemble(
            config, results, before, after, source_before,
            cold_seal, warm_verified_seal,
        )
        _verify_final_record(config, record)
        write_evidence_exclusive(config.output, record)
        return record
    except BaseException:
        raise
    finally: shutil.rmtree(caches, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 14 or arguments[0::2] != ["--repo-root", "--wheel", "--sdist", "--lockfile", "--scenario-dir", "--scenario-manifest", "--output"]:
        return 2
    try:
        run(ControllerConfig(*(Path(value).absolute() for value in arguments[1::2])))
    except Exception:
        print("Phase 6C private benchmark evidence: FAIL", file=sys.stderr)
        return 1
    print("Phase 6C private benchmark evidence: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
