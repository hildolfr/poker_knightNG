#!/usr/bin/env python3.13
"""Fail-closed Phase 5C CUDA statistical qualification.

The module import path is standard-library only. GPU/package imports occur only in
an explicitly gated private worker added below this CPU-verifiable core.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, getcontext
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any

THREADS = 128
TRIALS = 2000
STATISTICAL_ROW_SHA256 = "c66d625292d6548b334ea819efa9021c89163e45dbd4f15d6dd170e0bcc1f371"
STATISTICAL_ID = "river-high-card-wtl-2000"
STATISTICAL_SEED = "0x0000000000000001"
STATISTICAL_HASH = "7e088bbb705b1436c35c7ffbd6577bcac4656cbb8b60fd582d32ca0ce8163037"
EXPECTED_BRANCH = "revival/phase-6c-optimize-evaluator"
WILSON_Z = Decimal("4.891638475698591")
ALPHA = Decimal("0.000001")
GEOMETRIES = {
    "capacity_1": {"batch_blocks": 1, "vram_budget_bytes": None, "capacity": 1},
    "budget_capacity_3": {"batch_blocks": 256, "vram_budget_bytes": 4864, "capacity": 3},
    "capacity_7": {"batch_blocks": 7, "vram_budget_bytes": None, "capacity": 7},
    "capacity_256": {"batch_blocks": 256, "vram_budget_bytes": None, "capacity": 256},
}
HARNESS_BINDINGS = (
    "pyproject.toml", "uv.lock", "src/poker_knight_ng/_cuda_runtime.py",
    "src/poker_knight_ng/cuda-sources/philox.cuh", "src/poker_knight_ng/cuda-sources/dealer.cuh",
    "src/poker_knight_ng/cuda-sources/cards.cuh", "src/poker_knight_ng/cuda-sources/evaluator.cuh",
    "src/poker_knight_ng/cuda-sources/simulate.cuh", "src/poker_knight_ng/cuda-sources/reduce.cuh",
    "src/poker_knight_ng/cuda-sources/deterministic_kernels.cu", "tools/generate_rng_seed_bank.py",
    "tools/qualify_gpu.py", "tools/qualify_gpu_statistics.py",
    "validation/holdem/v1/cuda_statistical_qualification.schema.json",
    "validation/holdem/v1/STATISTICAL_QUALIFICATION.md", "validation/holdem/v1/SPEC.md",
    "validation/holdem/v1/rng_seed_bank.json", "validation/holdem/v1/manifests/rng_seed_bank.sha256",
)
DECIMAL = re.compile(r"0|[1-9][0-9]*")
TIE_KEYS = tuple(str(i) for i in range(1, 7))
CATEGORY_KEYS = (
    "high_card", "one_pair", "two_pair", "three_of_a_kind", "straight",
    "flush", "full_house", "four_of_a_kind", "straight_flush",
)
AGGREGATE_KEYS = {
    "completed_trials", "equity_share_units", "hero_category_counts", "losses",
    "rejection_count", "tie_by_other_winners", "unique_wins",
}


class QualificationError(Exception):
    """Stable Phase 5C qualification failure."""
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def preregistration_record() -> dict[str, str]:
    """Return the single runtime authority for the frozen statistical contract."""
    return {
        "interpretation": "fixed_stream_calibration_not_iid_coverage_guarantee",
        "per_gate_alpha": "0.000001",
        "wilson_z": "4.891638475698591",
        "hoeffding_formula": "sqrt(log(2/alpha)/(2*N))",
        "maximum_union_bound": "0.000016",
    }


def harness_source_digest(root: Path, base: Any) -> str:
    digest = hashlib.sha256()
    for name in HARNESS_BINDINGS:
        digest.update(name.encode("ascii") + b"\0")
        digest.update(base.read_limited(root / name, base.MAX_JSON_BYTES, "SOURCE"))
    return digest.hexdigest()


def _counter(value: Any, *, positive: bool = False) -> int:
    if type(value) is not str or DECIMAL.fullmatch(value) is None:
        raise QualificationError("STATISTICS")
    number = int(value)
    if number > (1 << 64) - 1 or (positive and number == 0):
        raise QualificationError("STATISTICS")
    return number


def _closed(value: Any, keys: set[str], code: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise QualificationError(code)
    return value


def expected_batch_plan(total: int, capacity: int) -> list[tuple[int, int, int, int]]:
    if type(total) is not int or type(capacity) is not int or total < 1 or capacity < 1:
        raise QualificationError("GEOMETRY")
    plan = []
    offset = 0
    ordinal = 0
    while offset < total:
        trials = min(total - offset, capacity * THREADS)
        blocks = (trials + THREADS - 1) // THREADS
        plan.append((ordinal, offset, trials, blocks))
        offset += trials
        ordinal += 1
    return plan


def select_statistical_case(bank: Any) -> dict[str, Any]:
    if (
        type(bank) is not dict
        or type(bank.get("statistical_vectors")) is not list
        or len(bank["statistical_vectors"]) != 1
    ):
        raise QualificationError("MANIFEST")
    row = bank["statistical_vectors"][0]
    if type(row) is not dict or hashlib.sha256(canonical(row)).hexdigest() != STATISTICAL_ROW_SHA256:
        raise QualificationError("MANIFEST")
    if (
        row.get("id") != STATISTICAL_ID
        or row.get("seed") != STATISTICAL_SEED
        or row.get("requested_trials") != str(TRIALS)
        or row.get("hero_card_ids") != [12, 37]
        or row.get("board_card_ids") != [0, 18, 33, 47, 48]
        or row.get("opponent_count") != 1
        or row.get("canonical_case_hash_hex") != STATISTICAL_HASH
    ):
        raise QualificationError("MANIFEST")
    return row


def _event(kernel: str, grid: tuple[int, ...], block: tuple[int, ...], **values: int) -> dict[str, Any]:
    return {"kernel": kernel, "grid": grid, "block": block, **values}


def validate_launch_trace(name: str, trace: Any) -> list[dict[str, Any]]:
    config = GEOMETRIES.get(name)
    if config is None or type(trace) is not list:
        raise QualificationError("GEOMETRY")
    plan = expected_batch_plan(TRIALS, config["capacity"])
    expected: list[dict[str, Any]] = []
    for _ordinal, offset, trials, blocks in plan:
        expected.extend([
            _event("simulate", (blocks,), (THREADS,), first_simulation_id=offset, trials=trials),
            _event("reduce", (1,), (THREADS,), partial_count=blocks),
        ])
    if trace != expected:
        raise QualificationError("GEOMETRY")
    batches = []
    for ordinal, offset, trials, blocks in plan:
        batches.append({
            "ordinal": str(ordinal), "first_simulation_id": str(offset),
            "trials": str(trials), "partial_blocks": str(blocks),
            "simulate_grid": [str(blocks)], "simulate_block": [str(THREADS)],
            "reduce_grid": ["1"], "reduce_block": [str(THREADS)],
        })
    return batches


def _wilson(successes: int, total: int) -> tuple[Decimal, Decimal]:
    getcontext().prec = 60
    n = Decimal(total)
    proportion = Decimal(successes) / n
    denominator = Decimal(1) + WILSON_Z * WILSON_Z / n
    center = (proportion + WILSON_Z * WILSON_Z / (2 * n)) / denominator
    radius = WILSON_Z * (
        proportion * (Decimal(1) - proportion) / n
        + WILSON_Z * WILSON_Z / (4 * n * n)
    ).sqrt() / denominator
    return center - radius, center + radius


def _unit_text(value: Decimal) -> str:
    text = format(value, "f").rstrip("0").rstrip(".")
    return text if text else "0"


def statistical_report(row: dict[str, Any], result: Any) -> dict[str, Any]:
    if type(result) is not dict or set(result) != AGGREGATE_KEYS:
        raise QualificationError("STATISTICS")
    total = _counter(result["completed_trials"], positive=True)
    if (
        total != TRIALS
        or type(result["tie_by_other_winners"]) is not dict
        or tuple(result["tie_by_other_winners"]) != TIE_KEYS
    ):
        raise QualificationError("STATISTICS")
    observed = {
        "unique_win": _counter(result["unique_wins"]),
        "tie": sum(_counter(result["tie_by_other_winners"][key]) for key in TIE_KEYS),
        "loss": _counter(result["losses"]),
    }
    wilson = {}
    for name in ("unique_win", "tie", "loss"):
        estimand = row["estimands"][name]
        numerator = _counter(estimand["numerator"])
        denominator = _counter(estimand["denominator"], positive=True)
        lower, upper = _wilson(observed[name], total)
        exact = Decimal(numerator) / Decimal(denominator)
        if not lower <= exact <= upper:
            raise QualificationError("STATISTICS")
        wilson[name] = {
            "successes": str(observed[name]), "trials": str(total),
            "lower": _unit_text(lower), "upper": _unit_text(upper),
            "population_numerator": str(numerator), "population_denominator": str(denominator),
            "status": "passed",
        }
    observed_units = _counter(result["equity_share_units"])
    bounded = row["bounded_mean_equity"]
    population_units = _counter(bounded["population_exact_units"])
    population_n = _counter(bounded["population_N"], positive=True)
    observed_equity = Decimal(observed_units) / (Decimal(420) * Decimal(total))
    exact_equity = Decimal(population_units) / (Decimal(420) * Decimal(population_n))
    radius = ((Decimal(2) / ALPHA).ln() / (Decimal(2) * Decimal(total))).sqrt()
    if abs(observed_equity - exact_equity) > radius:
        raise QualificationError("STATISTICS")
    return {
        "wilson": wilson,
        "hoeffding": {
            "observed_equity": _unit_text(observed_equity),
            "exact_equity": _unit_text(exact_equity), "radius": _unit_text(radius),
            "observed_units": str(observed_units), "trials": str(total),
            "population_exact_units": str(population_units), "population_N": str(population_n),
            "status": "passed",
        },
    }


_BASE_MODULE: Any = None


def _base() -> Any:
    """Load the stdlib-only Phase 5A operations module without importing CuPy."""
    global _BASE_MODULE
    if _BASE_MODULE is None:
        path = Path(__file__).with_name("qualify_gpu.py")
        spec = importlib.util.spec_from_file_location("_pkng_phase5a_operations", path)
        if spec is None or spec.loader is None:
            raise QualificationError("INTERNAL")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _BASE_MODULE = module
    return _BASE_MODULE


def aggregate_wire(result: Any) -> dict[str, Any]:
    try:
        return {
            "completed_trials": str(result.completed_trials),
            "unique_wins": str(result.unique_wins),
            "tie_by_other_winners": {str(i + 1): str(value) for i, value in enumerate(result.tie_by_other_winners)},
            "losses": str(result.losses),
            "equity_share_units": str(result.equity_share_units),
            "hero_category_counts": {
                name: str(value)
                for name, value in zip(CATEGORY_KEYS, result.hero_category_counts)
            },
            "rejection_count": str(result.rejection_count),
        }
    except Exception as exc:
        raise QualificationError("WORKER") from exc


def _worker(seed_bank_path: Path, wheel_path: Path) -> dict[str, Any]:
    if os.environ.get("PKNG_STATISTICAL_QUALIFICATION_WORKER") != "1":
        raise QualificationError("WORKER")
    base = _base()
    import cupy as cp  # type: ignore
    from poker_knight_ng import _cuda_runtime
    from poker_knight_ng.reference.monte_carlo import run_cpu_monte_carlo
    from poker_knight_ng.reference.rng import derive_philox_key

    distribution = importlib.metadata.distribution("poker-knight-ng")
    installation = base._verify_installed_wheel(distribution, wheel_path)
    bank = base.strict_json(base.read_limited(seed_bank_path, base.MAX_JSON_BYTES, "MANIFEST"))
    row = select_statistical_case(bank)
    case_hash = bytes.fromhex(row["canonical_case_hash_hex"])
    seed = int(row["seed"], 16)
    cpu_raw = run_cpu_monte_carlo(
        seed=seed,
        hero_card_ids=tuple(row["hero_card_ids"]),
        board_card_ids=tuple(row["board_card_ids"]),
        opponent_count=row["opponent_count"],
        requested_trials=int(row["requested_trials"]),
        replay_case_hash=case_hash,
    )
    cpu_aggregate = aggregate_wire(cpu_raw)
    frozen_aggregate = row["expected"]
    if cpu_aggregate != frozen_aggregate:
        raise QualificationError("VERIFY")
    _digest, key = derive_philox_key(seed, case_hash)

    device = int(cp.cuda.runtime.getDevice())
    properties = cp.cuda.runtime.getDeviceProperties(device)
    device_id = base._device_uuid(properties)
    major = properties.get("major", properties.get(b"major"))
    minor = properties.get("minor", properties.get(b"minor"))
    memory_before, _memory_total = cp.cuda.runtime.memGetInfo()
    geometries: dict[str, Any] = {}
    provenance: tuple[str, str] | None = None

    for name, config in GEOMETRIES.items():
        runtime = _cuda_runtime.CupyDeterministicRuntime(
            batch_blocks=config["batch_blocks"],
            vram_budget_bytes=config["vram_budget_bytes"],
        )
        actual_capacity = runtime._batch_capacity()
        if actual_capacity != config["capacity"]:
            raise QualificationError("GEOMETRY")
        trace: list[dict[str, Any]] = []
        original_kernels = runtime._kernels

        def observed_kernels() -> tuple[Any, Any]:
            simulate, reduce = original_kernels()

            def observed_simulate(grid: tuple[int, ...], block: tuple[int, ...], arguments: tuple[Any, ...]) -> Any:
                trace.append(_event(
                    "simulate", tuple(map(int, grid)), tuple(map(int, block)),
                    first_simulation_id=int(arguments[6]), trials=int(arguments[7]),
                ))
                return simulate(grid, block, arguments)

            def observed_reduce(grid: tuple[int, ...], block: tuple[int, ...], arguments: tuple[Any, ...]) -> Any:
                trace.append(_event(
                    "reduce", tuple(map(int, grid)), tuple(map(int, block)),
                    partial_count=int(arguments[1]),
                ))
                return reduce(grid, block, arguments)

            return observed_simulate, observed_reduce

        runtime._kernels = observed_kernels  # type: ignore[method-assign]
        started = time.monotonic_ns()
        cuda_raw = runtime.run(
            hero=tuple(row["hero_card_ids"]), board=tuple(row["board_card_ids"]),
            opponents=row["opponent_count"], key=key, first_simulation_id=0,
            count=int(row["requested_trials"]),
        )
        duration = time.monotonic_ns() - started
        if duration <= 0:
            raise QualificationError("WORKER")
        cuda_aggregate = aggregate_wire(cuda_raw)
        if cuda_aggregate != frozen_aggregate or cuda_aggregate != cpu_aggregate:
            raise QualificationError("VERIFY")
        batches = validate_launch_trace(name, trace)
        report = statistical_report(row, cuda_aggregate)
        current_provenance = runtime.provenance()
        if provenance is None:
            provenance = current_provenance
        elif current_provenance != provenance:
            raise QualificationError("SOURCE")
        geometries[name] = {
            "config": {
                "batch_blocks": str(config["batch_blocks"]),
                "vram_budget_bytes": (
                    "default"
                    if config["vram_budget_bytes"] is None
                    else str(config["vram_budget_bytes"])
                ),
            },
            "actual_capacity": str(actual_capacity), "batches": batches,
            "cpu_aggregate": cpu_aggregate, "cuda_aggregate": cuda_aggregate,
            "frozen_aggregate": frozen_aggregate, "statistics": report,
            "duration_ns": str(duration),
        }

    if provenance is None:
        raise QualificationError("WORKER")
    memory_after, _memory_total = cp.cuda.runtime.memGetInfo()
    return {
        "statistical_case": {
            "id": row["id"], "seed": row["seed"], "requested_trials": row["requested_trials"],
            "hero_card_ids": [str(value) for value in row["hero_card_ids"]],
            "board_card_ids": [str(value) for value in row["board_card_ids"]],
            "opponent_count": str(row["opponent_count"]),
            "canonical_case_hash_hex": row["canonical_case_hash_hex"],
        },
        "geometries": geometries,
        "provenance": {
            "qualification": "cuda-statistical-v1", "device_id": device_id,
            "kernel_id": provenance[1],
        },
        "environment": {
            "python_version": platform.python_version(), "cupy_version": cp.__version__,
            "cuda_driver_version": str(cp.cuda.runtime.driverGetVersion()),
            "cuda_runtime_version": str(cp.cuda.runtime.runtimeGetVersion()),
            "compute_capability": f"{int(major)}.{int(minor)}",
            "memory_free_before_bytes": str(int(memory_before)),
            "memory_free_after_bytes": str(int(memory_after)),
        },
        "installation": installation,
    }


def run_pre_admission(
    root: Path, target_sha: str, expected_branch: str, base: Any,
) -> tuple[Path, Path, dict[str, Any]]:
    """Complete all source/seed checks before the first GPU inventory call."""
    base.checkout_identity(root, target_sha, expected_branch)
    completed = base._run(
        [sys.executable, str(root / "tools/generate_rng_seed_bank.py"), "--verify"],
        cwd=root, code="VERIFY",
    )
    if completed.stdout or completed.stderr:
        raise QualificationError("VERIFY")
    manifest, seed_bank = base.verify_seed_manifest(root)
    admission = base.gpu_admission(root)
    return manifest, seed_bank, admission


def expected_batches_wire(geometry: str) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": str(ordinal), "first_simulation_id": str(offset),
            "trials": str(trials), "partial_blocks": str(blocks),
            "simulate_grid": [str(blocks)], "simulate_block": [str(THREADS)],
            "reduce_grid": ["1"], "reduce_block": [str(THREADS)],
        }
        for ordinal, (_plan_ordinal, offset, trials, blocks) in enumerate(
            expected_batch_plan(TRIALS, GEOMETRIES[geometry]["capacity"])
        )
    ]


def validate_worker_output(
    value: Any, row: dict[str, Any], admission: dict[str, Any],
    wheel_record: dict[str, str], cuda_source_sha256: str,
) -> dict[str, Any]:
    value = _closed(value, {"environment", "geometries", "installation", "provenance", "statistical_case"}, "WORKER")
    expected_case = {
        "id": row["id"], "seed": row["seed"], "requested_trials": row["requested_trials"],
        "hero_card_ids": [str(v) for v in row["hero_card_ids"]],
        "board_card_ids": [str(v) for v in row["board_card_ids"]],
        "opponent_count": str(row["opponent_count"]),
        "canonical_case_hash_hex": row["canonical_case_hash_hex"],
    }
    if value["statistical_case"] != expected_case:
        raise QualificationError("WORKER")
    installation = _closed(
        value["installation"],
        {"wheel_basename", "wheel_contents_verified", "wheel_sha256"},
        "ARTIFACT",
    )
    if (installation["wheel_contents_verified"] != "true"
            or installation["wheel_basename"] != wheel_record["basename"]
            or installation["wheel_sha256"] != wheel_record["sha256"]):
        raise QualificationError("ARTIFACT")
    provenance = _closed(value["provenance"], {"device_id", "kernel_id", "qualification"}, "SOURCE")
    normalized_uuid = "cuda-uuid:" + admission["nvidia_uuid"].removeprefix("GPU-").replace("-", "").lower()
    if (provenance["qualification"] != "cuda-statistical-v1"
            or provenance["device_id"] != normalized_uuid
            or provenance["kernel_id"] != "cuda-source-sha256:" + cuda_source_sha256):
        raise QualificationError("SOURCE")
    environment = _closed(value["environment"], {
        "compute_capability", "cuda_driver_version", "cuda_runtime_version", "cupy_version",
        "memory_free_after_bytes", "memory_free_before_bytes", "python_version",
    }, "ENVIRONMENT")
    if environment["compute_capability"] != admission["compute_capability"] or environment["cupy_version"] != "14.1.1":
        raise QualificationError("ENVIRONMENT")
    for key in ("cuda_driver_version", "cuda_runtime_version", "memory_free_after_bytes", "memory_free_before_bytes"):
        if type(environment[key]) is not str or DECIMAL.fullmatch(environment[key]) is None:
            raise QualificationError("ENVIRONMENT")
    geometries = _closed(value["geometries"], set(GEOMETRIES), "GEOMETRY")
    for name, config in GEOMETRIES.items():
        geometry = _closed(geometries[name], {
            "actual_capacity", "batches", "config", "cpu_aggregate", "cuda_aggregate",
            "duration_ns", "frozen_aggregate", "statistics",
        }, "GEOMETRY")
        expected_config = {
            "batch_blocks": str(config["batch_blocks"]),
            "vram_budget_bytes": (
                "default"
                if config["vram_budget_bytes"] is None
                else str(config["vram_budget_bytes"])
            ),
        }
        if (geometry["config"] != expected_config or geometry["actual_capacity"] != str(config["capacity"])
                or geometry["batches"] != expected_batches_wire(name)
                or type(geometry["duration_ns"]) is not str or DECIMAL.fullmatch(geometry["duration_ns"]) is None
                or int(geometry["duration_ns"]) <= 0):
            raise QualificationError("GEOMETRY")
        if not (
            geometry["cpu_aggregate"]
            == geometry["cuda_aggregate"]
            == geometry["frozen_aggregate"]
            == row["expected"]
        ):
            raise QualificationError("VERIFY")
        if geometry["statistics"] != statistical_report(row, geometry["cuda_aggregate"]):
            raise QualificationError("STATISTICS")
    return value


def _run_id(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value) is not None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def verify_checkout_identity(base: Any, root: Path, source: dict[str, Any]) -> None:
    """Bind offline evidence to the exact clean checkout named by the record."""
    base.checkout_identity(root, source["git_sha"], source["branch"])


def _failure(run_id: str, code: str) -> dict[str, Any]:
    return {"error_codes": [code], "format_version": "1", "run_id": run_id, "status": "failed"}


def _passed_shape(record: Any) -> dict[str, Any]:
    record = _closed(record, {
        "artifacts", "environment", "error_codes", "format_version", "gates", "geometries",
        "preregistration", "provenance", "run_id", "source", "statistical_case", "status",
    }, "VERIFY")
    if (
        record["format_version"] != "1"
        or record["status"] != "passed"
        or record["error_codes"] != []
        or not _run_id(record["run_id"])
    ):
        raise QualificationError("VERIFY")
    artifacts = _closed(record["artifacts"], {"sdist", "seed_bank", "seed_manifest", "wheel"}, "ARTIFACT")
    for artifact in artifacts.values():
        row = _closed(artifact, {"basename", "sha256", "size"}, "ARTIFACT")
        if (type(row["basename"]) is not str or "/" in row["basename"] or "\\" in row["basename"]
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
                or type(row["size"]) is not str or DECIMAL.fullmatch(row["size"]) is None):
            raise QualificationError("ARTIFACT")
    sdist_basename = artifacts["sdist"]["basename"]
    invalid_artifact_roles = (
        not artifacts["wheel"]["basename"].endswith(".whl")
        or not (sdist_basename.endswith(".tar.gz") or sdist_basename.endswith(".zip"))
        or artifacts["seed_bank"]["basename"] != "rng_seed_bank.json"
        or artifacts["seed_manifest"]["basename"] != "rng_seed_bank.sha256"
    )
    if invalid_artifact_roles:
        raise QualificationError("ARTIFACT")
    source = _closed(
        record["source"],
        {
            "branch",
            "clean",
            "cuda_source_sha256",
            "git_sha",
            "harness_source_sha256",
            "phase5c_tool_sha256",
            "seed_generator_sha256",
        },
        "SOURCE",
    )
    if (source["clean"] != "true" or source["branch"] != EXPECTED_BRANCH
            or re.fullmatch(r"[0-9a-f]{40}", source["git_sha"]) is None
            or any(re.fullmatch(r"[0-9a-f]{64}", source[key]) is None for key in (
                "cuda_source_sha256", "harness_source_sha256", "phase5c_tool_sha256", "seed_generator_sha256",
            ))):
        raise QualificationError("SOURCE")
    if record["gates"] != {
        "admission": "passed", "checkout": "passed", "exact_aggregate": "passed",
        "geometry": "passed", "seed_bank_verify": "passed", "statistics": "passed", "worker": "passed",
    }:
        raise QualificationError("VERIFY")
    if record["preregistration"] != preregistration_record():
        raise QualificationError("STATISTICS")
    environment = _closed(record["environment"], {
        "compute_capability", "cuda_driver_version", "cuda_runtime_version", "cupy_version",
        "memory_free_after_bytes", "memory_free_before_bytes", "python_version",
    }, "ENVIRONMENT")
    invalid_environment = (
        environment["cupy_version"] != "14.1.1"
        or re.fullmatch(r"[0-9]+\.[0-9]+", environment["compute_capability"]) is None
        or re.fullmatch(r"3\.13\.[0-9]+", environment["python_version"]) is None
        or any(
            type(environment[key]) is not str or DECIMAL.fullmatch(environment[key]) is None
            for key in (
                "cuda_driver_version",
                "cuda_runtime_version",
                "memory_free_after_bytes",
                "memory_free_before_bytes",
            )
        )
    )
    if invalid_environment:
        raise QualificationError("ENVIRONMENT")
    provenance = _closed(record["provenance"], {"device_id", "kernel_id", "qualification"}, "SOURCE")
    if (provenance["qualification"] != "cuda-statistical-v1"
            or re.fullmatch(r"cuda-uuid:[0-9a-f]{32}", provenance["device_id"]) is None
            or provenance["kernel_id"] != "cuda-source-sha256:" + source["cuda_source_sha256"]):
        raise QualificationError("SOURCE")
    statistical_case = _closed(record["statistical_case"], {
        "board_card_ids", "canonical_case_hash_hex", "hero_card_ids", "id", "opponent_count",
        "requested_trials", "seed",
    }, "VERIFY")
    if statistical_case != {
        "id": STATISTICAL_ID, "seed": STATISTICAL_SEED, "requested_trials": str(TRIALS),
        "hero_card_ids": ["12", "37"], "board_card_ids": ["0", "18", "33", "47", "48"],
        "opponent_count": "1", "canonical_case_hash_hex": STATISTICAL_HASH,
    }:
        raise QualificationError("VERIFY")
    _closed(record["geometries"], set(GEOMETRIES), "GEOMETRY")
    return record


def qualify(arguments: argparse.Namespace) -> int:
    if (os.environ.get("RUN_CUDA_STATISTICAL_QUALIFICATION") != "1"
            or not _run_id(arguments.run_id) or arguments.branch != EXPECTED_BRANCH):
        return 2
    root = _repo_root().resolve()
    base = _base()
    try:
        namespace = base.qualification_namespace(root, create=True)
    except Exception:
        return 2
    requested = Path(arguments.output_root).absolute() if arguments.output_root else namespace
    if requested != namespace:
        return 2
    inputs = {"wheel": Path(arguments.wheel), "sdist": Path(arguments.sdist)}
    try:
        if any(path.is_symlink() for path in inputs.values()):
            raise QualificationError("ARTIFACT")
        resolved = {name: path.resolve(strict=True) for name, path in inputs.items()}
        if not resolved["wheel"].name.endswith(".whl") or not (
            resolved["sdist"].name.endswith(".tar.gz") or resolved["sdist"].name.endswith(".zip")
        ):
            raise QualificationError("ARTIFACT")
        for path in resolved.values():
            base.regular(path, "ARTIFACT")
    except Exception:
        return 2
    run = namespace / arguments.run_id
    if run.exists() or run.is_symlink():
        return 2
    run.mkdir(mode=0o755)
    base._fsync_parent(namespace)
    destination = run / "qualification.json"
    evidence: dict[str, Any] = _failure(arguments.run_id, "INTERNAL")
    try:
        manifest, seed_bank, admission = run_pre_admission(root, arguments.target_sha, arguments.branch, base)
        files = run / "files"
        files.mkdir()
        sources = {
            "seed_manifest": manifest,
            "seed_bank": seed_bank,
            "wheel": resolved["wheel"],
            "sdist": resolved["sdist"],
        }
        if len({path.name for path in sources.values()}) != len(sources):
            raise QualificationError("ARTIFACT")
        copied: dict[str, Path] = {}
        for name, source in sources.items():
            copied[name] = files / source.name
            base.copy_artifact(source, copied[name])
        artifacts = {name: base.file_record(path) for name, path in copied.items()}
        cache = run / "cupy-cache"
        cache.mkdir()
        if any(cache.iterdir()):
            raise QualificationError("ENVIRONMENT")
        environment = base.child_environment(cache)
        environment["PKNG_STATISTICAL_QUALIFICATION_WORKER"] = "1"
        worker_argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--seed-bank",
            str(copied["seed_bank"]),
            "--wheel",
            str(copied["wheel"]),
        ]
        completed = base._run(
            worker_argv,
            cwd=root,
            code="WORKER",
            env=environment,
        )
        worker_bytes = completed.stdout.encode("utf-8")
        if completed.stderr or not worker_bytes.endswith(b"\n"):
            raise QualificationError("WORKER")
        worker = base.strict_json(worker_bytes)
        if canonical(worker) + b"\n" != worker_bytes:
            raise QualificationError("WORKER")
        cuda_source_sha256 = base.source_digest(root)
        seed_bytes = base.read_limited(
            copied["seed_bank"],
            base.MAX_JSON_BYTES,
            "MANIFEST",
        )
        row = select_statistical_case(base.strict_json(seed_bytes))
        worker = validate_worker_output(worker, row, admission, artifacts["wheel"], cuda_source_sha256)
        post = base.gpu_admission(root)
        if (
            post["nvidia_uuid"] != admission["nvidia_uuid"]
            or post["compute_capability"] != admission["compute_capability"]
        ):
            raise QualificationError("ADMISSION")
        evidence = {
            "format_version": "1",
            "run_id": arguments.run_id,
            "status": "passed",
            "error_codes": [],
            "source": {
                "git_sha": arguments.target_sha,
                "branch": arguments.branch,
                "clean": "true",
                "cuda_source_sha256": cuda_source_sha256,
                "harness_source_sha256": harness_source_digest(root, base),
                "phase5c_tool_sha256": base.file_record(Path(__file__).resolve())["sha256"],
                "seed_generator_sha256": base.file_record(root / "tools/generate_rng_seed_bank.py")["sha256"],
            },
            "artifacts": artifacts,
            "gates": {
                "checkout": "passed",
                "seed_bank_verify": "passed",
                "admission": "passed",
                "worker": "passed",
                "geometry": "passed",
                "exact_aggregate": "passed",
                "statistics": "passed",
            },
            "preregistration": preregistration_record(),
            "statistical_case": worker["statistical_case"],
            "geometries": worker["geometries"],
            "provenance": worker["provenance"],
            "environment": worker["environment"],
        }
        _passed_shape(evidence)
    except Exception as exc:
        code = getattr(exc, "code", "INTERNAL")
        if type(code) is not str or re.fullmatch(r"[A-Z_]{2,32}", code) is None:
            code = "INTERNAL"
        evidence = _failure(arguments.run_id, code)
    base.atomic_write(destination, canonical(evidence) + b"\n")
    return 0 if evidence["status"] == "passed" else 1


def verify(evidence_path: Path, root: Path) -> int:
    try:
        base = _base()
        root = root.resolve(strict=True)
        raw = base.read_limited(evidence_path, base.MAX_JSON_BYTES, "VERIFY")
        record = base.strict_json(raw)
        if canonical(record) + b"\n" != raw:
            raise QualificationError("VERIFY")
        if type(record) is not dict or record.get("status") != "passed":
            return 1
        record = _passed_shape(record)
        verify_checkout_identity(base, root, record["source"])
        if evidence_path.parent.name != record["run_id"]:
            raise QualificationError("VERIFY")
        files = evidence_path.parent / "files"
        for artifact in record["artifacts"].values():
            path = files / artifact["basename"]
            if path.parent != files or base.file_record(path) != artifact:
                raise QualificationError("ARTIFACT")
        tool_hash = base.file_record(root / "tools/qualify_gpu_statistics.py")["sha256"]
        generator_hash = base.file_record(root / "tools/generate_rng_seed_bank.py")["sha256"]
        harness_hash = harness_source_digest(root, base)
        cuda_hash = base.source_digest(root)
        source = record["source"]
        if (
            tool_hash != source["phase5c_tool_sha256"]
            or generator_hash != source["seed_generator_sha256"]
            or harness_hash != source["harness_source_sha256"]
            or cuda_hash != source["cuda_source_sha256"]
        ):
            raise QualificationError("SOURCE")
        bank_path = files / record["artifacts"]["seed_bank"]["basename"]
        bank_bytes = base.read_limited(bank_path, base.MAX_JSON_BYTES, "MANIFEST")
        row = select_statistical_case(base.strict_json(bank_bytes))
        expected_case = {
            "id": row["id"],
            "seed": row["seed"],
            "requested_trials": row["requested_trials"],
            "hero_card_ids": [str(value) for value in row["hero_card_ids"]],
            "board_card_ids": [str(value) for value in row["board_card_ids"]],
            "opponent_count": str(row["opponent_count"]),
            "canonical_case_hash_hex": row["canonical_case_hash_hex"],
        }
        if record["statistical_case"] != expected_case:
            raise QualificationError("VERIFY")
        for name, config in GEOMETRIES.items():
            geometry = _closed(record["geometries"][name], {
                "actual_capacity", "batches", "config", "cpu_aggregate", "cuda_aggregate",
                "duration_ns", "frozen_aggregate", "statistics",
            }, "GEOMETRY")
            expected_config = {
                "batch_blocks": str(config["batch_blocks"]),
                "vram_budget_bytes": (
                    "default"
                    if config["vram_budget_bytes"] is None
                    else str(config["vram_budget_bytes"])
                ),
            }
            aggregate_equal = (
                geometry["cpu_aggregate"]
                == geometry["cuda_aggregate"]
                == geometry["frozen_aggregate"]
                == row["expected"]
            )
            duration = geometry["duration_ns"]
            if (
                geometry["config"] != expected_config
                or geometry["batches"] != expected_batches_wire(name)
                or geometry["actual_capacity"] != str(config["capacity"])
                or type(duration) is not str
                or DECIMAL.fullmatch(duration) is None
                or int(duration) <= 0
                or not aggregate_equal
                or geometry["statistics"] != statistical_report(row, geometry["cuda_aggregate"])
            ):
                raise QualificationError("VERIFY")
        return 0
    except Exception:
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--seed-bank", required=True)
    worker.add_argument("--wheel", required=True)
    run = commands.add_parser("qualify")
    run.add_argument("--run-id", required=True)
    run.add_argument("--target-sha", required=True)
    run.add_argument("--branch", required=True)
    run.add_argument("--wheel", required=True)
    run.add_argument("--sdist", required=True)
    run.add_argument("--output-root")
    check = commands.add_parser("verify")
    check.add_argument("evidence")
    check.add_argument("--root", default=str(_repo_root()))
    arguments = parser.parse_args(argv)
    if arguments.command == "worker":
        result = _worker(Path(arguments.seed_bank), Path(arguments.wheel))
        sys.stdout.buffer.write(canonical(result) + b"\n")
        return 0
    if arguments.command == "qualify":
        return qualify(arguments)
    return verify(Path(arguments.evidence), Path(arguments.root))


if __name__ == "__main__":
    raise SystemExit(main())
