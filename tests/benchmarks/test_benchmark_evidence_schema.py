"""Private Phase 6C benchmark-evidence schema contract."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).parents[2]
SCHEMA = ROOT / "validation/holdem/v1/cuda_benchmark_private.schema.json"
HEX64 = "a" * 64


def _file(name: str) -> dict[str, str]:
    return {"basename": name, "sha256": HEX64, "size_bytes": "1"}


def _stages() -> dict[str, str]:
    return {"h2d_ns": "1", "simulate_ns": "1", "reduction_ns": "1", "d2h_ns": "1"}


def _sample() -> dict[str, object]:
    return {
        "analytical_sha256": HEX64,
        "duration_ns": "1000000",
        "throughput_per_second": "10000000.000",
    }


def _worker(mode: str, payload: dict[str, object]) -> dict[str, object]:
    modes = ("inventory", "expected", "cold", "warm", "steady", "stage")
    if payload.get("mode") != mode:
        raise AssertionError("fixture worker mode mismatch")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    cache_classes = {
        "inventory": "inventory",
        "expected": "expected-isolated",
        "cold": "startup-cold",
        "warm": "startup-warm",
        "steady": "steady-isolated",
        "stage": "stage-isolated",
    }
    return {
        "cache_class": cache_classes[mode],
        "mode": mode,
        "ordinal": str(modes.index(mode)),
        "fresh_process": "true",
        "exit_code": "0",
        "stdout_bytes": str(len(encoded)),
        "stderr_bytes": "0",
        "stdout_sha256": hashlib.sha256(encoded).hexdigest(),
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "output_limit_bytes": "2097152",
        "payload": payload,
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _admission(gpu_uuid: str) -> dict[str, object]:
    gpu = f"{gpu_uuid}, 2048, 4096\n".encode("ascii")
    compute = b""
    return {
        "gpu_uuid": gpu_uuid,
        "free_bytes": "2147483648",
        "total_bytes": "4294967296",
        "compute_applications": [],
        "gpu_snapshot_hex": gpu.hex(),
        "compute_snapshot_hex": compute.hex(),
        "gpu_snapshot_sha256": hashlib.sha256(gpu).hexdigest(),
        "compute_snapshot_sha256": hashlib.sha256(compute).hexdigest(),
    }


def _cell(street: str, opponents: str, trials: str) -> dict[str, object]:
    samples = [str(1_000_000 + index) for index in range(30)]
    throughput = format(
        (Decimal(trials) * Decimal(1_000_000_000) / Decimal(samples[14])).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        ),
        "f",
    )
    return {
        "cell_id": f"v1-{street}-o{opponents}-n{trials}",
        "street": street,
        "opponent_count": opponents,
        "requested_trials": trials,
        "backend": "cuda",
        "seed": "0x0123456789abcdef",
        "expected_analytical_sha256": HEX64,
        "steady": {
            "warmup_duration_ns": "1000000",
            "warmup_analytical_sha256": HEX64,
            "durations_ns": samples,
            "analytical_sha256s": [HEX64] * 30,
            "aggregate": {
                "count": "30",
                "minimum_ns": samples[0],
                "p5_ns": samples[1],
                "p50_ns": samples[14],
                "p95_ns": samples[28],
                "maximum_ns": samples[29],
                "throughput_per_second": throughput,
            },
        },
        "stage": {
            "durations": _stages(),
            "analytical_sha256": HEX64,
        },
    }


def _record() -> dict[str, object]:
    cells = [
        _cell(street, opponents, trials)
        for street in ("preflop", "flop", "turn", "river")
        for opponents in ("1", "3", "6")
        for trials in ("10000", "100000", "500000", "1000000")
    ]
    environment = {
        "os": "Linux",
        "kernel": "6.12.0",
        "python_version": "3.13.0",
        "cuda_driver_version": "13000",
        "cuda_runtime_version": "13020",
        "cupy_version": "14.1.1",
        "gpu_name": "Example GPU",
        "gpu_uuid": "GPU-0123456789abcdef0123456789abcdef",
        "compute_capability": "12.0",
        "device_memory_bytes": "4294967296",
    }
    closure = [{
        "basename": "poker_knight_ng/__init__.py",
        "sha256": hashlib.sha256(b"").hexdigest(),
        "size_bytes": "0",
    }]
    cold = {"mode": "cold", "duration_ns": "1000000", "analytical_sha256": HEX64}
    warm = {"mode": "warm", "duration_ns": "1000000", "analytical_sha256": HEX64}
    payloads = {
        "inventory": {
            "mode": "inventory",
            "installation": {
                "wheel_basename": "poker_knight_ng-0.1.0-py3-none-any.whl",
                "wheel_contents_verified": "true",
                "wheel_sha256": HEX64,
            },
            "environment": environment,
        },
        "expected": {
            "mode": "expected",
            "analytical_sha256s": {
                cell["cell_id"]: cell["expected_analytical_sha256"] for cell in cells
            },
        },
        "cold": cold,
        "warm": warm,
        "steady": {
            "mode": "steady",
            "cells": {
                cell["cell_id"]: {
                    key: value
                    for key, value in cast(dict[str, object], cell["steady"]).items()
                    if key != "aggregate"
                }
                for cell in cells
            },
        },
        "stage": {
            "mode": "stage",
            "planned_batch_blocks": "256",
            "batch_counts": {
                cell["cell_id"]: str(
                    (int(str(cell["cell_id"]).rsplit("-n", 1)[1]) + 256 * 128 - 1)
                    // (256 * 128)
                )
                for cell in cells
            },
            "cells": {cell["cell_id"]: cell["stage"] for cell in cells},
        },
    }
    return {
        "format_version": "phase6c-private-evidence-v1",
        "benchmark_id": "holdem-v1-cuda-baseline-1",
        "source": {
            "git_sha": "0" * 40,
            "branch": "revival/phase-6c-performance",
            "clean": "true",
            "benchmark_tool_sha256": HEX64,
            "cuda_runtime_source_sha256": HEX64,
        },
        "artifacts": {
            "wheel": _file("poker_knight_ng-0.1.0-py3-none-any.whl"),
            "sdist": _file("poker_knight_ng-0.1.0.tar.gz"),
            "lock": _file("uv.lock"),
            "scenario_manifest": _file("scenarios-v1.json"),
            "phase5b_qualification": _file("cuda_release_qualification.json"),
            "phase5b_manifest": _file("cuda_release_qualification.sha256"),
            "phase5c_qualification": _file("cuda_statistical_release_qualification.json"),
            "phase5c_manifest": _file("cuda_statistical_release_qualification.sha256"),
            "installed_wheel_byte_closure": closure,
        },
        "environment": environment,
        "admission": {
            "before": _admission(environment["gpu_uuid"]),
            "after": _admission(environment["gpu_uuid"]),
        },
        "workers": [
            _worker(mode, payloads[mode])
            for mode in ("inventory", "expected", "cold", "warm", "steady", "stage")
        ],
        "startup_cache": {
            "cold_worker_ordinal": "2",
            "warm_worker_ordinal": "3",
            "relationship": "shared_run_owned_0700_prepared_empty_sealed_after_cold_verified_immediately_before_warm",
            "cold_seal": {
                "cold_result_sha256": HEX64,
                "files": [{"path": "kernel.bin", "sha256": HEX64, "size_bytes": "1"}],
                "format_version": "phase6c-cache-seal-v1",
            },
            "warm_verified_seal": {
                "cold_result_sha256": HEX64,
                "files": [{"path": "kernel.bin", "sha256": HEX64, "size_bytes": "1"}],
                "format_version": "phase6c-cache-seal-v1",
            },
        },
        "startup": {
            "canary_cell_id": "v1-preflop-o1-n10000",
            "cold": {**_sample(), "analytical_sha256": cold["analytical_sha256"], "duration_ns": cold["duration_ns"]},
            "warm": {**_sample(), "analytical_sha256": warm["analytical_sha256"], "duration_ns": warm["duration_ns"]},
        },
        "matrix": cells,
        "gates": {
            "checkout": "passed", "artifacts": "passed", "environment": "passed",
            "admission": "passed", "worker": "passed", "matrix": "passed",
            "statistics": "passed",
        },
    }


def _semantic_matrix_errors(record: dict[str, object]) -> list[str]:
    """Test-local checks JSON Schema cannot express: closed ID set and uniqueness."""
    matrix = record["matrix"]
    assert isinstance(matrix, list)
    expected = {
        f"v1-{street}-o{opponents}-n{trials}"
        for street in ("preflop", "flop", "turn", "river")
        for opponents in ("1", "3", "6")
        for trials in ("10000", "100000", "500000", "1000000")
    }
    ids = [cell["cell_id"] for cell in matrix if isinstance(cell, dict)]
    errors = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate cell_id")
    if set(ids) != expected:
        errors.append("matrix IDs are not the canonical closed 48-cell set")
    return errors


def test_private_evidence_schema_accepts_the_preregistered_closed_record() -> None:
    schema = json.loads(SCHEMA.read_text("ascii"))
    Draft202012Validator.check_schema(schema)
    record = _record()
    Draft202012Validator(schema).validate(record)
    assert _semantic_matrix_errors(record) == []


def test_private_evidence_schema_accepts_a_resolved_zero_stage_duration() -> None:
    schema = json.loads(SCHEMA.read_text("ascii"))
    record: Any = _record()
    record["matrix"][0]["stage"]["durations"]["h2d_ns"] = "0"
    Draft202012Validator(schema).validate(record)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("unexpected", "x"),
        lambda value: value["matrix"][0]["steady"]["durations_ns"].__setitem__(0, 1000000),
        lambda value: value["matrix"][0]["steady"]["durations_ns"].pop(),
        lambda value: value["matrix"][0]["steady"]["durations_ns"].append("1000030"),
        lambda value: value["artifacts"]["wheel"].__setitem__("sha256", "not-a-hash"),
    ],
)
def test_private_evidence_schema_rejects_hostile_structural_mutations(mutation) -> None:
    validator = Draft202012Validator(json.loads(SCHEMA.read_text("ascii")))
    candidate = _record()
    mutation(candidate)
    with pytest.raises(ValidationError):
        validator.validate(candidate)


def test_test_local_semantic_helper_rejects_missing_and_duplicate_matrix_ids() -> None:
    missing = _record()
    missing["matrix"].pop()
    assert _semantic_matrix_errors(missing) == ["matrix IDs are not the canonical closed 48-cell set"]

    duplicate = _record()
    duplicate["matrix"][-1]["cell_id"] = duplicate["matrix"][0]["cell_id"]
    assert _semantic_matrix_errors(duplicate) == [
        "duplicate cell_id", "matrix IDs are not the canonical closed 48-cell set"
    ]
