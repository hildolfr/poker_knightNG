#!/usr/bin/env python3.13
"""Verify the privacy-safe Phase 5C CUDA statistical qualification record."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RECORD_RELATIVE = Path("validation/holdem/v1/cuda_statistical_release_qualification.json")
MANIFEST_RELATIVE = Path("validation/holdem/v1/manifests/cuda_statistical_release_qualification.sha256")
SOURCE_SHA = "7fb617b900c06102caafe240ff95afe7fef2aa58"
SOURCE_BRANCH = "revival/phase-6c-optimize-evaluator"
PUBLISHED_BRANCH = "revival/phase-6c-optimize-evaluator"
EVIDENCE_SHA256 = "27bf23106e44e399fb5157d1c61891852f9d9ce0f32d2504dc6f490271a24017"
HARNESS_SHA256 = "b2429bea1ce9f3721b0fc4f43294b046eaf530114a5c1b69ab438ef31812a85c"
CUDA_SHA256 = "8da8349bed65e782a18d29f83de884341b3838f40c1e83904d07860c2c4ade5a"
QUALIFICATION_ID = "cuda-statistical-v1"
RUN_ID = "phase6c-opt-7fb617b-5c1"
AGGREGATE_SHA256 = "aaed2605d43a639af1f8345dbcd768bd4366a75b50b7f5f30ca4eaeee5494b7f"
SOURCE_BINDINGS = (
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
ARTIFACTS = {
    "sdist": {
        "basename": "poker_knight_ng-0.1.0.tar.gz",
        "sha256": "429f2dab5e7c3dc3b9f1fa124b1ab61c4fae8e76457f3f519c55ae636784a625",
        "size": "68995",
    },
    "seed_bank": {
        "basename": "rng_seed_bank.json",
        "sha256": "a3cab71f5f7564381917826f75f46c9ab572369adb2fbb55ca13c3c286b2af8e",
        "size": "3520",
    },
    "seed_manifest": {
        "basename": "rng_seed_bank.sha256",
        "sha256": "f558526017cd52e3453c48f41b6d93225177cecceb0e047cc6824a56c7c985ec",
        "size": "1381",
    },
    "wheel": {
        "basename": "poker_knight_ng-0.1.0-py3-none-any.whl",
        "sha256": "e742fb044d6b1065362f49c15d565e3dbc908156a69b3649a9d2625a608492bb",
        "size": "77162",
    },
}
GEOMETRIES = {
    "budget_capacity_3": {
        "actual_capacity": "3", "batch_count": "6",
        "batch_plan_sha256": "f94cc1a040e6ad0f922c6aa0ca8c4a3a92d72795d7bbae9fa038119c1526d03e",
    },
    "capacity_1": {
        "actual_capacity": "1", "batch_count": "16",
        "batch_plan_sha256": "c2206a5ee6826bc09833714ec99c51a39adb65abecc0b58fe157a52e4d49951a",
    },
    "capacity_256": {
        "actual_capacity": "256", "batch_count": "1",
        "batch_plan_sha256": "184d6bc16122deac4394fa32689773adc2c547b3d5899cb11975d7bef5b015d8",
    },
    "capacity_7": {
        "actual_capacity": "7", "batch_count": "3",
        "batch_plan_sha256": "78a6c4cd548a9be9f5a3e7a08100390acfc4fdf20bdf05c01ada857eda13c32f",
    },
}
MANIFEST_PATHS = tuple(sorted((
    "README.md",
    "pyproject.toml",
    "src/poker_knight_ng/README.md",
    "src/poker_knight_ng/_cuda_runtime.py",
    "src/poker_knight_ng/cuda-sources/cards.cuh",
    "src/poker_knight_ng/cuda-sources/dealer.cuh",
    "src/poker_knight_ng/cuda-sources/deterministic_kernels.cu",
    "src/poker_knight_ng/cuda-sources/evaluator.cuh",
    "src/poker_knight_ng/cuda-sources/philox.cuh",
    "src/poker_knight_ng/cuda-sources/reduce.cuh",
    "src/poker_knight_ng/cuda-sources/simulate.cuh",
    "tests/cuda/test_cuda_statistical_release_qualification.py",
    "tests/cuda/test_gpu_statistical_qualification.py",
    "tests/cuda/test_gpu_statistical_qualification_contract.py",
    "tests/reference/test_rng_seed_bank_verifier.py",
    "tools/generate_rng_seed_bank.py",
    "tools/qualify_gpu.py",
    "tools/qualify_gpu_statistics.py",
    "tools/verify_cuda_release_qualification.py",
    "tools/verify_cuda_statistical_release_qualification.py",
    "uv.lock",
    "validation/holdem/v1/QUALIFICATION.md",
    "validation/holdem/v1/SPEC.md",
    "validation/holdem/v1/STATISTICAL_QUALIFICATION.md",
    "validation/holdem/v1/cuda_statistical_qualification.schema.json",
    "validation/holdem/v1/cuda_statistical_release_qualification.json",
    "validation/holdem/v1/manifests/rng_seed_bank.sha256",
    "validation/holdem/v1/rng_seed_bank.json",
)))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = _load("_phase5b_public_verifier", ROOT / "tools/verify_cuda_release_qualification.py")
_harness = _load("_phase5c_harness", ROOT / "tools/qualify_gpu_statistics.py")
VerificationError = _base.VerificationError
strict_json = _base.strict_json
canonical = _base.canonical
read_regular = _base.read_regular
sha256 = _base.sha256
closed = _base.closed
decimal = _base.decimal
digest = _base.digest


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


def _source_blobs(root: Path) -> dict[str, bytes]:
    _base._git(root, ["merge-base", "--is-ancestor", SOURCE_SHA, "HEAD"], capture=False)
    return {path: _base._git(root, ["show", f"{SOURCE_SHA}:{path}"]) for path in SOURCE_BINDINGS}


def _harness_digest(blobs: dict[str, bytes]) -> str:
    state = hashlib.sha256()
    for path in SOURCE_BINDINGS:
        state.update(path.encode("ascii"))
        state.update(b"\0")
        state.update(blobs[path])
    return state.hexdigest()


def _verify_manifest(root: Path) -> None:
    raw = read_regular(root / MANIFEST_RELATIVE)
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as exc:
        raise VerificationError("invalid manifest") from exc
    rows: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_./-]+)", line)
        if match is None or match.group(2) in rows:
            raise VerificationError("invalid manifest")
        rows[match.group(2)] = match.group(1)
    if tuple(rows) != MANIFEST_PATHS:
        raise VerificationError("manifest closure mismatch")
    if tuple(rows) != tuple(sorted(rows)):
        raise VerificationError("manifest is not sorted")
    if MANIFEST_RELATIVE.as_posix() in rows:
        raise VerificationError("manifest self-reference")
    for path, expected in rows.items():
        if sha256(root / path) != expected:
            raise VerificationError("manifest hash mismatch")


def _verify_record(record: dict[str, Any], root: Path) -> None:
    record = closed(record, {
        "device", "environment", "evidence", "format_version", "geometries",
        "preregistration", "qualification_id", "source", "statistical_case", "status",
    })
    if (
        record["format_version"] != "1"
        or record["qualification_id"] != QUALIFICATION_ID
        or record["status"] != "passed"
    ):
        raise VerificationError("record identity mismatch")

    source = closed(record["source"], {
        "bindings", "cuda_source_sha256", "executed_branch", "git_sha",
        "harness_source_sha256", "published_branch",
    })
    if source != {
        **source,
        "git_sha": SOURCE_SHA,
        "executed_branch": SOURCE_BRANCH,
        "published_branch": PUBLISHED_BRANCH,
        "harness_source_sha256": HARNESS_SHA256,
        "cuda_source_sha256": CUDA_SHA256,
    }:
        raise VerificationError("source identity mismatch")
    bindings = closed(source["bindings"], set(SOURCE_BINDINGS))
    blobs = _source_blobs(root)
    for path in SOURCE_BINDINGS:
        if digest(bindings[path]) != hashlib.sha256(blobs[path]).hexdigest():
            raise VerificationError("historical source binding mismatch")
    for path in (
        "tools/qualify_gpu.py",
        "tools/qualify_gpu_statistics.py",
        "validation/holdem/v1/rng_seed_bank.json",
    ):
        if sha256(root / path) != bindings[path]:
            raise VerificationError("verification authority drift")
    if _harness_digest(blobs) != HARNESS_SHA256:
        raise VerificationError("harness closure mismatch")
    if _base._cuda_digest(blobs) != CUDA_SHA256:
        raise VerificationError("CUDA closure mismatch")

    evidence = closed(record["evidence"], {"artifacts", "qualification_sha256", "run_id"})
    if evidence["run_id"] != RUN_ID or digest(evidence["qualification_sha256"]) != EVIDENCE_SHA256:
        raise VerificationError("evidence identity mismatch")
    artifacts = closed(evidence["artifacts"], set(ARTIFACTS))
    if artifacts != ARTIFACTS:
        raise VerificationError("artifact projection mismatch")
    if artifacts["seed_bank"]["sha256"] != bindings["validation/holdem/v1/rng_seed_bank.json"]:
        raise VerificationError("seed-bank binding mismatch")
    if artifacts["seed_manifest"]["sha256"] != bindings["validation/holdem/v1/manifests/rng_seed_bank.sha256"]:
        raise VerificationError("seed-manifest binding mismatch")

    statistical_case = closed(record["statistical_case"], {
        "board_card_ids", "canonical_case_hash_hex", "hero_card_ids", "id",
        "opponent_count", "requested_trials", "seed",
    })
    expected_case = {
        "id": "river-high-card-wtl-2000", "seed": "0x0000000000000001",
        "requested_trials": "2000", "hero_card_ids": ["12", "37"],
        "board_card_ids": ["0", "18", "33", "47", "48"], "opponent_count": "1",
        "canonical_case_hash_hex": "7e088bbb705b1436c35c7ffbd6577bcac4656cbb8b60fd582d32ca0ce8163037",
    }
    if statistical_case != expected_case:
        raise VerificationError("case identity mismatch")
    preregistration = closed(record["preregistration"], set(_harness.preregistration_record()))
    if preregistration != _harness.preregistration_record():
        raise VerificationError("preregistration mismatch")

    device = closed(record["device"], {"compute_capability", "device_id"})
    if device != {"compute_capability": "12.0", "device_id": "cuda-uuid:7caf8277fe9ff935229b57b2a0d3ff5f"}:
        raise VerificationError("device projection mismatch")
    environment = closed(record["environment"], {
        "cuda_driver_version", "cuda_runtime_version", "cupy_version", "python_version",
    })
    if environment != {
        "cuda_driver_version": "13030", "cuda_runtime_version": "13020",
        "cupy_version": "14.1.1", "python_version": "3.13.15",
    }:
        raise VerificationError("environment projection mismatch")

    bank = _harness._base().strict_json(
        blobs["validation/holdem/v1/rng_seed_bank.json"]
    )
    row = _harness.select_statistical_case(bank)
    aggregate_sha256 = hashlib.sha256(_harness.canonical(row["expected"])).hexdigest()
    if aggregate_sha256 != AGGREGATE_SHA256:
        raise VerificationError("aggregate authority mismatch")
    expected_statistics = _harness.statistical_report(row, row["expected"])
    geometries = closed(record["geometries"], set(GEOMETRIES))
    for name, expected in GEOMETRIES.items():
        geometry = closed(geometries[name], {
            "actual_capacity", "aggregate_sha256", "batch_count", "batch_plan_sha256", "statistics",
        })
        expected_plan = _harness.expected_batches_wire(name)
        plan_sha256 = hashlib.sha256(_harness.canonical(expected_plan)).hexdigest()
        if (
            geometry["actual_capacity"] != expected["actual_capacity"]
            or decimal(geometry["batch_count"]) != len(expected_plan)
        ):
            raise VerificationError("geometry projection mismatch")
        if (
            digest(geometry["batch_plan_sha256"]) != expected["batch_plan_sha256"]
            or plan_sha256 != expected["batch_plan_sha256"]
        ):
            raise VerificationError("batch plan mismatch")
        if digest(geometry["aggregate_sha256"]) != AGGREGATE_SHA256:
            raise VerificationError("aggregate projection mismatch")
        if geometry["statistics"] != expected_statistics:
            raise VerificationError("statistics projection mismatch")

    forbidden = (
        "/home/", "/tmp/", "desktop-drizzt", "NVIDIA GeForce", "RTX 5070",
        "compute_apps", "process_name", "memory_free", "duration_ns", "pid",
    )
    if any(any(token in value for token in forbidden) for value in _all_strings(record)):
        raise VerificationError("private runtime inventory leaked")


def verify(record_path: Path, root: Path) -> int:
    try:
        root = root.resolve(strict=True)
        record_path = record_path.resolve(strict=True)
        if record_path != root / RECORD_RELATIVE:
            raise VerificationError("record path mismatch")
        _verify_manifest(root)
        raw = read_regular(record_path)
        record = strict_json(raw)
        if raw != canonical(record):
            raise VerificationError("noncanonical record")
        _verify_record(record, root)
        return 0
    except (OSError, VerificationError, ValueError):
        return 1


def main() -> int:
    return verify(ROOT / RECORD_RELATIVE, ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
