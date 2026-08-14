#!/usr/bin/env python3
"""CPU-only verifier for private Phase 6C benchmark evidence."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import hashlib
import importlib.util
import json
from math import ceil
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any
import zipfile


MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
STAGE_BATCH_BLOCKS = 256
STAGE_THREADS = 128
QUALIFICATION_FILES = {
    "phase5b_qualification": "validation/holdem/v1/cuda_release_qualification.json",
    "phase5b_manifest": "validation/holdem/v1/manifests/cuda_release_qualification.sha256",
    "phase5c_qualification": "validation/holdem/v1/cuda_statistical_release_qualification.json",
    "phase5c_manifest": "validation/holdem/v1/manifests/cuda_statistical_release_qualification.sha256",
}
IMMUTABLE_QUALIFICATION_SHA256 = {
    "tools/verify_cuda_release_qualification.py": "5cd551c91a3a375d918898bde59c2c89e1ad38a074fbe08c8ee39afcb42e57b6",
    "tools/verify_cuda_statistical_release_qualification.py": "17308df0575d5b77524b2e84ebf21bf2c4a008c11a8562de8ab257422247e36d",
    "validation/holdem/v1/cuda_release_qualification.json": "4fd724d4ae4b3e81db8f89279d8344adf78d36ad4975c3963b1a1884f37fb3f4",
    "validation/holdem/v1/manifests/cuda_release_qualification.sha256": "9d9da3e526a1dc6734d46a1ea3b8f8a54f2838e775187271efba9f690ef15a1d",
    "validation/holdem/v1/cuda_statistical_release_qualification.json": "be01bcc0cdb91e887d77e16dbbc2c663e55595847efb94c2e51abecc91d0213b",
    "validation/holdem/v1/manifests/cuda_statistical_release_qualification.sha256": "978a1226db823ca63cf35f904cebce3724811c0677ee6acbdc343e2ea0caf57e",
    "src/poker_knight_ng/_cuda_runtime.py": "efbf7ebf0a0069beb62904cad56ce7ed363be79fc724bc9e6e8f7fae04e3273e",
}
SCHEMA_PATH = Path(__file__).parents[1] / "validation/holdem/v1/cuda_benchmark_private.schema.json"
_ALLOWED_KEYWORDS = {
    "$defs", "$id", "$ref", "$schema", "additionalProperties", "const", "enum",
    "items", "maxItems", "maxLength", "minItems", "pattern", "properties",
    "required", "title", "type",
}


class VerificationError(ValueError):
    """Closed private-evidence verification failure."""


class VerificationContext:
    """Explicit local authority for evidence identity binding."""

    def __init__(
        self, *, repo_root: Path, wheel: Path, sdist: Path, lockfile: Path,
        scenario_directory: Path, scenario_manifest: Path,
    ) -> None:
        self.repo_root = repo_root
        self.wheel = wheel
        self.sdist = sdist
        self.lockfile = lockfile
        self.scenario_directory = scenario_directory
        self.scenario_manifest = scenario_manifest


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise VerificationError("DUPLICATE_JSON_KEY")
        value[key] = item
    return value


def _canonical(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise VerificationError("CANONICAL_JSON") from exc


def _has_symlink_component(path: Path) -> bool:
    if not path.is_absolute():
        return True
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _load_canonical(path: Path, *, maximum: int) -> Any:
    if _has_symlink_component(path.absolute()) or not path.is_file():
        raise VerificationError("PATH")
    data = path.read_bytes()
    if not data or len(data) > maximum or not data.endswith(b"\n") or b"\r" in data:
        raise VerificationError("FRAMING")
    try:
        value = json.loads(
            data.decode("ascii"),
            object_pairs_hook=_pairs,
            parse_int=lambda _value: (_ for _ in ()).throw(VerificationError("JSON_NUMBER")),
            parse_float=lambda _value: (_ for _ in ()).throw(VerificationError("JSON_NUMBER")),
            parse_constant=lambda _value: (_ for _ in ()).throw(VerificationError("JSON_CONSTANT")),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError("JSON") from exc
    if _canonical(value) != data:
        raise VerificationError("NONCANONICAL")
    return value


def _load_schema(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise VerificationError("SCHEMA_PATH")
    data = path.read_bytes()
    if not data or len(data) > 256 * 1024:
        raise VerificationError("SCHEMA_FRAMING")
    try:
        return json.loads(
            data.decode("ascii"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(VerificationError("SCHEMA_NUMBER")),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError("SCHEMA_JSON") from exc


def _check_schema_keywords(node: Any) -> None:
    if isinstance(node, dict):
        unknown = set(node) - _ALLOWED_KEYWORDS
        if unknown:
            raise VerificationError("SCHEMA_KEYWORD")
        for key, value in node.items():
            if key in {"properties", "$defs"}:
                if type(value) is not dict:
                    raise VerificationError("SCHEMA_DEFINITION")
                for child in value.values():
                    _check_schema_keywords(child)
            elif key == "items":
                _check_schema_keywords(value)
    elif not isinstance(node, (str, bool, int, list)) and node is not None:
        raise VerificationError("SCHEMA_DEFINITION")


def _schema_type(value: Any, expected: str) -> bool:
    return {
        "array": type(value) is list,
        "object": type(value) is dict,
        "string": type(value) is str,
    }.get(expected, False)


def _validate_schema(value: Any, schema: dict[str, Any], root: dict[str, Any], path: str = "$") -> None:
    reference = schema.get("$ref")
    if reference is not None:
        if type(reference) is not str or not reference.startswith("#/$defs/"):
            raise VerificationError("SCHEMA_REFERENCE")
        name = reference.removeprefix("#/$defs/")
        definitions = root.get("$defs")
        if type(definitions) is not dict or name not in definitions:
            raise VerificationError("SCHEMA_REFERENCE")
        _validate_schema(value, definitions[name], root, path)
        return
    if "const" in schema and value != schema["const"]:
        raise VerificationError("SCHEMA_CONST")
    if "enum" in schema:
        options = schema["enum"]
        if type(options) is not list or value not in options:
            raise VerificationError("SCHEMA_ENUM")
    expected_type = schema.get("type")
    if expected_type is not None and (type(expected_type) is not str or not _schema_type(value, expected_type)):
        raise VerificationError("SCHEMA_TYPE")
    if type(value) is dict and expected_type == "object":
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        if type(required) is not list or type(properties) is not dict:
            raise VerificationError("SCHEMA_DEFINITION")
        if any(type(key) is not str for key in required) or not set(required).issubset(value):
            raise VerificationError("SCHEMA_REQUIRED")
        if schema.get("additionalProperties") is False and not set(value).issubset(properties):
            raise VerificationError("SCHEMA_ADDITIONAL")
        for key, item in value.items():
            if key in properties:
                _validate_schema(item, properties[key], root, f"{path}.{key}")
    if type(value) is list and expected_type == "array":
        minimum, maximum = schema.get("minItems"), schema.get("maxItems")
        if (minimum is not None and len(value) < minimum) or (maximum is not None and len(value) > maximum):
            raise VerificationError("SCHEMA_ARRAY")
        item_schema = schema.get("items")
        if type(item_schema) is not dict:
            raise VerificationError("SCHEMA_DEFINITION")
        for index, item in enumerate(value):
            _validate_schema(item, item_schema, root, f"{path}[{index}]")
    if type(value) is str:
        maximum = schema.get("maxLength")
        pattern = schema.get("pattern")
        if maximum is not None and len(value) > maximum:
            raise VerificationError("SCHEMA_STRING")
        if pattern is not None and (type(pattern) is not str or re.fullmatch(pattern, value) is None):
            raise VerificationError("SCHEMA_PATTERN")


def _throughput(trials: str, duration_ns: str) -> str:
    value = Decimal(trials) * Decimal(1_000_000_000) / Decimal(duration_ns)
    return format(value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP), "f")


def _expected_cells() -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (f"v1-{street}-o{opponents}-n{trials}", street, opponents, trials)
        for street in ("preflop", "flop", "turn", "river")
        for opponents in ("1", "3", "6")
        for trials in ("10000", "100000", "500000", "1000000")
    )


def _verify_sample(sample: dict[str, str], trials: str) -> None:
    if sample["throughput_per_second"] != _throughput(trials, sample["duration_ns"]):
        raise VerificationError("THROUGHPUT")


def _verify_admission(record: dict[str, Any]) -> None:
    environment = record["environment"]
    for name in ("before", "after"):
        snapshot = record["admission"][name]
        try:
            gpu_raw = bytes.fromhex(snapshot["gpu_snapshot_hex"])
            compute_raw = bytes.fromhex(snapshot["compute_snapshot_hex"])
            parts = [part.strip() for part in gpu_raw.decode("ascii").strip().split(",")]
        except (ValueError, UnicodeError) as exc:
            raise VerificationError("ADMISSION") from exc
        raw_uuid_ok = re.fullmatch(
            r"GPU-(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
            parts[0],
        ) if parts else None
        normalized_uuid = (
            "GPU-" + parts[0][4:].replace("-", "").lower()
            if raw_uuid_ok else ""
        )
        if (
            len(parts) != 3
            or raw_uuid_ok is None
            or normalized_uuid != snapshot["gpu_uuid"]
            or snapshot["gpu_uuid"] != environment["gpu_uuid"]
            or not all(re.fullmatch(r"[0-9]+", item) for item in parts[1:])
            or snapshot["free_bytes"] != str(int(parts[1]) * 1024 * 1024)
            or snapshot["total_bytes"] != str(int(parts[2]) * 1024 * 1024)
            or compute_raw != b""
            or snapshot["compute_applications"] != []
            or snapshot["gpu_snapshot_sha256"] != hashlib.sha256(gpu_raw).hexdigest()
            or snapshot["compute_snapshot_sha256"] != hashlib.sha256(compute_raw).hexdigest()
        ):
            raise VerificationError("ADMISSION")


def _verify_matrix(record: dict[str, Any]) -> None:
    matrix = record["matrix"]
    if tuple(cell["cell_id"] for cell in matrix) != tuple(row[0] for row in _expected_cells()):
        raise VerificationError("MATRIX_ORDER")
    for cell, (cell_id, street, opponents, trials) in zip(matrix, _expected_cells()):
        if (
            cell["cell_id"] != cell_id
            or cell["street"] != street
            or cell["opponent_count"] != opponents
            or cell["requested_trials"] != trials
            or cell["seed"] != "0x0123456789abcdef"
        ):
            raise VerificationError("MATRIX_IDENTITY")
        steady = cell["steady"]
        durations = tuple(int(item) for item in steady["durations_ns"])
        ordered = sorted(durations)
        aggregate = steady["aggregate"]
        expected = {
            "count": "30",
            "minimum_ns": str(ordered[0]),
            "p5_ns": str(ordered[ceil(5 * len(ordered) / 100) - 1]),
            "p50_ns": str(ordered[ceil(50 * len(ordered) / 100) - 1]),
            "p95_ns": str(ordered[ceil(95 * len(ordered) / 100) - 1]),
            "maximum_ns": str(ordered[-1]),
            "throughput_per_second": _throughput(trials, str(ordered[14])),
        }
        if aggregate != expected:
            raise VerificationError("STATISTICS")
        analytical = (
            cell["expected_analytical_sha256"],
            steady["warmup_analytical_sha256"],
            *steady["analytical_sha256s"],
            cell["stage"]["analytical_sha256"],
        )
        if any(digest != analytical[0] for digest in analytical[1:]):
            raise VerificationError("ANALYTICAL_MISMATCH")


def verify_record(record: Any, schema: Any) -> None:
    if type(schema) is not dict:
        raise VerificationError("SCHEMA")
    _check_schema_keywords(schema)
    _validate_schema(record, schema, schema)
    environment = record["environment"]
    if (
        environment["os"] != "Linux"
        or re.fullmatch(r"3\.13\.[0-9]+", environment["python_version"]) is None
        or environment["cupy_version"] != "14.1.1"
    ):
        raise VerificationError("ENVIRONMENT")
    if record["startup"]["canary_cell_id"] != "v1-preflop-o1-n10000":
        raise VerificationError("STARTUP_CANARY")
    _verify_sample(record["startup"]["cold"], "10000")
    _verify_sample(record["startup"]["warm"], "10000")
    _verify_admission(record)
    canary_digest = record["matrix"][0]["expected_analytical_sha256"]
    if (
        record["startup"]["cold"]["analytical_sha256"] != canary_digest
        or record["startup"]["warm"]["analytical_sha256"] != canary_digest
        or record["startup_cache"]["cold_seal"]["cold_result_sha256"] != canary_digest
        or record["startup_cache"]["warm_verified_seal"] != record["startup_cache"]["cold_seal"]
    ):
        raise VerificationError("ANALYTICAL_MISMATCH")
    cache_files = record["startup_cache"]["cold_seal"]["files"]
    cache_paths = [item["path"] for item in cache_files]
    if (
        cache_paths != sorted(cache_paths)
        or len(cache_paths) != len(set(cache_paths))
        or any(
            path == ".phase6c-cache-seal.json"
            or path.startswith("/")
            or any(part in ("", ".", "..") for part in path.split("/"))
            for path in cache_paths
        )
    ):
        raise VerificationError("CACHE_SEAL")
    expected_modes = ["inventory", "expected", "cold", "warm", "steady", "stage"]
    expected_cache_classes = [
        "inventory", "expected-isolated", "startup-cold", "startup-warm",
        "steady-isolated", "stage-isolated",
    ]
    for ordinal, (worker, mode, cache_class) in enumerate(
        zip(record["workers"], expected_modes, expected_cache_classes)
    ):
        payload = _canonical(worker["payload"])
        digest = hashlib.sha256(payload).hexdigest()
        if (
            worker["mode"] != mode
            or worker["ordinal"] != str(ordinal)
            or worker["cache_class"] != cache_class
            or worker["payload"].get("mode") != mode
            or worker["stdout_bytes"] != str(len(payload))
            or worker["stderr_bytes"] != "0"
            or worker["stdout_sha256"] != digest
            or worker["stderr_sha256"] != hashlib.sha256(b"").hexdigest()
            or worker["payload_sha256"] != digest
            or worker["output_limit_bytes"] != str(MAX_EVIDENCE_BYTES)
        ):
            raise VerificationError("WORKER_TRANSCRIPT")
    payloads = {worker["mode"]: worker["payload"] for worker in record["workers"]}
    installation = payloads["inventory"].get("installation")
    if (
        payloads["inventory"].get("environment") != record["environment"]
        or type(installation) is not dict
        or installation.get("wheel_basename") != record["artifacts"]["wheel"]["basename"]
        or installation.get("wheel_sha256") != record["artifacts"]["wheel"]["sha256"]
        or installation.get("wheel_contents_verified") != "true"
    ):
        raise VerificationError("INVENTORY_BINDING")
    expected_payload = payloads["expected"].get("analytical_sha256s")
    steady_payload = payloads["steady"].get("cells")
    stage_worker = payloads["stage"]
    plan_text = stage_worker.get("planned_batch_blocks")
    stage_payload = stage_worker.get("cells")
    if (
        set(stage_worker) != {"mode", "planned_batch_blocks", "batch_counts", "cells"}
        or plan_text != str(STAGE_BATCH_BLOCKS)
        or not all(type(value) is dict for value in (expected_payload, steady_payload, stage_payload))
    ):
        raise VerificationError("WORKER_RECONSTRUCTION")
    expected_batch_counts = {
        cell["cell_id"]: str(ceil(int(cell["requested_trials"]) / (STAGE_BATCH_BLOCKS * STAGE_THREADS)))
        for cell in record["matrix"]
    }
    if stage_worker.get("batch_counts") != expected_batch_counts:
        raise VerificationError("WORKER_RECONSTRUCTION")
    for cell in record["matrix"]:
        cell_id = cell["cell_id"]
        expected_steady = dict(cell["steady"])
        expected_steady.pop("aggregate")
        if (
            expected_payload.get(cell_id) != cell["expected_analytical_sha256"]
            or steady_payload.get(cell_id) != expected_steady
            or stage_payload.get(cell_id) != cell["stage"]
        ):
            raise VerificationError("WORKER_RECONSTRUCTION")
    for mode in ("cold", "warm"):
        payload = payloads[mode]
        sample = record["startup"][mode]
        if (
            set(payload) != {"mode", "duration_ns", "analytical_sha256"}
            or payload["duration_ns"] != sample["duration_ns"]
            or payload["analytical_sha256"] != sample["analytical_sha256"]
        ):
            raise VerificationError("WORKER_RECONSTRUCTION")
    closure = record["artifacts"]["installed_wheel_byte_closure"]
    if [item["basename"] for item in closure] != sorted(item["basename"] for item in closure):
        raise VerificationError("ARTIFACT_ORDER")
    _verify_matrix(record)


def _regular_file(path: Path) -> bytes:
    try:
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise VerificationError("LOCAL_PATH")
        return path.read_bytes()
    except OSError as exc:
        raise VerificationError("LOCAL_PATH") from exc


def _file_record(path: Path, *, basename: str | None = None) -> dict[str, str]:
    data = _regular_file(path)
    return {
        "basename": path.name if basename is None else basename,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": str(len(data)),
    }


def _run_git(root: Path, arguments: list[str]) -> str:
    """Run a small fixed git query without a shell or inherited command text."""
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(root), *arguments], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
            timeout=5, text=True, encoding="ascii", errors="strict",
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
        raise VerificationError("CHECKOUT") from exc
    if result.returncode != 0 or len(result.stdout) > 4096:
        raise VerificationError("CHECKOUT")
    return result.stdout


def _checkout_identity(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise VerificationError("CHECKOUT")
    head = _run_git(root, ["rev-parse", "--verify", "HEAD"]).strip()
    branch = _run_git(root, ["symbolic-ref", "--quiet", "--short", "HEAD"]).strip()
    status = _run_git(root, ["status", "--porcelain=v1", "--untracked-files=all"])
    if re.fullmatch(r"[0-9a-f]{40}", head) is None or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", branch) is None:
        raise VerificationError("CHECKOUT")
    return {"git_sha": head, "branch": branch, "clean": "true" if not status else "false"}


def _archive_closure(path: Path) -> list[dict[str, str]]:
    _regular_file(path)
    try:
        with zipfile.ZipFile(path) as archive:
            rows: list[dict[str, str]] = []
            names: set[str] = set()
            for member in archive.infolist():
                name = member.filename
                parts = name.split("/")
                mode = member.external_attr >> 16
                if (
                    not name or name in names or member.flag_bits & 1 or name.startswith("/")
                    or "" in parts or "." in parts or ".." in parts or "\\" in name
                    or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))
                ):
                    raise VerificationError("WHEEL_ARCHIVE")
                names.add(name)
                try:
                    data = archive.read(member)
                except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    raise VerificationError("WHEEL_ARCHIVE") from exc
                rows.append({"basename": name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": str(len(data))})
    except (OSError, zipfile.BadZipFile) as exc:
        raise VerificationError("WHEEL_ARCHIVE") from exc
    if not rows:
        raise VerificationError("WHEEL_ARCHIVE")
    return sorted(rows, key=lambda row: row["basename"])


def _scenario_manifest(directory: Path) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise VerificationError("SCENARIO_AUTHORITY")
    rows: list[dict[str, str]] = []
    try:
        paths = sorted(directory.glob("*.json"), key=lambda path: path.name)
    except OSError as exc:
        raise VerificationError("SCENARIO_AUTHORITY") from exc
    for path in paths:
        value = _load_canonical(path, maximum=1024)
        if type(value) is not dict or type(value.get("id")) is not str or path.name != value["id"].removeprefix("v1-") + ".json":
            raise VerificationError("SCENARIO_AUTHORITY")
        data = _regular_file(path)
        rows.append({"id": value["id"], "path": path.name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": str(len(data))})
    if len(rows) != 12 or len({row["id"] for row in rows}) != 12:
        raise VerificationError("SCENARIO_AUTHORITY")
    return {"format_version": "phase6c-scenario-manifest-v1", "scenarios": rows}


def _validate_context(context: VerificationContext) -> None:
    if type(context) is not VerificationContext or any(
        not isinstance(path, Path)
        or not path.is_absolute()
        or _has_symlink_component(path)
        for path in context.__dict__.values()
    ):
        raise VerificationError("CONTEXT")
    try:
        root = context.repo_root.resolve(strict=True)
        for path in (context.wheel, context.sdist, context.lockfile, context.scenario_directory, context.scenario_manifest):
            path.resolve(strict=True).relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise VerificationError("CONTEXT") from exc


def _verify_qualification_authorities(root: Path) -> None:
    for relative, expected in IMMUTABLE_QUALIFICATION_SHA256.items():
        path = root / relative
        if (
            _has_symlink_component(path)
            or not path.is_file()
            or hashlib.sha256(_regular_file(path)).hexdigest() != expected
        ):
            raise VerificationError("QUALIFICATION_AUTHORITY")
    authorities = (
        (
            "tools/verify_cuda_release_qualification.py",
            "validation/holdem/v1/cuda_release_qualification.json",
            "_phase5b_qualification_verifier",
        ),
        (
            "tools/verify_cuda_statistical_release_qualification.py",
            "validation/holdem/v1/cuda_statistical_release_qualification.json",
            "_phase5c_qualification_verifier",
        ),
    )
    for verifier_relative, record_relative, module_name in authorities:
        verifier_path = root / verifier_relative
        record_path = root / record_relative
        if (
            _has_symlink_component(verifier_path)
            or _has_symlink_component(record_path)
            or not verifier_path.is_file()
            or not record_path.is_file()
        ):
            raise VerificationError("QUALIFICATION_AUTHORITY")
        spec = importlib.util.spec_from_file_location(module_name, verifier_path)
        if spec is None or spec.loader is None:
            raise VerificationError("QUALIFICATION_AUTHORITY")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            result = module.verify(record_path, root)
        except Exception as exc:
            raise VerificationError("QUALIFICATION_AUTHORITY") from exc
        if result != 0:
            raise VerificationError("QUALIFICATION_AUTHORITY")


def verify_bound_record(record: Any, schema: Any, context: VerificationContext) -> None:
    """Verify canonical evidence against explicit, local checkout/artifact authority."""
    verify_record(record, schema)
    _validate_context(context)
    root = context.repo_root.resolve()
    _verify_qualification_authorities(root)
    source = _checkout_identity(root)
    source["benchmark_tool_sha256"] = hashlib.sha256(_regular_file(root / "tools/benchmark_equity.py")).hexdigest()
    source["cuda_runtime_source_sha256"] = hashlib.sha256(_regular_file(root / "src/poker_knight_ng/_cuda_runtime.py")).hexdigest()
    if record["source"] != source:
        raise VerificationError("SOURCE_BINDING")
    artifacts = record["artifacts"]
    expected = {
        "wheel": _file_record(context.wheel), "sdist": _file_record(context.sdist),
        "lock": _file_record(context.lockfile), "scenario_manifest": _file_record(context.scenario_manifest),
        **{
            name: _file_record(root / relative)
            for name, relative in QUALIFICATION_FILES.items()
        },
    }
    if any(artifacts[key] != value for key, value in expected.items()):
        raise VerificationError("ARTIFACT_BINDING")
    manifest = _scenario_manifest(context.scenario_directory)
    if _regular_file(context.scenario_manifest) != _canonical(manifest):
        raise VerificationError("SCENARIO_AUTHORITY")
    if artifacts["installed_wheel_byte_closure"] != _archive_closure(context.wheel):
        raise VerificationError("WHEEL_CLOSURE")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 13 or arguments[1::2] != [
        "--repo-root", "--wheel", "--sdist", "--lockfile", "--scenario-dir", "--scenario-manifest",
    ]:
        return 2
    try:
        evidence = _load_canonical(Path(arguments[0]), maximum=MAX_EVIDENCE_BYTES)
        context = VerificationContext(
            repo_root=Path(arguments[2]), wheel=Path(arguments[4]), sdist=Path(arguments[6]),
            lockfile=Path(arguments[8]), scenario_directory=Path(arguments[10]), scenario_manifest=Path(arguments[12]),
        )
        verify_bound_record(evidence, _load_schema(SCHEMA_PATH), context)
    except (VerificationError, OSError):
        print("Phase 6C private benchmark evidence: FAIL", file=sys.stderr)
        return 1
    print("Phase 6C private benchmark evidence: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
