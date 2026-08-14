#!/usr/bin/env python3
"""Deterministically project verified private Phase 6C evidence into an allowlisted public baseline."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import hashlib
import importlib.util
import json
from math import ceil
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

MAX_PUBLIC_BYTES = 256 * 1024
MAX_PRIVATE_BYTES = 2 * 1024 * 1024
HERE = Path(__file__).resolve().parents[1]
PRIVATE_VERIFIER = HERE / "tools/verify_cuda_benchmark_evidence.py"
SCHEMA = HERE / "validation/holdem/v1/cuda_benchmark_baseline.schema.json"


class ProjectionError(ValueError):
    pass


def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            raise ProjectionError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def canonical(value: Any) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProjectionError("CANONICAL_JSON") from exc


def load_canonical(path: Path, *, maximum: int) -> Any:
    try:
        data = path.read_bytes()
        value = json.loads(data.decode("ascii"), object_pairs_hook=pairs,
            parse_int=lambda _: (_ for _ in ()).throw(ProjectionError("JSON_NUMBER")),
            parse_float=lambda _: (_ for _ in ()).throw(ProjectionError("JSON_NUMBER")),
            parse_constant=lambda _: (_ for _ in ()).throw(ProjectionError("JSON_CONSTANT")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProjectionError("JSON") from exc
    if not data or len(data) > maximum or not data.endswith(b"\n") or b"\r" in data or canonical(value) != data:
        raise ProjectionError("NONCANONICAL")
    return value


def git_bytes(root: Path, commit: str, relative: str) -> bytes:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None or not re.fullmatch(r"[A-Za-z0-9._/-]+", relative):
        raise ProjectionError("SOURCE")
    try:
        answer = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{relative}"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProjectionError("SOURCE") from exc
    if answer.returncode or len(answer.stdout) > 4 * 1024 * 1024:
        raise ProjectionError("SOURCE")
    return answer.stdout


def source_hash(root: Path, commit: str, relative: str) -> str:
    return hashlib.sha256(git_bytes(root, commit, relative)).hexdigest()


def expected_cells() -> tuple[tuple[str, str, str, str], ...]:
    return tuple((f"v1-{street}-o{opponents}-n{trials}", street, opponents, trials)
        for street in ("preflop", "flop", "turn", "river") for opponents in ("1", "3", "6") for trials in ("10000", "100000", "500000", "1000000"))


def throughput(trials: str, ns: str) -> str:
    return format((Decimal(trials) * Decimal(1_000_000_000) / Decimal(ns)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP), "f")


def private_verifier() -> Any:
    spec = importlib.util.spec_from_file_location("phase6c_private_verifier", PRIVATE_VERIFIER)
    if spec is None or spec.loader is None:
        raise ProjectionError("PRIVATE_VERIFIER")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project(private_path: Path, private_sha256: str, root: Path, source_commit: str) -> dict[str, Any]:
    data = private_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != private_sha256 or re.fullmatch(r"[0-9a-f]{64}", private_sha256) is None:
        raise ProjectionError("PRIVATE_DIGEST")
    private = load_canonical(private_path, maximum=MAX_PRIVATE_BYTES)
    try:
        verifier = private_verifier()
        verifier.verify_record(private, verifier._load_schema(verifier.SCHEMA_PATH))
    except Exception as exc:
        raise ProjectionError("PRIVATE_SEMANTICS") from exc
    if private["source"]["git_sha"] != source_commit or private["source"]["clean"] != "true":
        raise ProjectionError("SOURCE_BINDING")
    artifacts = private["artifacts"]
    cells = []
    for cell, (ident, street, opponents, trials) in zip(private["matrix"], expected_cells()):
        aggregate = cell["steady"]["aggregate"]
        cells.append({"scenario_id": ident, "street": street, "opponent_count": opponents, "requested_trials": trials,
            "run_class": "steady", "steady": {"p5_ns": aggregate["p5_ns"], "p50_ns": aggregate["p50_ns"], "p95_ns": aggregate["p95_ns"], "p50_throughput_per_second": aggregate["throughput_per_second"]},
            "stages": dict(cell["stage"]["durations"])})
    result = {"format_version": "phase6c-public-baseline-v1", "benchmark_id": private["benchmark_id"],
        "source": {"clean_commit_sha": source_commit},
        "hashes": {"lockfile_sha256": artifacts["lock"]["sha256"], "scenario_manifest_sha256": artifacts["scenario_manifest"]["sha256"],
            "controller_sha256": source_hash(root, source_commit, "tools/run_cuda_benchmark.py"), "method_sha256": source_hash(root, source_commit, "docs/performance-method.md"),
            "wheel_sha256": artifacts["wheel"]["sha256"], "sdist_sha256": artifacts["sdist"]["sha256"], "installed_closure_sha256": hashlib.sha256(canonical(artifacts["installed_wheel_byte_closure"])).hexdigest(),
            "phase5b_authority_sha256": artifacts["phase5b_qualification"]["sha256"], "phase5b_manifest_sha256": artifacts["phase5b_manifest"]["sha256"],
            "phase5c_authority_sha256": artifacts["phase5c_qualification"]["sha256"], "phase5c_manifest_sha256": artifacts["phase5c_manifest"]["sha256"], "private_evidence_sha256": private_sha256},
        "cuda": {"identity": ["cuda-uuid:" + private["environment"]["gpu_uuid"][4:], "cuda-source-sha256:" + private["source"]["cuda_runtime_source_sha256"]]},
        "outcome": {"protocol": "complete", "category": "complete"}, "cells": cells}
    verify_public(result, root, source_commit)
    return result


def verify_public(value: Any, root: Path, source_commit: str) -> None:
    if type(value) is not dict or set(value) != {"format_version", "benchmark_id", "source", "hashes", "cuda", "outcome", "cells"}:
        raise ProjectionError("SCHEMA")
    if value["format_version"] != "phase6c-public-baseline-v1" or value["benchmark_id"] != "holdem-v1-cuda-baseline-1" or value["source"] != {"clean_commit_sha": source_commit} or value["outcome"] != {"protocol":"complete", "category":"complete"}:
        raise ProjectionError("IDENTITY")
    hashes = value["hashes"]
    required_hashes = {"lockfile_sha256","scenario_manifest_sha256","controller_sha256","method_sha256","wheel_sha256","sdist_sha256","installed_closure_sha256","phase5b_authority_sha256","phase5b_manifest_sha256","phase5c_authority_sha256","phase5c_manifest_sha256","private_evidence_sha256"}
    if type(hashes) is not dict or set(hashes) != required_hashes or any(type(item) is not str or re.fullmatch(r"[0-9a-f]{64}", item) is None for item in hashes.values()):
        raise ProjectionError("HASHES")
    if hashes["controller_sha256"] != source_hash(root, source_commit, "tools/run_cuda_benchmark.py") or hashes["method_sha256"] != source_hash(root, source_commit, "docs/performance-method.md"):
        raise ProjectionError("SOURCE_BYTES")
    identity = value["cuda"].get("identity") if type(value["cuda"]) is dict and set(value["cuda"]) == {"identity"} else None
    if type(identity) is not list or len(identity) != 2 or re.fullmatch(r"cuda-uuid:[0-9a-f]{32}", identity[0] or "") is None or identity[1] != "cuda-source-sha256:" + source_hash(root, source_commit, "src/poker_knight_ng/_cuda_runtime.py"):
        raise ProjectionError("CUDA")
    cells = value["cells"]
    if type(cells) is not list or len(cells) != 48:
        raise ProjectionError("CELLS")
    for cell, (ident, street, opponents, trials) in zip(cells, expected_cells()):
        if type(cell) is not dict or set(cell) != {"scenario_id","street","opponent_count","requested_trials","run_class","steady","stages"} or [cell["scenario_id"],cell["street"],cell["opponent_count"],cell["requested_trials"],cell["run_class"]] != [ident,street,opponents,trials,"steady"]:
            raise ProjectionError("CELL_IDENTITY")
        steady, stages = cell["steady"], cell["stages"]
        if type(steady) is not dict or set(steady) != {"p5_ns","p50_ns","p95_ns","p50_throughput_per_second"} or any(type(steady[key]) is not str and key != "p50_throughput_per_second" for key in steady):
            raise ProjectionError("CELL_SCHEMA")
        if not all(re.fullmatch(r"[1-9][0-9]*", steady[key]) for key in ("p5_ns","p50_ns","p95_ns")) or not int(steady["p5_ns"]) <= int(steady["p50_ns"]) <= int(steady["p95_ns"]) or steady["p50_throughput_per_second"] != throughput(trials, steady["p50_ns"]):
            raise ProjectionError("STATISTICS")
        if type(stages) is not dict or set(stages) != {"h2d_ns","simulate_ns","reduction_ns","d2h_ns"} or any(type(item) is not str or re.fullmatch(r"(?:0|[1-9][0-9]*)", item) is None for item in stages.values()):
            raise ProjectionError("STAGES")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        if args[:1] == ["project"] and len(args) == 10 and args[2::2] == ["--private-sha256","--repo-root","--source-commit","--output"]:
            output = Path(args[9])
            if output.exists(): raise ProjectionError("OUTPUT_EXISTS")
            output.write_bytes(canonical(project(Path(args[1]), args[3], Path(args[5]), args[7])))
        elif args[:1] == ["verify"] and len(args) == 8 and args[2::2] == ["--manifest","--repo-root","--source-commit"]:
            public = Path(args[1]); manifest = Path(args[3]); verify_public(load_canonical(public, maximum=MAX_PUBLIC_BYTES), Path(args[5]), args[7])
            if manifest.read_text("ascii") != f"{hashlib.sha256(public.read_bytes()).hexdigest()}  {public.name}\n": raise ProjectionError("MANIFEST")
        else: return 2
    except (ProjectionError, OSError): return 1
    print("Phase 6C public baseline projection: PASS")
    return 0

if __name__ == "__main__": raise SystemExit(main())
