"""Hostile local identity binding tests for the stdlib evidence verifier."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).parents[2]
VERIFIER = ROOT / "tools/verify_cuda_benchmark_evidence.py"
SCHEMA_TEST = ROOT / "tests/benchmarks/test_benchmark_evidence_schema.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file(path: Path, name: str | None = None) -> dict[str, str]:
    data = path.read_bytes()
    return {"basename": name or path.name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": str(len(data))}


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _refresh_worker(evidence: dict[str, object], mode: str) -> None:
    workers = evidence["workers"]
    assert isinstance(workers, list)
    worker = next(item for item in workers if item["mode"] == mode)
    encoded = _canonical(worker["payload"])
    digest = hashlib.sha256(encoded).hexdigest()
    worker["stdout_bytes"] = str(len(encoded))
    worker["stdout_sha256"] = digest
    worker["payload_sha256"] = digest


def _scenario(name: str, street: str, opponents: str, board: list[str]) -> dict[str, object]:
    return {"board_cards": board, "format_version": "phase6c-scenario-v1", "hero_cards": ["Ac", "Kd"], "id": f"v1-{street}-o{opponents}", "opponent_count": opponents, "seed": "0x0123456789abcdef", "street": street}


def _bound(tmp_path: Path):
    verifier = _load(VERIFIER, "binding_verifier")
    fixture = _load(SCHEMA_TEST, "binding_fixture")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "tools").mkdir()
    (repo / "src/poker_knight_ng").mkdir(parents=True)
    tool = repo / "tools/benchmark_equity.py"
    runtime = repo / "src/poker_knight_ng/_cuda_runtime.py"
    tool.write_bytes(b"benchmark tool\n")
    runtime.write_bytes(b"cuda runtime\n")
    (repo / "tools/verify_cuda_release_qualification.py").write_text(
        "def verify(record, root):\n    return 0\n", "ascii"
    )
    (repo / "tools/verify_cuda_statistical_release_qualification.py").write_text(
        "def verify(record, root):\n    return 0\n", "ascii"
    )
    lock = repo / "uv.lock"
    lock.write_bytes(b"lock\n")
    scenarios = repo / "benchmarks/scenarios"
    scenarios.mkdir(parents=True)
    rows = []
    for street, board in (("preflop", []), ("flop", ["2c", "3d", "4h"]), ("turn", ["2c", "3d", "4h", "5s"]), ("river", ["2c", "3d", "4h", "5s", "6c"])):
        for opponents in ("1", "3", "6"):
            value = _scenario(f"{street}-o{opponents}.json", street, opponents, board)
            path = scenarios / f"{street}-o{opponents}.json"
            path.write_bytes(_canonical(value))
            data = path.read_bytes()
            rows.append({"id": value["id"], "path": path.name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": str(len(data))})
    manifest = repo / "benchmarks/scenarios-v1.json"
    manifest.write_bytes(_canonical({"format_version": "phase6c-scenario-manifest-v1", "scenarios": sorted(rows, key=lambda item: item["path"])}))
    qualification_files = {
        "phase5b_qualification": repo / "validation/holdem/v1/cuda_release_qualification.json",
        "phase5b_manifest": repo / "validation/holdem/v1/manifests/cuda_release_qualification.sha256",
        "phase5c_qualification": repo / "validation/holdem/v1/cuda_statistical_release_qualification.json",
        "phase5c_manifest": repo / "validation/holdem/v1/manifests/cuda_statistical_release_qualification.sha256",
    }
    for name, path in qualification_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((name + "\n").encode("ascii"))
    wheel = repo / "dist/pkg.whl"
    wheel.parent.mkdir()
    members = {"poker_knight_ng/__init__.py": b"init\n", "poker_knight_ng/core.py": b"core\n"}
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    sdist = repo / "dist/pkg.tar.gz"
    sdist.write_bytes(b"sdist\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=test@example.invalid", "-c", "user.name=test", "commit", "-qm", "fixture"], cwd=repo, check=True)
    evidence = fixture._record()
    evidence["source"] = {"git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip(), "branch": "main", "clean": "true", "benchmark_tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(), "cuda_runtime_source_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest()}
    evidence["artifacts"] = {"wheel": _file(wheel), "sdist": _file(sdist), "lock": _file(lock), "scenario_manifest": _file(manifest), **{name: _file(path) for name, path in qualification_files.items()}, "installed_wheel_byte_closure": [{"basename": name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": str(len(data))} for name, data in members.items()]}
    inventory = next(item for item in evidence["workers"] if item["mode"] == "inventory")
    inventory["payload"]["installation"] = {
        "wheel_basename": wheel.name,
        "wheel_contents_verified": "true",
        "wheel_sha256": evidence["artifacts"]["wheel"]["sha256"],
    }
    _refresh_worker(evidence, "inventory")
    context = verifier.VerificationContext(repo_root=repo, wheel=wheel, sdist=sdist, lockfile=lock, scenario_directory=scenarios, scenario_manifest=manifest)
    setattr(verifier, "_verify_qualification_authorities", lambda _root: None)
    return verifier, evidence, context, repo


def test_bound_context_accepts_exact_temporary_checkout_and_artifacts(tmp_path):
    verifier, evidence, context, _repo = _bound(tmp_path)
    verifier.verify_bound_record(evidence, verifier._load_schema(verifier.SCHEMA_PATH), context)


def test_bound_context_rejects_symlinked_authority_ancestor(tmp_path):
    verifier, evidence, context, repo = _bound(tmp_path)
    alias = tmp_path / "repo-alias"
    alias.symlink_to(repo, target_is_directory=True)
    aliased = verifier.VerificationContext(
        repo_root=alias,
        wheel=alias / context.wheel.relative_to(repo),
        sdist=alias / context.sdist.relative_to(repo),
        lockfile=alias / context.lockfile.relative_to(repo),
        scenario_directory=alias / context.scenario_directory.relative_to(repo),
        scenario_manifest=alias / context.scenario_manifest.relative_to(repo),
    )
    with pytest.raises(verifier.VerificationError, match="CONTEXT"):
        verifier.verify_bound_record(evidence, verifier._load_schema(verifier.SCHEMA_PATH), aliased)


@pytest.mark.parametrize("target", ["git_sha", "branch", "benchmark_tool_sha256", "cuda_runtime_source_sha256"])
def test_bound_context_rejects_forged_source_identity(tmp_path, target):
    verifier, evidence, context, _repo = _bound(tmp_path)
    evidence["source"][target] = "0" * (40 if target == "git_sha" else 64) if target != "branch" else "forged"
    with pytest.raises(verifier.VerificationError):
        verifier.verify_bound_record(evidence, verifier._load_schema(verifier.SCHEMA_PATH), context)


@pytest.mark.parametrize("target", [
    "wheel", "sdist", "lock", "scenario_manifest",
    "phase5b_qualification", "phase5b_manifest",
    "phase5c_qualification", "phase5c_manifest",
])
def test_bound_context_rejects_forged_file_record_sha_basename_or_size(tmp_path, target):
    verifier, evidence, context, _repo = _bound(tmp_path)
    for field, value in (("sha256", "0" * 64), ("basename", "forged"), ("size_bytes", "99")):
        candidate = json.loads(json.dumps(evidence))
        candidate["artifacts"][target][field] = value
        with pytest.raises(verifier.VerificationError):
            verifier.verify_bound_record(candidate, verifier._load_schema(verifier.SCHEMA_PATH), context)


def test_bound_context_rejects_dirty_checkout_and_reordered_or_incomplete_closure(tmp_path):
    verifier, evidence, context, repo = _bound(tmp_path)
    (repo / "dirty").write_text("x", "ascii")
    with pytest.raises(verifier.VerificationError):
        verifier.verify_bound_record(evidence, verifier._load_schema(verifier.SCHEMA_PATH), context)
    (repo / "dirty").unlink()
    evidence["artifacts"]["installed_wheel_byte_closure"].reverse()
    with pytest.raises(verifier.VerificationError):
        verifier.verify_bound_record(evidence, verifier._load_schema(verifier.SCHEMA_PATH), context)
    evidence["artifacts"]["installed_wheel_byte_closure"].pop()
    with pytest.raises(verifier.VerificationError):
        verifier.verify_bound_record(evidence, verifier._load_schema(verifier.SCHEMA_PATH), context)


@pytest.mark.parametrize("member", ["../escape.py", "poker_knight_ng/link.py"])
def test_bound_context_rejects_unsafe_or_symlink_wheel_members(tmp_path, member):
    verifier, evidence, context, _repo = _bound(tmp_path)
    with zipfile.ZipFile(context.wheel, "w") as archive:
        info = zipfile.ZipInfo(member)
        if member.endswith("link.py"):
            info.external_attr = 0o120777 << 16
        archive.writestr(info, b"x")
    evidence["artifacts"]["wheel"] = _file(context.wheel)
    with pytest.raises(verifier.VerificationError):
        verifier.verify_bound_record(evidence, verifier._load_schema(verifier.SCHEMA_PATH), context)


def test_bound_context_rejects_duplicate_wheel_member(tmp_path):
    verifier, evidence, context, _repo = _bound(tmp_path)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(context.wheel, "w") as archive:
            archive.writestr("poker_knight_ng/core.py", b"first")
            archive.writestr("poker_knight_ng/core.py", b"second")
    evidence["artifacts"]["wheel"] = _file(context.wheel)
    with pytest.raises(verifier.VerificationError):
        verifier.verify_bound_record(evidence, verifier._load_schema(verifier.SCHEMA_PATH), context)


def test_cli_refuses_synthetic_authority_and_reports_generic_failure(tmp_path):
    _verifier, evidence, context, _repo = _bound(tmp_path)
    path = tmp_path / "evidence.json"
    path.write_bytes(_canonical(evidence))
    args = [sys.executable, str(VERIFIER), str(path), "--repo-root", str(context.repo_root), "--wheel", str(context.wheel), "--sdist", str(context.sdist), "--lockfile", str(context.lockfile), "--scenario-dir", str(context.scenario_directory), "--scenario-manifest", str(context.scenario_manifest)]
    failed = subprocess.run(args, capture_output=True, text=True, check=False)
    assert failed.returncode == 1
    assert failed.stdout == ""
    assert failed.stderr == "Phase 6C private benchmark evidence: FAIL\n"


def test_context_runtime_resolve_failure_is_generic_verification_error():
    verifier = _load(VERIFIER, "context_runtime_verifier")
    base = type(Path())

    class ExplodingPath(base):
        def resolve(self, strict=False):
            raise RuntimeError("resolution loop")

    path = ExplodingPath("/absolute/context")
    context = verifier.VerificationContext(
        repo_root=path,
        wheel=path / "wheel.whl",
        sdist=path / "sdist.tar.gz",
        lockfile=path / "uv.lock",
        scenario_directory=path / "scenarios",
        scenario_manifest=path / "manifest.json",
    )
    with pytest.raises(verifier.VerificationError, match="CONTEXT"):
        verifier._validate_context(context)
