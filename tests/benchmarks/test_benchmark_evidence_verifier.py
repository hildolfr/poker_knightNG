from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_TEST = ROOT / "tests/benchmarks/test_benchmark_evidence_schema.py"
VERIFIER = ROOT / "tools/verify_cuda_benchmark_evidence.py"
SCHEMA = ROOT / "validation/holdem/v1/cuda_benchmark_private.schema.json"
SCENARIO_DIRECTORY = ROOT / "benchmarks/scenarios/v1"
SCENARIO_MANIFEST = ROOT / "benchmarks/scenarios-v1.json"


def load_fixture_module():
    spec = importlib.util.spec_from_file_location("benchmark_evidence_fixture", SCHEMA_TEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_verifier():
    spec = importlib.util.spec_from_file_location("benchmark_evidence_verifier", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_scenario_manifest_is_canonical_and_complete():
    verifier = load_verifier()
    expected = verifier._scenario_manifest(SCENARIO_DIRECTORY)
    assert SCENARIO_MANIFEST.read_bytes() == verifier._canonical(expected)


def test_published_phase5_qualification_authorities_verify_against_checkout():
    verifier = load_verifier()
    verifier._verify_qualification_authorities(ROOT)


def test_immutable_phase5_authority_byte_change_is_rejected_before_execution(tmp_path):
    verifier = load_verifier()
    for relative in verifier.IMMUTABLE_QUALIFICATION_SHA256:
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    target = tmp_path / "tools/verify_cuda_release_qualification.py"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(verifier.VerificationError, match="QUALIFICATION_AUTHORITY"):
        verifier._verify_qualification_authorities(tmp_path)


def record():
    return load_fixture_module()._record()


def test_stdlib_semantic_verifier_accepts_record_but_unbound_cli_is_refused(tmp_path):
    verifier = load_verifier()
    evidence = record()
    schema = json.loads(SCHEMA.read_text("ascii"))
    verifier.verify_record(evidence, schema)

    path = tmp_path / "private.json"
    path.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n", "ascii")
    completed = subprocess.run(
        [sys.executable, str(VERIFIER), str(path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == ""


@pytest.mark.parametrize("mutate", [
    lambda value: value["matrix"].__setitem__(1, deepcopy(value["matrix"][0])),
    lambda value: value["matrix"].reverse(),
    lambda value: value["workers"].reverse(),
    lambda value: value["workers"][5]["payload"].__setitem__("planned_batch_blocks", "1"),
    lambda value: value["environment"].__setitem__("os", "Windows"),
    lambda value: value["environment"].__setitem__("python_version", "3.14.0"),
    lambda value: value["environment"].__setitem__("cupy_version", "14.1.2"),
    lambda value: value["admission"]["before"].__setitem__("compute_snapshot_hex", "00"),
    lambda value: value["admission"]["before"].__setitem__("gpu_snapshot_sha256", "0" * 64),
    lambda value: value["admission"]["after"].__setitem__("gpu_uuid", "GPU-" + "f" * 32),
    lambda value: value["startup_cache"]["cold_seal"].__setitem__("cold_result_sha256", "f" * 64),
    lambda value: value["startup_cache"]["warm_verified_seal"]["files"][0].__setitem__("sha256", "f" * 64),
    lambda value: value["matrix"][0]["steady"]["aggregate"].__setitem__("p50_ns", "1"),
    lambda value: value["matrix"][0]["steady"]["aggregate"].__setitem__(
        "throughput_per_second", "1.000"
    ),
    lambda value: value["matrix"][0]["steady"].__setitem__(
        "stages", {"h2d_ns": "1", "simulate_ns": "1", "reduction_ns": "1", "d2h_ns": "1"}
    ),
    lambda value: value["startup"].__setitem__("canary_cell_id", "v1-river-o6-n1000000"),
])
def test_verifier_rejects_semantic_and_timing_conflation_mutations(mutate):
    verifier = load_verifier()
    evidence = record()
    mutate(evidence)
    with pytest.raises(verifier.VerificationError):
        verifier.verify_record(evidence, json.loads(SCHEMA.read_text("ascii")))


def test_verifier_rejects_an_unsupported_schema_keyword():
    verifier = load_verifier()
    schema = json.loads(SCHEMA.read_text("ascii"))
    schema["properties"]["benchmark_id"]["default"] = "unreviewed"
    with pytest.raises(verifier.VerificationError, match="SCHEMA_KEYWORD"):
        verifier.verify_record(record(), schema)


def test_verifier_rejects_transcript_consistent_stage_batch_plan_mismatch():
    verifier = load_verifier()
    evidence = record()
    worker = evidence["workers"][5]
    worker["payload"]["planned_batch_blocks"] = "1"
    encoded = verifier._canonical(worker["payload"])
    digest = hashlib.sha256(encoded).hexdigest()
    worker["stdout_bytes"] = str(len(encoded))
    worker["stdout_sha256"] = digest
    worker["payload_sha256"] = digest
    with pytest.raises(verifier.VerificationError, match="WORKER_RECONSTRUCTION"):
        verifier.verify_record(evidence, json.loads(SCHEMA.read_text("ascii")))


@pytest.mark.parametrize("location", ["expected", "cold", "warm", "warmup", "steady", "stage"])
def test_verifier_rejects_any_analytical_digest_disagreement(location):
    verifier = load_verifier()
    evidence = record()
    cell = evidence["matrix"][0]
    if location == "expected":
        cell["expected_analytical_sha256"] = "b" * 64
    elif location in ("cold", "warm"):
        evidence["startup"][location]["analytical_sha256"] = "b" * 64
    elif location == "warmup":
        cell["steady"]["warmup_analytical_sha256"] = "b" * 64
    elif location == "steady":
        cell["steady"]["analytical_sha256s"][17] = "b" * 64
    else:
        cell["stage"]["analytical_sha256"] = "b" * 64
    with pytest.raises(verifier.VerificationError):
        verifier.verify_record(evidence, json.loads(SCHEMA.read_text("ascii")))
