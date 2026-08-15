"""Public Phase 6C baseline projection contract (no private values are asserted)."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, getcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[2]
TOOL = ROOT / "tools/project_cuda_benchmark_baseline.py"
PUBLIC = ROOT / "validation/holdem/v1/cuda_benchmark_baseline.json"
MANIFEST = ROOT / "validation/holdem/v1/manifests/cuda_benchmark_baseline.sha256"
PRIVATE_TEXT = os.environ.get("PKNG_PHASE6C_PRIVATE_EVIDENCE")
PRIVATE = Path(PRIVATE_TEXT) if PRIVATE_TEXT else None
PRIVATE_SHA256 = "b708cc18ce5ded2131620f21cc5cd6e15dad1f7c2b42140872dca394859c3251"
SOURCE = "71393540fefba4b987a446e0d2ec954e72bbe2e2"
SUPERSEDED_SOURCE = "3932868ff43b175b80f02b5f8a4a45d1d450ec13"
SUPERSEDED_PUBLIC_COMMIT = "7fb617b900c06102caafe240ff95afe7fef2aa58"
SUPERSEDED_PUBLIC_SHA256 = "78fbb7f734763528c703b532a0cc7cd80a9a1d46fcc9813feefb20cb45a820b8"


def load_tool():
    spec = importlib.util.spec_from_file_location("public_baseline_projection", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_projection_is_canonical_complete_and_manifest_bound():
    tool = load_tool()
    public = tool.load_canonical(PUBLIC, maximum=tool.MAX_PUBLIC_BYTES)
    tool.verify_public(public, ROOT, SOURCE)
    assert len(public["cells"]) == 48
    assert PUBLIC.read_bytes() == tool.canonical(public)
    assert MANIFEST.read_text("ascii") == f"{hashlib.sha256(PUBLIC.read_bytes()).hexdigest()}  {PUBLIC.name}\n"


def test_accepted_candidate_satisfies_preregistered_public_comparison() -> None:
    getcontext().prec = 50
    historical_bytes = subprocess.run(
        ["git", "show", f"{SUPERSEDED_PUBLIC_COMMIT}:validation/holdem/v1/cuda_benchmark_baseline.json"],
        cwd=ROOT, check=True, capture_output=True, timeout=20,
    ).stdout
    assert hashlib.sha256(historical_bytes).hexdigest() == SUPERSEDED_PUBLIC_SHA256
    baseline = json.loads(historical_bytes)
    candidate = json.loads(PUBLIC.read_bytes())
    baseline_cells = {cell["scenario_id"]: cell for cell in baseline["cells"]}
    candidate_cells = {cell["scenario_id"]: cell for cell in candidate["cells"]}
    selected = sorted(
        scenario_id for scenario_id, cell in baseline_cells.items()
        if cell["requested_trials"] == "1000000"
    )
    assert len(selected) == 12

    baseline_p50 = [int(baseline_cells[scenario_id]["steady"]["p50_ns"]) for scenario_id in selected]
    candidate_p50 = [int(candidate_cells[scenario_id]["steady"]["p50_ns"]) for scenario_id in selected]
    baseline_p95 = [int(baseline_cells[scenario_id]["steady"]["p95_ns"]) for scenario_id in selected]
    candidate_p95 = [int(candidate_cells[scenario_id]["steady"]["p95_ns"]) for scenario_id in selected]
    assert all(after <= before for before, after in zip(baseline_p50, candidate_p50))
    assert all(after <= before for before, after in zip(baseline_p95, candidate_p95))
    assert math.prod(candidate_p50) * 10**12 <= math.prod(baseline_p50) * 9**12

    ratios = [Decimal(after) / Decimal(before) for before, after in zip(baseline_p50, candidate_p50)]
    geometric_ratio = (sum(ratio.ln() for ratio in ratios) / Decimal(12)).exp()
    assert ((Decimal(1) - geometric_ratio) * Decimal(100)).quantize(Decimal("0.001")) == Decimal("64.742")
    baseline_stage = sum(int(baseline_cells[item]["stages"]["simulate_ns"]) for item in selected)
    candidate_stage = sum(int(candidate_cells[item]["stages"]["simulate_ns"]) for item in selected)
    assert candidate_stage < baseline_stage


@pytest.mark.skipif(PRIVATE is None or not PRIVATE.is_file(), reason="access-controlled Phase 6C private evidence unavailable")
def test_projection_is_reproducible_from_private_digest_without_exposing_private_values(tmp_path):
    tool = load_tool()
    assert PRIVATE is not None
    output = tmp_path / "baseline.json"
    generated = tool.project(PRIVATE, PRIVATE_SHA256, ROOT, SOURCE)
    output.write_bytes(tool.canonical(generated))
    assert output.read_bytes() == PUBLIC.read_bytes()
    tool.verify_public(generated, ROOT, SOURCE)


@pytest.mark.parametrize("mutate", [
    lambda value: value.__setitem__("hostname", "forbidden"),
    lambda value: value["cells"][0].__setitem__("seed", "forbidden"),
    lambda value: value["cells"][0]["steady"].__setitem__("durations_ns", ["1"]),
    lambda value: value["cuda"].__setitem__("gpu_product", "forbidden"),
    lambda value: value["cells"].reverse(),
    lambda value: value["cells"][0]["steady"].__setitem__("p50_throughput_per_second", "1.000"),
    lambda value: value["outcome"].__setitem__("category", "raw-error"),
])
def test_verifier_rejects_public_schema_privacy_order_and_statistics_attacks(mutate):
    tool = load_tool()
    value = tool.load_canonical(PUBLIC, maximum=tool.MAX_PUBLIC_BYTES)
    mutate(value)
    with pytest.raises(tool.ProjectionError):
        tool.verify_public(value, ROOT, SOURCE)


def test_cli_verifies_manifest_and_refuses_digest_mismatch(tmp_path):
    tool = load_tool()
    bad_digest = "0" * 64
    with pytest.raises(tool.ProjectionError, match="PRIVATE_DIGEST"):
        tool.project(PUBLIC, bad_digest, ROOT, SOURCE)
    output = tmp_path / "projection.json"
    completed = subprocess.run(
        [sys.executable, str(TOOL), "project", str(PUBLIC), "--private-sha256", bad_digest,
         "--repo-root", str(ROOT), "--source-commit", SOURCE, "--output", str(output)],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 1
    assert not output.exists()

    completed = subprocess.run(
        [sys.executable, str(TOOL), "verify", str(PUBLIC), "--manifest", str(MANIFEST),
         "--repo-root", str(ROOT), "--source-commit", SOURCE],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout == "Phase 6C public baseline projection: PASS\n"
    assert completed.stderr == ""
