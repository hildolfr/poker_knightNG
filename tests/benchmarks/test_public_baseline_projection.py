"""Public Phase 6C baseline projection contract (no private values are asserted)."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
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
PRIVATE_SHA256 = "432c583e98856eef186939cf0e3adc7fb63296a3f3ca1b78b011dbb5a1b3fd1d"
SOURCE = "3932868ff43b175b80f02b5f8a4a45d1d450ec13"


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
