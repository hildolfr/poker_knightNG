"""Public Phase 5C CUDA statistical qualification closeout."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "validation/holdem/v1/cuda_statistical_release_qualification.json"
TOOL = ROOT / "tools/verify_cuda_statistical_release_qualification.py"
PUBLIC_PATHS = (
    RECORD,
    ROOT / "README.md",
    ROOT / "src/poker_knight_ng/README.md",
    ROOT / "validation/holdem/v1/QUALIFICATION.md",
    ROOT / "validation/holdem/v1/SPEC.md",
    ROOT / "validation/holdem/v1/STATISTICAL_QUALIFICATION.md",
)


def load_tool():
    spec = importlib.util.spec_from_file_location("verify_cuda_statistical_release", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_phase5c_public_record_verifies_strictly() -> None:
    tool = load_tool()
    assert tool.verify(RECORD, ROOT) == 0
    assert RECORD.read_bytes() == tool.canonical(tool.strict_json(RECORD.read_bytes()))


def test_public_record_rejects_hostile_mutations() -> None:
    tool = load_tool()
    record = tool.strict_json(RECORD.read_bytes())
    mutations = []
    extra = deepcopy(record)
    extra["unexpected"] = "x"
    mutations.append(extra)
    source = deepcopy(record)
    source["source"]["git_sha"] = "0" * 40
    mutations.append(source)
    evidence = deepcopy(record)
    evidence["evidence"]["qualification_sha256"] = "0" * 64
    mutations.append(evidence)
    plan = deepcopy(record)
    plan["geometries"]["capacity_1"]["batch_plan_sha256"] = "0" * 64
    mutations.append(plan)
    aggregate = deepcopy(record)
    aggregate["geometries"]["capacity_7"]["aggregate_sha256"] = "0" * 64
    mutations.append(aggregate)
    statistics = deepcopy(record)
    statistics["geometries"]["capacity_256"]["statistics"]["hoeffding"]["status"] = "failed"
    mutations.append(statistics)
    private = deepcopy(record)
    private["environment"]["host_path"] = "/home/operator/private"
    mutations.append(private)
    for candidate in mutations:
        with pytest.raises(tool.VerificationError):
            tool._verify_record(candidate, ROOT)


def test_public_projection_is_private_safe_and_cpu_only() -> None:
    forbidden = (
        "/home/", "/tmp/", "desktop-drizzt", "NVIDIA GeForce", "RTX 5070",
        "compute_apps", "process_name", "memory_free",
    )
    for path in PUBLIC_PATHS:
        text = path.read_text("utf-8")
        assert not any(token in text for token in forbidden)
    record_text = RECORD.read_text("utf-8")
    assert not any(token in record_text for token in (
        '"duration_ns"', '"pid"', '"memory_free_before_bytes"',
        '"memory_free_after_bytes"',
    ))

    probe = """
import importlib.util
from pathlib import Path
import sys

path = Path.cwd() / "tools/verify_cuda_statistical_release_qualification.py"
spec = importlib.util.spec_from_file_location("phase5c_public_verifier", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "cupy" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT,
        check=False, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_public_source_bindings_are_from_executed_commit() -> None:
    tool = load_tool()
    record = tool.strict_json(RECORD.read_bytes())
    relative = "validation/holdem/v1/SPEC.md"
    historical = record["source"]["bindings"][relative]
    blobs = tool._source_blobs(ROOT)
    assert historical == tool.hashlib.sha256(blobs[relative]).hexdigest()
    tool._verify_record(record, ROOT)


def test_duplicate_keys_and_json_numbers_fail_closed() -> None:
    tool = load_tool()
    with pytest.raises(tool.VerificationError, match="duplicate key"):
        tool.strict_json(b'{"a":"1","a":"2"}\n')
    with pytest.raises(tool.VerificationError, match="JSON numbers"):
        tool.strict_json(b'{"a":1}\n')


def test_publication_manifest_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = load_tool()
    manifest_path = ROOT / tool.MANIFEST_RELATIVE
    raw = manifest_path.read_bytes()
    tool._verify_manifest(ROOT)

    real_read = tool.read_regular
    lines = raw.splitlines(keepends=True)
    variants = (
        b"".join(lines[:-1]),
        raw + (b"0" * 64) + b"  unexpected.txt\n",
    )
    for candidate in variants:
        def fake_read(path, limit=tool._base.MAX_BYTES, *, _candidate=candidate):
            if path == manifest_path:
                return _candidate
            return real_read(path, limit)

        monkeypatch.setattr(tool, "read_regular", fake_read)
        with pytest.raises(tool.VerificationError, match="manifest closure mismatch"):
            tool._verify_manifest(ROOT)
