"""Phase 7B no-listener HTTP runtime-selection evidence."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[2]
PROBE = ROOT / "tools" / "spikes" / "phase7b_h11_probe.py"
EVIDENCE = ROOT / "validation" / "service" / "v1" / "phase7b_runtime_spike.json"
MANIFEST = ROOT / "validation" / "service" / "v1" / "phase7b_runtime_spike.sha256"
README = ROOT / "spikes" / "001-http-runtime-selection" / "README.md"
CI = ROOT / ".github" / "workflows" / "ci.yml"
BASE = "6c75cf7b260a1a83f2fbc8cba7cd81c5d1198d70"
H11_WHEEL_SHA256 = "63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86"


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def test_dependency_selection_preserves_zero_dependency_base() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text("utf-8")
    assert "dependencies = []" in pyproject
    assert "h11" not in pyproject
    assert 'name = "h11"' not in (ROOT / "uv.lock").read_text("utf-8")


def test_probe_is_no_listener_and_reproduces_canonical_evidence() -> None:
    tree = ast.parse(PROBE.read_text("utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "socket" not in imported
    assert "asyncio" not in imported


def test_ci_reproduces_probe_from_hash_verified_wheel_bytes() -> None:
    ci = CI.read_text("utf-8")
    assert "Reproduce no-listener HTTP parser selection" in ci
    assert "https://files.pythonhosted.org/packages/04/4b/29cac41a4d98d144bf5f6d33995617b185d14b22401f75ca86f384e87ff1/h11-0.16.0-py3-none-any.whl" in ci
    assert H11_WHEEL_SHA256 in ci
    assert 'test "$(stat -c %s "$wheel")" = "37515"' in ci
    assert "sha256sum --check -" in ci
    assert 'uv pip install --python "$venv/bin/python" "$wheel"' in ci
    assert '"$venv/bin/python" tools/spikes/phase7b_h11_probe.py' in ci
    assert "validation/service/v1/phase7b_runtime_spike.json" in ci


def test_evidence_is_closed_canonical_and_hash_bound() -> None:
    raw = EVIDENCE.read_bytes()
    value = json.loads(raw)
    assert raw == canonical(value)
    assert MANIFEST.read_text("ascii") == (
        f"{hashlib.sha256(raw).hexdigest()}  {EVIDENCE.name}\n"
    )
    assert set(value) == {
        "adapter_requirements",
        "base_commit",
        "candidate",
        "format_version",
        "no_listener",
        "observations",
        "rejected_candidates",
        "selection",
        "verdict",
    }
    assert value["format_version"] == "poker-knight-ng-phase7b-runtime-spike-v1"
    assert value["base_commit"] == BASE
    assert value["candidate"] == {
        "name": "h11",
        "version": "0.16.0",
        "wheel_sha256": H11_WHEEL_SHA256,
    }
    assert value["no_listener"] is True
    assert value["selection"] == "direct-h11-with-bounded-raw-admission"
    assert value["verdict"] == "VALIDATED"
    assert value["rejected_candidates"] == {
        "aiohttp": "3.14.3",
        "uvicorn": "0.52.3",
    }
    assert value["observations"] == {
        "conflicting_content_length_rejected": True,
        "duplicate_equal_content_length_normalized": True,
        "explicit_connection_close_serialized": True,
        "malformed_header_rejected": True,
        "pipelined_bytes_exposed_as_trailing_data": True,
        "transfer_encoding_chunked_accepted": True,
        "transfer_encoding_with_content_length_accepted": True,
        "valid_request_body_events": True,
    }
    assert value["adapter_requirements"] == [
        "bounded-raw-header-admission-before-h11",
        "exactly-one-canonical-content-length",
        "reject-all-transfer-encoding",
        "header-and-body-monotonic-deadlines",
        "exact-body-and-surplus-byte-rejection",
        "inspect-h11-trailing-data",
        "never-call-start-next-cycle",
        "explicit-connection-close-and-socket-close",
    ]


def test_root_import_remains_h11_inert() -> None:
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
        "import poker_knight_ng; assert 'h11' not in sys.modules"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stdout == completed.stderr == b""


def test_spike_readme_records_comparison_and_validated_verdict() -> None:
    text = README.read_text("utf-8")
    for required in (
        "## Verdict: VALIDATED",
        "h11 0.16.0",
        "Uvicorn 0.52.3",
        "aiohttp 3.14.3",
        "no listener",
        "raw framing admission",
        "do not call `start_next_cycle()`",
    ):
        assert required in text
