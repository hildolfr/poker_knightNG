"""Committed CUDA release-qualification record contract."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "verify_cuda_release_qualification.py"
RECORD = ROOT / "validation" / "holdem" / "v1" / "cuda_release_qualification.json"


def load_tool():
    spec = importlib.util.spec_from_file_location("verify_cuda_release_qualification", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_cuda_qualification_record_verifies_strictly() -> None:
    assert TOOL.is_file() and RECORD.is_file()
    tool = load_tool()
    assert tool.verify(RECORD, ROOT) == 0


def test_public_docs_state_current_cpu_cuda_and_qualification_boundaries() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (ROOT / "README.md").read_bytes() == (ROOT / "src/poker_knight_ng/README.md").read_bytes()
    for required in (
        "CPUReferenceEngine",
        "CUDAEngine",
        "c2b3eb96413d17194a85144491c71539a4818452",
        "cuda_release_qualification.json",
        "solve() remains CPU-default",
    ):
        assert required in readme
    assert "There is no production CPU solver, deterministic deal stream, CUDA backend" not in readme

    specification = (ROOT / "validation/holdem/v1/SPEC.md").read_text(encoding="utf-8")
    for required in (
        "Phase 3 — deterministic CPU stream and engine: COMPLETE",
        "Phase 4 — explicit CUDA engine: COMPLETE",
        "Phase 5A — deterministic GPU qualification harness: COMPLETE",
        "Phase 5B — qualification publication: COMPLETE",
    ):
        assert required in specification

    qualification = (ROOT / "validation/holdem/v1/QUALIFICATION.md").read_text(encoding="utf-8")
    for required in (
        "cuda_release_qualification.json",
        "verify_cuda_release_qualification.py",
        "9e2edef60ec2a890b970ef83a8c114af0f56e74f7f0bf1c7e20c21e84ae5178d",
        "d011e0f5c4db4d12fcb5240b5996f0911af7f153c7942f16772f871917ca5263",
    ):
        assert required in qualification

    for legacy_name in ("TODO.md", "ARCHITECTURE.md"):
        opening = "\n".join((ROOT / legacy_name).read_text(encoding="utf-8").splitlines()[:8])
        assert "LEGACY — NON-AUTHORITATIVE" in opening


def test_record_parser_and_closed_schema_reject_hostile_mutations() -> None:
    tool = load_tool()
    raw = RECORD.read_bytes()
    record = tool.strict_json(raw)
    with pytest.raises(tool.VerificationError, match="duplicate key"):
        tool.strict_json(b'{"format_version":"1","format_version":"1"}')
    with pytest.raises(tool.VerificationError, match="JSON numbers"):
        tool.strict_json(b'{"value":1}')

    mutations = []
    extra = json.loads(json.dumps(record)); extra["unexpected"] = "x"; mutations.append(extra)
    decimal = json.loads(json.dumps(record)); decimal["tests"]["passed"] = "0605"; mutations.append(decimal)
    binding = json.loads(json.dumps(record)); binding["source"]["bindings"]["uv.lock"] = "0" * 64; mutations.append(binding)
    race = json.loads(json.dumps(record)); race["sanitizers"]["racecheck"]["summary"] = "ERROR SUMMARY: 0 errors"; mutations.append(race)
    private = json.loads(json.dumps(record)); private["environment"]["cpp_compiler"] = "/home/operator/compiler"; mutations.append(private)
    evidence = json.loads(json.dumps(record)); evidence["evidence"]["qualification_sha256"] = "0" * 64; mutations.append(evidence)
    for mutated in mutations:
        with pytest.raises(tool.VerificationError):
            tool._verify_record(mutated, ROOT)


def test_public_record_and_docs_contain_no_private_runtime_inventory() -> None:
    public_paths = (
        RECORD,
        ROOT / "README.md",
        ROOT / "src/poker_knight_ng/README.md",
        ROOT / "validation/holdem/v1/QUALIFICATION.md",
    )
    forbidden = ("/home/", "/tmp/", "desktop-drizzt", "compute_applications", "used_memory_mib", "power_draw_w")
    for path in public_paths:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden)


def test_source_bindings_are_read_from_recorded_commit_not_later_docs() -> None:
    tool = load_tool()
    record = tool.strict_json(RECORD.read_bytes())
    historical = record["source"]["bindings"]["validation/holdem/v1/manifests/rng_seed_bank.sha256"]
    current = tool.sha256(ROOT / "validation/holdem/v1/manifests/rng_seed_bank.sha256")
    assert current != historical
    tool._verify_record(record, ROOT)


def _fake_git(directory: Path, body: str) -> Path:
    executable = directory / "git"
    executable.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body), encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _process_gone(pid: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            return True
        if state == "Z":
            return True
        time.sleep(0.02)
    return False


def test_read_regular_rejects_path_replacement_between_lstat_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    target = tmp_path / "authority"
    replacement = tmp_path / "replacement"
    target.write_bytes(b"original")
    replacement.write_bytes(b"replaced")
    real_open = tool.os.open

    def racing_open(path, flags):
        if Path(path) == target:
            target.unlink()
            replacement.rename(target)
        return real_open(path, flags)

    monkeypatch.setattr(tool.os, "open", racing_open)
    with pytest.raises(tool.VerificationError, match="identity changed"):
        tool.read_regular(target)


def test_git_output_is_killed_at_the_byte_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    marker = tmp_path / "completed"
    _fake_git(tmp_path, """
        import os
        import time
        from pathlib import Path
        for _ in range(32):
            os.write(1, b"x" * 65536)
            time.sleep(0.02)
        Path(os.environ["MARKER"]).write_text("unbounded")
    """)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("MARKER", str(marker))
    monkeypatch.setattr(tool, "MAX_BYTES", 65536)
    monkeypatch.setattr(tool, "GIT_TIMEOUT_SECONDS", 2.0)
    with pytest.raises(tool.VerificationError, match="output limit"):
        tool._git(tmp_path, ["ignored"])
    assert not marker.exists()


def test_git_timeout_kills_the_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    pidfile = tmp_path / "pid"
    _fake_git(tmp_path, """
        import os
        import time
        from pathlib import Path
        Path(os.environ["PIDFILE"]).write_text(str(os.getpid()))
        time.sleep(30)
    """)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PIDFILE", str(pidfile))
    monkeypatch.setattr(tool, "GIT_TIMEOUT_SECONDS", 0.15)
    started = time.monotonic()
    with pytest.raises(tool.VerificationError, match="timed out"):
        tool._git(tmp_path, ["ignored"])
    assert time.monotonic() - started < 2.0
    assert _process_gone(int(pidfile.read_text()))


def test_git_rejects_pipe_inheriting_descendant_and_kills_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    pidfile = tmp_path / "descendant"
    _fake_git(tmp_path, """
        import os
        import subprocess
        import sys
        from pathlib import Path
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        Path(os.environ["PIDFILE"]).write_text(str(child.pid))
    """)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PIDFILE", str(pidfile))
    monkeypatch.setattr(tool, "GIT_TIMEOUT_SECONDS", 1.0)
    with pytest.raises(tool.VerificationError, match="inherited pipes"):
        tool._git(tmp_path, ["ignored"])
    pid = int(pidfile.read_text())
    try:
        assert _process_gone(pid)
    finally:
        if not _process_gone(pid, 0.05):
            os.kill(pid, signal.SIGKILL)


def test_git_propagates_keyboard_interrupt_after_group_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    pidfile = tmp_path / "pid"
    _fake_git(tmp_path, """
        import os
        import time
        from pathlib import Path
        Path(os.environ["PIDFILE"]).write_text(str(os.getpid()))
        time.sleep(30)
    """)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PIDFILE", str(pidfile))

    def interrupt(_process, _timeout):
        deadline = time.monotonic() + 1.0
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        raise KeyboardInterrupt

    monkeypatch.setattr(tool, "_wait_leader", interrupt)
    with pytest.raises(KeyboardInterrupt):
        tool._git(tmp_path, ["ignored"])
    assert _process_gone(int(pidfile.read_text()))
