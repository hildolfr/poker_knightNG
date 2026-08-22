"""Unit tests for tools/conformance_run.py.

These tests exercise the harness logic WITHOUT root, a dedicated identity, or a
live service: syscalls and preconditions are mocked, and JSON schema validity,
sidecar correctness, failure-path exit codes, and remediation text are asserted.
All tests pass unprivileged.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat as stat_module
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import conformance_run as c  # noqa: E402  # pyright: ignore[reportMissingImports]

SCHEMA_PATH = REPO_ROOT / "validation" / "service" / "v1" / "deployment_conformance.schema.json"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def _dir_stat(uid: int, gid: int, mode: int = 0o750) -> os.stat_result:
    return os.stat_result((stat_module.S_IFDIR | mode, 1, 1, 1, uid, gid, 0, 0, 0, 0))


def _file_stat(uid: int, gid: int, mode: int = 0o660) -> os.stat_result:
    return os.stat_result((stat_module.S_IFREG | mode, 2, 1, 1, uid, gid, 0, 0, 0, 0))


class FakeSyscalls:
    """Injectable harness seam. Tune fields to emulate any precondition state."""

    def __init__(self):
        self.euid = 0
        self.sudo = True
        self.user: tuple[str, int] | None = (c.SERVICE_NAME, 1001)
        self.group: tuple[str, int] | None = (c.SERVICE_NAME, 1001)
        self.runtime_stat: os.stat_result | None = _dir_stat(1001, 1001, 0o750)
        self.live = False
        self.unit_text: str | None = (
            "[Unit]\nDescription=x\nRefuseManualStart=yes\n"
            "[Service]\nUser=poker-knight-ng\nGroup=poker-knight-ng\n"
        )  # pre-activation scaffold; the real unit is activated post-WS-3

    def geteuid(self) -> int:
        return self.euid

    def sudo_available(self) -> bool:
        return self.sudo

    def getpwnam(self, name: str):
        if self.user is None:
            raise KeyError(name)
        class _P:
            pw_uid = self.user[1]
        return _P()

    def getgrnam(self, name: str):
        if self.group is None:
            raise KeyError(name)
        class _G:
            gr_gid = self.group[1]
        return _G()

    def stat(self, path: str) -> os.stat_result | None:
        if path != c.PARENT_PATH:
            return None
        return self.runtime_stat

    def read_text(self, path: Path) -> str:
        if self.unit_text is not None:
            return self.unit_text
        return path.read_text(encoding="utf-8")

    def socket_can_connect(self, path: str) -> bool:
        del path
        return self.live


def _passed_preconditions() -> list[c.Precondition]:
    return [
        c.Precondition("privileged", True, "root"),
        c.Precondition("identity_user", True, "user exists"),
        c.Precondition("identity_group", True, "group exists"),
        c.Precondition("runtime_dir", True, "dir ok"),
        c.Precondition("no_live_service", True, "free"),
        c.Precondition("deployment_posture", True, "scaffold inactive"),
    ]


def _sample_checks() -> list[c.CheckResult]:
    return [
        c.CheckResult("healthz_204", True, 1.0, "ok"),
        c.CheckResult("valid_solve_round_trip", True, 2.0, "ok"),
        c.CheckResult("socket_file_mode_owner_after_bind", True, 0.5, "ok"),
        c.CheckResult("seventeenth_connection_rejected", True, 1.0, "ok"),
        c.CheckResult("stale_socket_recovery", True, 3.0, "ok"),
        c.CheckResult("stop_during_idle_closes_promptly", True, 0.2, "ok"),
        c.CheckResult("stop_with_admitted_solve_waits_past_grace", True, 4.0, "ok"),
        c.CheckResult("admission_stopped_during_shutdown", True, 0.3, "ok"),
    ]


# --------------------------------------------------------------------------- #
# Schema validity + sidecar correctness
# --------------------------------------------------------------------------- #
def test_schema_file_is_valid_json() -> None:
    raw = SCHEMA_PATH.read_text(encoding="utf-8")
    schema = json.loads(raw)
    assert schema["$schema"] is not None
    assert schema["type"] == "object"
    assert set(schema["required"]) == {
        "format_version", "schema_url", "timestamp_utc", "host", "service",
        "commit", "preconditions", "checks", "summary",
    }


def _assert_conforms(doc: dict, schema: dict) -> None:
    """Structural validator mirroring the schema's required fields and types."""
    for required in schema["required"]:
        assert required in doc, f"missing required field {required}"
    props = schema["properties"]
    assert doc["format_version"] == props["format_version"]["const"]
    assert len(doc["commit"]["rev_parse_head"]) == 40
    assert set(doc["commit"]["rev_parse_head"]).issubset("0123456789abcdef")
    assert doc["schema_url"] == props["schema_url"]["pattern"].replace("\\", "").replace("^", "").replace("$", "")
    assert doc["timestamp_utc"].endswith("Z")
    # host / service / commit nested required fields
    for obj, want in (
        (doc["host"], {"kernel", "effective_uid", "effective_gid"}),
        (doc["service"], {"name", "uid", "gid"}),
        (doc["commit"], {"rev_parse_head", "branch"}),
    ):
        assert want.issubset(obj.keys())
    assert doc["service"]["name"] == "poker-knight-ng"
    assert isinstance(doc["service"]["uid"], int) and isinstance(doc["service"]["gid"], int)
    assert doc["preconditions"]["passed"] is False or doc["preconditions"]["passed"] is True
    for check in doc["preconditions"]["checks"]:
        assert set(check.keys()) == {"name", "passed", "detail"}
    for check in doc["checks"]:
        assert set(check.keys()) == {"name", "passed", "duration_ms", "detail"}
        assert isinstance(check["duration_ms"], (int, float)) and check["duration_ms"] >= 0
    assert set(doc["summary"].keys()) == {"passed", "total", "passed_count", "failed_count"}
    assert doc["summary"]["total"] == len(doc["checks"])
    assert doc["summary"]["passed_count"] + doc["summary"]["failed_count"] == doc["summary"]["total"]
    # summary.passed is gated on preconditions AND zero failed checks.
    assert doc["summary"]["passed"] == (
        doc["preconditions"]["passed"] and doc["summary"]["failed_count"] == 0
    )


def test_write_evidence_produces_schema_conformant_json_and_correct_sidecar(tmp_path) -> None:
    evidence, sidecar, digest = c.write_evidence(
        _passed_preconditions(), _sample_checks(), REPO_ROOT, 1001, 1001,
        output_dir=tmp_path,
    )
    assert evidence.name == "deployment_conformance.json"
    assert sidecar.name == "deployment_conformance.json.sha256"

    doc = json.loads(evidence.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    _assert_conforms(doc, schema)

    # Summary reflects the sample checks (all pass).
    assert doc["summary"]["passed"] is True
    assert doc["summary"]["total"] == len(_sample_checks())
    assert doc["summary"]["passed_count"] == len(_sample_checks())

    # Sidecar must be sha256 of the evidence body, in '<hex>  <name>' format.
    expected = hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert digest == expected
    text = sidecar.read_text(encoding="utf-8")
    fields = text.split()
    assert fields == [expected, evidence.name]


def test_write_evidence_failed_preconditions_still_writes_conformant_evidence(tmp_path) -> None:
    failed = [
        c.Precondition("privileged", False, "not root"),
        c.Precondition("identity_user", False, "no user. Remediation: useradd ..."),
    ]
    evidence, _sidecar, _digest = c.write_evidence(failed, [], REPO_ROOT, 0, 0, output_dir=tmp_path)
    doc = json.loads(evidence.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    _assert_conforms(doc, schema)
    assert doc["summary"]["passed"] is False
    assert doc["preconditions"]["passed"] is False


# --------------------------------------------------------------------------- #
# Preconditions + remediation text
# --------------------------------------------------------------------------- #
def test_preconditions_pass_when_deployment_model_is_provisioned() -> None:
    checks = c.precondition_checks(FakeSyscalls())
    names = {p.name: p for p in checks}
    assert names["privileged"].passed is True
    assert names["identity_user"].passed is True
    assert names["identity_group"].passed is True
    assert names["runtime_dir"].passed is True
    assert names["no_live_service"].passed is True
    assert names["deployment_posture"].passed is True
    assert all(p.passed for p in checks)


def test_precondition_failures_produce_remediation_text() -> None:
    fake = FakeSyscalls()
    fake.user = None
    fake.group = None
    fake.runtime_stat = None
    fake.live = True
    checks = c.precondition_checks(fake)
    by_name = {p.name: p for p in checks}

    assert by_name["identity_user"].passed is False
    assert "useradd" in by_name["identity_user"].detail
    assert by_name["identity_group"].passed is False
    assert "groupadd" in by_name["identity_group"].detail
    assert by_name["runtime_dir"].passed is False
    assert "tmpfiles" in by_name["runtime_dir"].detail
    assert by_name["no_live_service"].passed is False
    assert "stop the running service" in by_name["no_live_service"].detail
    # All failures must carry operator-facing remediation text.
    for name in ("identity_user", "identity_group", "runtime_dir", "no_live_service"):
        assert "Remediation" in by_name[name].detail


def test_unprivileged_without_sudo_fails_privileged_precondition() -> None:
    fake = FakeSyscalls()
    fake.euid = 1000
    fake.sudo = False
    checks = c.precondition_checks(fake)
    priv = {p.name: p for p in checks}["privileged"]
    assert priv.passed is False
    assert "root" in priv.detail and "sudo" in priv.detail


def test_deployment_posture_refuses_when_refusemanualstart_removed() -> None:
    fake = FakeSyscalls()
    # A unit that no longer declares RefuseManualStart=yes (activation implied).
    fake.unit_text = (
        "[Unit]\nDescription=x\n[Service]\nUser=poker-knight-ng\n"
        "Group=poker-knight-ng\nRuntimeDirectory=poker-knight-ng\n"
        "RuntimeDirectoryMode=0750\nRestrictAddressFamilies=AF_UNIX\n"
    )
    checks = c.precondition_checks(fake)
    posture = {p.name: p for p in checks}["deployment_posture"]
    assert posture.passed is False
    assert "RefuseManualStart" in posture.detail
    assert "Refuse" in posture.detail


def test_deployment_posture_refuses_when_install_added() -> None:
    fake = FakeSyscalls()
    fake.unit_text = (
        "[Unit]\nDescription=x\nRefuseManualStart=yes\n"
        "[Service]\nUser=poker-knight-ng\nGroup=poker-knight-ng\n"
        "[Install]\nWantedBy=multi-user.target\n"
    )
    checks = c.precondition_checks(fake)
    posture = {p.name: p for p in checks}["deployment_posture"]
    assert posture.passed is False
    assert "[Install]" in posture.detail


# --------------------------------------------------------------------------- #
# Failure paths exit nonzero
# --------------------------------------------------------------------------- #
def test_exit_code_zero_only_when_all_pass() -> None:
    assert c._exit_code(_passed_preconditions(), _sample_checks()) == 0
    assert c._exit_code(_passed_preconditions(), [c.CheckResult("x", False, 0.0, "fail")]) == 1
    failed_pre = [c.Precondition("privileged", False, "not root")]
    assert c._exit_code(failed_pre, _sample_checks()) == 1


def test_main_returns_nonzero_when_preconditions_fail(tmp_path, monkeypatch) -> None:
    # Point the harness's output dir at tmp so the real repo isn't written to,
    # and force a precondition failure via a no-op identity environment check.
    monkeypatch.setattr(c, "_repo_root", lambda: tmp_path)
    # Without a poker-knight-ng identity the real Syscalls fails preconditions on
    # any unprivileged host; assert nonzero (the guard blocks the run).
    code = c.main(["--dry-run"])
    assert code == 1
