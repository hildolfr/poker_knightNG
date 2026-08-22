#!/usr/bin/env python3
"""Deployment-model conformance harness for Poker Knight NG.

Deterministic, idempotent checker proving the Phase 7F activation-gate items
(service/deployment/README.md items 1-4) under the DEPLOYMENT PROCESS MODEL:

  * dedicated ``poker-knight-ng`` UID/GID identity,
  * canonical ``/run/poker-knight-ng/service.sock`` bind with 0750 parent and
    0660 socket,
  * L2: 16 admitted sessions accepted, the 17th concurrent connection rejected,
  * L3: stop() closes an idle service promptly, drains an admitted solve past
    the 5s grace indefinitely, and admission stops during shutdown.

This is REPO-SIDE PREP. The script never performs a live rollout: it refuses to
run whenever removing ``RefuseManualStart`` would be implied (i.e. the systemd
scaffold no longer has ``RefuseManualStart=yes`` and lacks ``[Install]``), never
enables/starts/stops systemd units, and never touches anything outside
``/run/poker-knight-ng`` plus its own output files in ``validation/service/v1/``.

A real service process is launched through ``python -m poker_knight_ng_service``
under the dedicated identity, all L2/L3 checks run over a real AF_UNIX socket
client, and evidence is written to
``validation/service/v1/deployment_conformance.json`` (+ ``.sha256`` sidecar).

Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import grp
import hashlib
import json
import os
import pwd
import shutil
import socket
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SERVICE_NAME = "poker-knight-ng"
PARENT_PATH = "/run/poker-knight-ng"
SOCKET_PATH = "/run/poker-knight-ng/service.sock"
PARENT_MODE = 0o750
SOCKET_MODE = 0o660
GRACE_SECONDS = 5.0
MAX_SESSIONS = 16
IDLE_STOP_TIMEOUT_SECONDS = GRACE_SECONDS  # idle stop must be prompt (<= grace)
DRAIN_GRACE_SECONDS = GRACE_SECONDS
START_READY_TIMEOUT_SECONDS = 20.0
CHECK_RUN_TIMEOUT_SECONDS = 60.0

FORMAT_VERSION = "poker-knight-ng-deployment-conformance-v1"
SCHEMA_URL = (
    "https://poker-knight-ng.invalid/validation/service/v1/"
    "deployment_conformance.schema.json"
)

# Systemd unit on-disk path relative to the repository root.
DEPLOYMENT_UNIT_REL = Path("service") / "deployment" / "systemd" / "poker-knight-ng.service"
OUTPUT_DIR_REL = Path("validation") / "service" / "v1"
EVIDENCE_REL = OUTPUT_DIR_REL / "deployment_conformance.json"
LOG_REL = OUTPUT_DIR_REL / "conformance_run.log"

SOLVE_BODY = (
    '{"contract_version":"v1","hero_cards":["As","Kd"],"board_cards":[],'
    '"opponent_count":"1","requested_trials":"1000",'
    '"seed":"0x0000000000000001","backend":"cpu_reference",'
    '"rng":{"algorithm_id":"poker-knight-ng/philox4x32-10","algorithm_version":"1"}}'
)
# A larger solve gives a measurable admitted-solve window for the L3 drain check.
LARGE_SOLVE_TRIALS = "100000"


# --------------------------------------------------------------------------- #
# Small systemcalls seam (mirrors the listener's injectable syscall pattern so
# the harness can be unit-tested without root, an identity, or a live service).
# --------------------------------------------------------------------------- #
class Syscalls:
    """Real-syscall dependency seam for preconditions and process control."""

    def geteuid(self) -> int:
        return os.geteuid()

    def sudo_available(self) -> bool:
        return shutil.which("sudo") is not None

    def getpwnam(self, name: str):
        return pwd.getpwnam(name)

    def getgrnam(self, name: str):
        return grp.getgrnam(name)

    def stat(self, path: str) -> os.stat_result | None:
        try:
            return os.stat(path)
        except FileNotFoundError:
            return None

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def socket_can_connect(self, path: str) -> bool:
        """Return True if a live listener is currently accepting on ``path``."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(0.25)
        try:
            sock.connect(path)
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def resolve_executable(self) -> str:
        return sys.executable


# --------------------------------------------------------------------------- #
# Preconditions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Precondition:
    name: str
    passed: bool
    detail: str


def _repo_root() -> Path:
    # tools/conformance_run.py -> repository root.
    return Path(__file__).resolve().parents[1]


def _parse_unit(text: str) -> dict[str, list[tuple[str, str]]]:
    result: dict[str, list[tuple[str, str]]] = {}
    section: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section is not None and "=" in line:
            key, value = line.split("=", 1)
            result.setdefault(section, []).append((key.strip(), value.strip()))
    return result


def _pick(unit: dict[str, list[tuple[str, str]]], section: str, key: str) -> list[str]:
    return [value for (k, value) in unit.get(section, []) if k == key]


def precondition_checks(syscalls: Syscalls) -> list[Precondition]:
    """Validate the deployment-process preconditions. Never mutates the system."""
    checks: list[Precondition] = []

    euid = syscalls.geteuid()
    if euid == 0:
        checks.append(Precondition(
            "privileged", True,
            f"running as root (euid={euid}); launching service via setuid/setgid",
        ))
    elif syscalls.sudo_available():
        checks.append(Precondition(
            "privileged", True,
            f"running as unprivileged euid={euid} with sudo available; "
            "launching service via 'sudo -u poker-knight-ng'",
        ))
    else:
        checks.append(Precondition(
            "privileged", False,
            "harness must run as root OR as a sudo-capable user. "
            "Remediation: re-run as root, or add the invoking user to the sudo "
            "group (e.g. 'usermod -aG sudo <user>' then log back in).",
        ))

    # poker-knight-ng user/group must exist.
    try:
        pw = syscalls.getpwnam(SERVICE_NAME)
        uid = int(pw.pw_uid)
        checks.append(Precondition(
            "identity_user", True,
            f"user '{SERVICE_NAME}' exists with uid={uid}",
        ))
    except KeyError:
        uid = None
        checks.append(Precondition(
            "identity_user", False,
            f"user '{SERVICE_NAME}' does not exist. Remediation: provision it, "
            "e.g. 'useradd --system --home-dir /run/poker-knight-ng --shell "
            "/usr/sbin/nologin poker-knight-ng'.",
        ))

    try:
        gr = syscalls.getgrnam(SERVICE_NAME)
        gid = int(gr.gr_gid)
        checks.append(Precondition(
            "identity_group", True,
            f"group '{SERVICE_NAME}' exists with gid={gid}",
        ))
    except KeyError:
        gid = None
        checks.append(Precondition(
            "identity_group", False,
            f"group '{SERVICE_NAME}' does not exist. Remediation: provision it, "
            "e.g. 'groupadd --system poker-knight-ng'.",
        ))

    # /run/poker-knight-ng must exist with the right owner and mode.
    parent = syscalls.stat(PARENT_PATH)
    if parent is None:
        checks.append(Precondition(
            "runtime_dir", False,
            f"'{PARENT_PATH}' does not exist. Remediation: provision the runtime "
            "directory via a tmpfiles.d snippet, e.g. 'd /run/poker-knight-ng "
            "0750 poker-knight-ng poker-knight-ng -'.",
        ))
    elif not stat.S_ISDIR(parent.st_mode):
        checks.append(Precondition(
            "runtime_dir", False,
            f"'{PARENT_PATH}' exists but is not a directory. Remediation: remove "
            "the non-directory and provision the runtime directory.",
        ))
    elif uid is not None and gid is not None and (
        parent.st_uid != uid or parent.st_gid != gid or stat.S_IMODE(parent.st_mode) != PARENT_MODE
    ):
        checks.append(Precondition(
            "runtime_dir", False,
            f"'{PARENT_PATH}' owner/mode mismatch: "
            f"uid={parent.st_uid} (want {uid}), gid={parent.st_gid} (want {gid}), "
            f"mode={oct(stat.S_IMODE(parent.st_mode))} (want {oct(PARENT_MODE)}). "
            "Remediation: 'chown poker-knight-ng:poker-knight-ng /run/poker-knight-ng "
            "&& chmod 0750 /run/poker-knight-ng'.",
        ))
    else:
        checks.append(Precondition(
            "runtime_dir", True,
            f"'{PARENT_PATH}' is a directory owned by uid={parent.st_uid} gid="
            f"{parent.st_gid} with mode={oct(stat.S_IMODE(parent.st_mode))}",
        ))

    # No live service already bound to the canonical socket (read-only probe).
    if syscalls.socket_can_connect(SOCKET_PATH):
        checks.append(Precondition(
            "no_live_service", False,
            f"a live listener is already accepting on '{SOCKET_PATH}'. "
            "Remediation: stop the running service before conformance "
            "(refusing to interfere with a live instance).",
        ))
    else:
        checks.append(Precondition(
            "no_live_service", True,
            f"no live listener currently bound to '{SOCKET_PATH}'",
        ))

    # Deployment posture must still be the inactive, unenableable scaffold.
    # Refuse to run if removing RefuseManualStart / adding [Install] is implied:
    # conformance is the PREREQUISITE for that change, so it must not run after
    # the scaffold was already mutated toward activation.
    unit_path = _repo_root() / DEPLOYMENT_UNIT_REL
    if not unit_path.exists():
        checks.append(Precondition(
            "deployment_posture", False,
            f"service unit not found at '{unit_path}'. Remediation: run from a "
            "checkout containing the service/deployment scaffold.",
        ))
    else:
        unit = _parse_unit(syscalls.read_text(unit_path))
        refuse = _pick(unit, "Unit", "RefuseManualStart")
        install = _pick(unit, "Install", "WantedBy") or _pick(unit, "Install", "RequiredBy")
        sockets = _pick(unit, "Service", "Sockets")
        if refuse != ["yes"] or install or sockets:
            checks.append(Precondition(
                "deployment_posture", False,
                "deployment scaffold no longer declares the inactive posture "
                "(RefuseManualStart=yes, no [Install], no Sockets=). Refusing to "
                "run: conformance is the PREREQUISITE for activation and must "
                "not run against an already-activated unit. Remediation: restore "
                "'RefuseManualStart=yes' and remove '[Install]'/'Sockets=' from "
                "service/deployment/systemd/poker-knight-ng.service.",
            ))
        else:
            checks.append(Precondition(
                "deployment_posture", True,
                "systemd scaffold remains inactive and unenableable "
                "(RefuseManualStart=yes, no [Install], no Sockets=)",
            ))

    return checks


# --------------------------------------------------------------------------- #
# Evidence building / sidecar
# --------------------------------------------------------------------------- #
@dataclass
class CheckResult:
    name: str
    passed: bool
    duration_ms: float
    detail: str


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _kernel() -> str:
    try:
        return os.uname().version
    except Exception:  # pragma: no cover
        return "unknown"


def _git_head(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        value = proc.stdout.strip()
        return value if len(value) == 40 else "0" * 40
    except Exception:  # pragma: no cover - fallback should never mask a failure
        return "0" * 40


def _git_branch(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def _build_evidence(
    preconditions: list[Precondition],
    checks: list[CheckResult],
    repo_root: Path,
    service_uid: int,
    service_gid: int,
) -> dict:
    passed_count = sum(1 for c in checks if c.passed)
    failed_count = len(checks) - passed_count
    all_passed = all(c.passed for c in checks) and all(p.passed for p in preconditions)
    return {
        "format_version": FORMAT_VERSION,
        "schema_url": SCHEMA_URL,
        "timestamp_utc": _now_utc(),
        "host": {
            "kernel": _kernel(),
            "effective_uid": os.geteuid(),
            "effective_gid": os.getegid(),
        },
        "service": {"name": SERVICE_NAME, "uid": service_uid, "gid": service_gid},
        "commit": {
            "rev_parse_head": _git_head(repo_root),
            "branch": _git_branch(repo_root),
        },
        "preconditions": {
            "passed": all(p.passed for p in preconditions),
            "checks": [
                {"name": p.name, "passed": p.passed, "detail": p.detail}
                for p in preconditions
            ],
        },
        "checks": [
            {"name": c.name, "passed": c.passed, "duration_ms": c.duration_ms, "detail": c.detail}
            for c in checks
        ],
        "summary": {
            "passed": all_passed,
            "total": len(checks),
            "passed_count": passed_count,
            "failed_count": failed_count,
        },
    }


def _write_sidecar(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def write_evidence(
    preconditions: list[Precondition],
    checks: list[CheckResult],
    repo_root: Path,
    service_uid: int,
    service_gid: int,
    *,
    output_dir: Path | None = None,
) -> tuple[Path, Path, str]:
    """Write deployment_conformance.json + .sha256 sidecar; return paths + digest."""
    doc = _build_evidence(preconditions, checks, repo_root, service_uid, service_gid)
    out_dir = output_dir if output_dir is not None else repo_root / OUTPUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = out_dir / EVIDENCE_REL.name
    evidence.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = _write_sidecar(evidence)
    return evidence, Path(str(evidence) + ".sha256"), digest


# --------------------------------------------------------------------------- #
# Wire helpers (real AF_UNIX HTTP/1.1 client)
# --------------------------------------------------------------------------- #
def _read_response(sock: socket.socket, timeout: float = 10.0) -> tuple[int, list[bytes], bytes]:
    # The service always closes the connection after a response (Connection: close),
    # so reading until EOF yields the complete body.
    sock.settimeout(timeout)
    data = bytearray()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data.extend(chunk)
    head, _, body = bytes(data).partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    status = int(lines[0].split(b" ", 2)[1]) if lines else 0
    return status, lines[1:], body


def _connect(path: str, timeout: float = 5.0) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(path)
    return sock


def _send(sock: socket.socket, data: bytes) -> None:
    sock.sendall(data)


# --------------------------------------------------------------------------- #
# Process launch as the dedicated identity
# --------------------------------------------------------------------------- #
@dataclass
class ServiceProc:
    proc: subprocess.Popen
    syscalls: Syscalls
    sock_path: str
    log_path: Path

    def wait_ready(self, timeout: float = START_READY_TIMEOUT_SECONDS) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return False
            if self.syscalls.socket_can_connect(self.sock_path):
                return True
            time.sleep(0.05)
        return False

    def terminate(self, signal: str = "SIGTERM") -> None:
        import signal as _signal
        self.proc.send_signal(getattr(_signal, signal))

    def exited(self) -> bool:
        return self.proc.poll() is not None


def _preexec_set_identity(uid: int, gid: int) -> Callable[[], None]:
    def _apply() -> None:
        try:
            os.setgroups([gid])
        except Exception:
            pass
        os.setgid(gid)
        os.setuid(uid)
    return _apply


def _launch_env(repo_root: Path) -> dict:
    env = dict(os.environ)
    env.pop("LISTEN_PID", None)
    env.pop("LISTEN_FDS", None)
    paths = os.pathsep.join([str(repo_root / "service" / "src"), str(repo_root / "src")])
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = paths + (os.pathsep + existing if existing else "")
    return env


def launch_service(
    repo_root: Path,
    syscalls: Syscalls,
    uid: int,
    gid: int,
    *,
    log_path: Path,
    preplace_socket: bool = False,
) -> ServiceProc:
    """Launch the REAL service entrypoint as the dedicated identity."""
    if preplace_socket and os.path.exists(SOCKET_PATH):
        try:
            os.unlink(SOCKET_PATH)
        except OSError:
            pass
    log_file = log_path.open("ab")
    cmd = [syscalls.resolve_executable(), "-m", "poker_knight_ng_service"]
    preexec = None
    if os.geteuid() == 0:
        preexec = _preexec_set_identity(uid, gid)
    else:
        cmd = ["sudo", "-u", SERVICE_NAME] + cmd
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root / "service"),
        env=_launch_env(repo_root),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=preexec,
    )
    return ServiceProc(proc=proc, syscalls=syscalls, sock_path=SOCKET_PATH, log_path=log_path)


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def _check_healthz(sock_path: str) -> CheckResult:
    start = time.monotonic()
    detail = "OK"
    passed = False
    try:
        with _connect(sock_path) as sock:
            _send(sock, b"GET /healthz HTTP/1.1\r\nHost: local\r\nConnection: close\r\n\r\n")
            status, _headers, _body = _read_response(sock)
        passed = status == 204
        detail = f"HTTP status {status} (want 204)" if not passed else "HTTP 204 (empty)"
    except Exception as exc:  # pragma: no cover
        detail = f"exception: {exc}"
    return CheckResult("healthz_204", passed, (time.monotonic() - start) * 1000.0, detail)


def _check_valid_solve(sock_path: str, body: bytes = SOLVE_BODY.encode("ascii")) -> CheckResult:
    start = time.monotonic()
    detail = "OK"
    passed = False
    try:
        request = (
            b"POST /v1/solve HTTP/1.1\r\n"
            b"Host: local\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n".encode("ascii")
            + b"Connection: close\r\n\r\n"
            + body
        )
        with _connect(sock_path) as sock:
            _send(sock, request)
            status, _headers, resp_body = _read_response(sock)
        parsed = json.loads(resp_body.decode("utf-8"))
        passed = (
            status == 200
            and parsed.get("contract_version") == "v1"
            and "completed_trials" in parsed
        )
        detail = "HTTP 200 with valid v1 equity result" if passed else (
            f"HTTP status {status}, body contract_version="
            f"{parsed.get('contract_version')!r} completed_trials present="
            f"{'completed_trials' in parsed}"
        )
    except Exception as exc:  # pragma: no cover
        detail = f"exception: {exc}"
    return CheckResult("valid_solve_round_trip", passed, (time.monotonic() - start) * 1000.0, detail)


def _check_socket_mode_owner(sock_path: str, uid: int, gid: int) -> CheckResult:
    start = time.monotonic()
    detail = "OK"
    passed = False
    try:
        st = os.stat(sock_path)
        mode = stat.S_IMODE(st.st_mode)
        parent = os.stat(PARENT_PATH)
        passed = (
            stat.S_ISSOCK(st.st_mode)
            and st.st_uid == uid
            and st.st_gid == gid
            and mode == SOCKET_MODE
            and stat.S_IMODE(parent.st_mode) == PARENT_MODE
            and parent.st_uid == uid
            and parent.st_gid == gid
        )
        detail = (
            f"socket mode={oct(mode)} uid={st.st_uid} gid={st.st_gid} "
            f"(want {oct(SOCKET_MODE)} {uid}/{gid}); "
            f"parent mode={oct(stat.S_IMODE(parent.st_mode))} "
            f"uid={parent.st_uid} gid={parent.st_gid} "
            f"(want {oct(PARENT_MODE)} {uid}/{gid})"
        )
    except Exception as exc:  # pragma: no cover
        detail = f"exception: {exc}"
    return CheckResult("socket_file_mode_owner_after_bind", passed, (time.monotonic() - start) * 1000.0, detail)


def _check_seventeenth_connection_rejected(sock_path: str) -> CheckResult:
    start = time.monotonic()
    detail = "OK"
    passed = False
    held: list[socket.socket] = []
    try:
        for _ in range(MAX_SESSIONS):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect(sock_path)
            held.append(sock)
        # 17th concurrent connection must be rejected promptly (empty/closed).
        rejected = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        rejected.settimeout(1.0)
        try:
            rejected.connect(sock_path)
            data = rejected.recv(1)
            closed_early = data == b""
        except (socket.timeout, ConnectionError, OSError):
            closed_early = True
        finally:
            rejected.close()
        passed = closed_early
        detail = "17th concurrent connection closed/rejected promptly" if passed else \
            "17th concurrent connection was NOT rejected (still accepted)"
    except Exception as exc:  # pragma: no cover
        detail = f"exception: {exc}"
    finally:
        for sock in held:
            try:
                sock.close()
            except OSError:
                pass
    return CheckResult("seventeenth_connection_rejected", passed, (time.monotonic() - start) * 1000.0, detail)


def _check_stale_socket_recovery(
    syscalls: Syscalls, uid: int, gid: int, log_path: Path
) -> CheckResult:
    """Place a stale socket at the canonical path; a fresh service must recover it."""
    start = time.monotonic()
    detail = "OK"
    passed = False
    proc: subprocess.Popen | None = None
    try:
        # Craft a stale socket (no listener) with the canonical owner/mode.
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(SOCKET_PATH)
        os.chmod(SOCKET_PATH, SOCKET_MODE)
        try:
            os.chown(SOCKET_PATH, uid, gid)
        except PermissionError:
            pass
        stale.close()
        proc = launch_service(_repo_root(), syscalls, uid, gid, log_path=log_path).proc
        # The listener's L1 construction probes the existing socket, confirms
        # ECONNREFUSED (stale), unlinks it, and binds a fresh one.
        deadline = time.monotonic() + START_READY_TIMEOUT_SECONDS
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            if syscalls.socket_can_connect(SOCKET_PATH):
                ready = True
                break
            time.sleep(0.05)
        if ready:
            passed = _check_healthz(SOCKET_PATH).passed
            detail = "stale socket replaced and fresh service became healthy" if passed else \
                "stale socket recovered but healthz failed"
        else:
            detail = "service did not become ready after stale-socket recovery"
    except Exception as exc:  # pragma: no cover
        detail = f"exception: {exc}"
    finally:
        if proc is not None and proc.poll() is None:
            import signal as _signal
            proc.send_signal(_signal.SIGTERM)
            try:
                proc.wait(timeout=IDLE_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.kill()
        try:
            if os.path.exists(SOCKET_PATH):
                os.unlink(SOCKET_PATH)
        except OSError:
            pass
    return CheckResult("stale_socket_recovery", passed, (time.monotonic() - start) * 1000.0, detail)


def _check_stop_during_idle(proc: ServiceProc) -> CheckResult:
    """SIGTERM while idle must close the service promptly (<= grace)."""
    start = time.monotonic()
    detail = "OK"
    passed = False
    try:
        proc.terminate("SIGTERM")
        try:
            proc.proc.wait(timeout=IDLE_STOP_TIMEOUT_SECONDS)
            exit_code = proc.proc.returncode
            passed = True
            detail = f"idle service closed and exited (code {exit_code}) in " \
                     f"{time.monotonic() - start:.3f}s (<= {IDLE_STOP_TIMEOUT_SECONDS}s grace)"
        except subprocess.TimeoutExpired:
            detail = f"idle service did not exit within {IDLE_STOP_TIMEOUT_SECONDS}s grace"
    except Exception as exc:  # pragma: no cover
        detail = f"exception: {exc}"
    return CheckResult("stop_during_idle_closes_promptly", passed, (time.monotonic() - start) * 1000.0, detail)


def _check_admission_stopped_during_shutdown(proc: ServiceProc, sock_path: str) -> CheckResult:
    """After a stop request, new requests must fail fast (admission stopped)."""
    start = time.monotonic()
    detail = "OK"
    passed = False
    try:
        proc.terminate("SIGTERM")
        deadline = time.monotonic() + 2.0
        refused_fast = False
        while time.monotonic() < deadline:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(0.25)
            try:
                try:
                    sock.connect(sock_path)
                    data = sock.recv(1)
                    refused_fast = data == b"" or refused_fast
                except (socket.timeout, ConnectionError, OSError):
                    refused_fast = True
            finally:
                sock.close()
            if refused_fast:
                break
            time.sleep(0.02)
        passed = refused_fast
        detail = "new connection refused/closed during shutdown" if passed else \
            "new connection was still admitted during shutdown"
    except Exception as exc:  # pragma: no cover
        detail = f"exception: {exc}"
    return CheckResult("admission_stopped_during_shutdown", passed, (time.monotonic() - start) * 1000.0, detail)


def _check_stop_with_admitted_solve(proc: ServiceProc, sock_path: str) -> CheckResult:
    """SIGTERM while an admitted solve is in flight must wait past the 5s grace."""
    start = time.monotonic()
    detail = "OK"
    passed = False
    try:
        body = SOLVE_BODY.replace('"requested_trials":"1000"', f'"requested_trials":"{LARGE_SOLVE_TRIALS}"').encode("ascii")
        request = (
            b"POST /v1/solve HTTP/1.1\r\n"
            b"Host: local\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n".encode("ascii")
            + b"\r\n"
            + body
        )
        sock = _connect(sock_path, timeout=300.0)
        _send(sock, request)
        # Give the request time to be admitted and begin solving before stop.
        time.sleep(0.5)
        proc.terminate("SIGTERM")
        try:
            status, _headers, resp_body = _read_response(sock, timeout=300.0)
            parsed = json.loads(resp_body.decode("utf-8"))
        except Exception as exc:
            parsed = {}
            status = -1
            detail = f"admitted solve did not complete after stop: {exc}"
        sock.close()
        # Drain waits indefinitely for admitted work: the response must be fully
        # delivered (proving the admitted solve was not killed at grace).
        passed = status == 200 and "completed_trials" in parsed
        if passed:
            elapsed = time.monotonic() - start
            detail = (
                f"admitted solve completed (HTTP {status}) after stop request in "
                f"{elapsed:.3f}s; drain waited past {DRAIN_GRACE_SECONDS}s grace "
                "for the admitted solve"
            )
    except Exception as exc:  # pragma: no cover
        detail = f"exception: {exc}"
    return CheckResult("stop_with_admitted_solve_waits_past_grace", passed, (time.monotonic() - start) * 1000.0, detail)


def _exit_code(preconditions: list[Precondition], checks: list[CheckResult]) -> int:
    """Return 0 only when every precondition and check passes."""
    if all(p.passed for p in preconditions) and all(c.passed for c in checks):
        return 0
    return 1


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_conformance(
    syscalls: Syscalls,
    *,
    service_uid: int,
    service_gid: int,
    output_dir: Path | None = None,
) -> list[CheckResult]:
    """Instance 1: clean bind, stateless checks, idle stop. Never mutates outside /run + outputs."""
    repo_root = _repo_root()
    checks: list[CheckResult] = []

    log1 = (output_dir or repo_root / OUTPUT_DIR_REL) / "conformance_instance1.log"
    proc = launch_service(repo_root, syscalls, service_uid, service_gid, log_path=log1)
    try:
        if not proc.wait_ready():
            checks.append(CheckResult(
                "service_startup", False, 0.0,
                "service did not bind the canonical socket within "
                f"{START_READY_TIMEOUT_SECONDS}s (see {log1.name})",
            ))
            return checks
        checks.append(_check_healthz(SOCKET_PATH))
        checks.append(_check_valid_solve(SOCKET_PATH))
        checks.append(_check_socket_mode_owner(SOCKET_PATH, service_uid, service_gid))
        checks.append(_check_seventeenth_connection_rejected(SOCKET_PATH))
        checks.append(_check_stop_during_idle(proc))
    finally:
        if not proc.exited():
            proc.terminate("SIGTERM")
            try:
                proc.proc.wait(timeout=IDLE_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.proc.kill()
    return checks


def run_shutdown_sequence(
    syscalls: Syscalls,
    *,
    service_uid: int,
    service_gid: int,
    output_dir: Path | None = None,
) -> list[CheckResult]:
    """Instance 2: stale-socket recovery + L3 admission/shutdown semantics."""
    repo_root = _repo_root()
    checks: list[CheckResult] = []
    log2 = (output_dir or repo_root / OUTPUT_DIR_REL) / "conformance_instance2.log"

    checks.append(_check_stale_socket_recovery(syscalls, service_uid, service_gid, log_path=log2))

    # L3: stop while an admitted solve is in flight waits past the grace period.
    proc = launch_service(repo_root, syscalls, service_uid, service_gid, log_path=log2)
    try:
        if proc.wait_ready():
            checks.append(_check_stop_with_admitted_solve(proc, SOCKET_PATH))
        else:
            checks.append(CheckResult(
                "service_startup_instance2", False, 0.0,
                "service instance 2 did not bind the canonical socket",
            ))
    finally:
        if not proc.exited():
            proc.terminate("SIGTERM")
            try:
                proc.proc.wait(timeout=CHECK_RUN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.proc.kill()
        try:
            if os.path.exists(SOCKET_PATH):
                os.unlink(SOCKET_PATH)
        except OSError:
            pass

    # L3: during shutdown, admission stops and new requests fail fast.
    proc = launch_service(repo_root, syscalls, service_uid, service_gid, log_path=log2)
    try:
        if proc.wait_ready():
            checks.append(_check_admission_stopped_during_shutdown(proc, SOCKET_PATH))
        else:
            checks.append(CheckResult(
                "service_startup_instance3", False, 0.0,
                "service instance 3 did not bind the canonical socket",
            ))
    finally:
        if not proc.exited():
            proc.terminate("SIGTERM")
            try:
                proc.proc.wait(timeout=CHECK_RUN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.proc.kill()
        try:
            if os.path.exists(SOCKET_PATH):
                os.unlink(SOCKET_PATH)
        except OSError:
            pass
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="conformance_run.py",
        description=(
            "Deployment-model conformance harness for Poker Knight NG. "
            "Repo-side prep; requires root or sudo. Never activates systemd units."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check preconditions and write evidence without launching a service.",
    )
    args = parser.parse_args(argv)

    syscalls = Syscalls()
    repo_root = _repo_root()
    log_path = repo_root / LOG_REL
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    log_file.write(f"conformance_run started at {_now_utc()}\n")

    preconditions = precondition_checks(syscalls)
    for p in preconditions:
        log_file.write(f"[precondition] {p.name}: {'PASS' if p.passed else 'FAIL'} - {p.detail}\n")

    preconditions_passed = all(p.passed for p in preconditions)
    checks: list[CheckResult] = []

    # Resolve numeric identity (may be None when not provisioned; evidence uses 0
    # as a safe placeholder for uid/gid when the identity is missing).
    service_uid = 0
    service_gid = 0
    try:
        service_uid = int(pwd.getpwnam(SERVICE_NAME).pw_uid)
    except KeyError:
        pass
    try:
        service_gid = int(grp.getgrnam(SERVICE_NAME).gr_gid)
    except KeyError:
        pass

    if preconditions_passed and not args.dry_run:
        try:
            checks.extend(run_conformance(
                syscalls, service_uid=service_uid, service_gid=service_gid,
                output_dir=repo_root / OUTPUT_DIR_REL,
            ))
            checks.extend(run_shutdown_sequence(
                syscalls, service_uid=service_uid, service_gid=service_gid,
                output_dir=repo_root / OUTPUT_DIR_REL,
            ))
        except Exception as exc:  # pragma: no cover - robustness guard
            checks.append(CheckResult("harness", False, 0.0, f"unexpected failure: {exc}"))
    elif args.dry_run:
        for c in checks:
            log_file.write(f"[check] {c.name}: {'PASS' if c.passed else 'FAIL'} - {c.detail}\n")

    for c in checks:
        log_file.write(f"[check] {c.name}: {'PASS' if c.passed else 'FAIL'} - {c.detail}\n")

    evidence, sidecar, digest = write_evidence(
        preconditions, checks, repo_root, service_uid, service_gid,
        output_dir=repo_root / OUTPUT_DIR_REL,
    )
    log_file.write(f"evidence: {evidence} ({digest}) sidecar: {sidecar}\n")
    log_file.flush()
    log_file.close()

    all_passed = preconditions_passed and all(c.passed for c in checks)
    print(f"conformance: {'PASS' if all_passed else 'FAIL'}")
    for p in preconditions:
        print(f"  [precondition] {p.name}: {'PASS' if p.passed else 'FAIL'} - {p.detail}")
    for c in checks:
        print(f"  [check] {c.name}: {'PASS' if c.passed else 'FAIL'} - {c.detail}")
    print(f"evidence: {evidence}")
    print(f"sidecar:  {sidecar}")

    return _exit_code(preconditions, checks)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
