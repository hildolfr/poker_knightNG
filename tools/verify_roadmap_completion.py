#!/usr/bin/env python3
"""Run the repository-local acceptance gate for roadmap completion claims.

This intentionally avoids network access and release/deployment side effects.  It
combines the maintained tests that bind roadmap claims to contracts, CI/release
policy, and the checked-in runtime/deployment artifacts.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "service"

ROOT_TESTS = (
    "tests/test_roadmap_status.py",
    "tests/test_ci_release_readiness.py",
    "tests/service/test_phase7b_listener_contract.py",
    "tests/service/test_http_service_contract.py",
    "tests/service/test_phase7b_service_package.py",
)
SERVICE_TESTS = (
    "service/tests/test_runtime.py",
    "service/tests/test_listener.py",
    "service/tests/test_deployment_exposure.py",
    "service/tests/test_runtime_diagnostics.py",
)


def require_deployment_artifacts() -> None:
    """Check the checked-in systemd rollout artifacts without starting a service."""
    service_unit = ROOT / "service/deployment/systemd/poker-knight-ng.service"
    deployment_readme = ROOT / "service/deployment/README.md"

    required = {
        service_unit: ("ExecStart=/usr/bin/env poker-knight-ng-service", "NoNewPrivileges=yes"),
        deployment_readme: ("not an installation or activation procedure", "Socket activation is forbidden"),
    }
    socket_units = list((ROOT / "service/deployment/systemd").glob("*.socket"))
    if socket_units:
        raise RuntimeError(f"socket activation units are forbidden: {socket_units}")
    for path, expectations in required.items():
        if not path.is_file():
            raise RuntimeError(f"missing deployment gate artifact: {path.relative_to(ROOT)}")
        text = path.read_text("utf-8")
        missing = [expectation for expectation in expectations if expectation not in text]
        if missing:
            raise RuntimeError(
                f"deployment gate artifact {path.relative_to(ROOT)} is missing: {', '.join(missing)}"
            )


def run_pytest(label: str, command: list[str], cwd: Path) -> None:
    print(f"==> {label}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def resolve_pytest_command(scope: str) -> list[str]:
    if not shutil.which("uv"):
        raise RuntimeError("'uv' is required for this gate script in this repository layout")

    if scope == "root":
        return ["uv", "run", "--frozen", "pytest", "-q", *ROOT_TESTS]
    if scope == "service":
        return ["uv", "run", "--project", "service", "--frozen", "pytest", "-q", *SERVICE_TESTS]
    raise ValueError(f"unknown scope: {scope}")


def main() -> int:
    print("==> deployment and observability documentation contract", flush=True)
    require_deployment_artifacts()

    run_pytest(
        "roadmap, contract, CI readiness, and release-documentation tests",
        resolve_pytest_command("root"),
        ROOT,
    )
    run_pytest(
        "service runtime and secure-listener tests",
        resolve_pytest_command("service"),
        ROOT,
    )
    print("Roadmap completion gate passed (local, deterministic, no deployment performed).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as error:
        print(f"Roadmap completion gate failed: {error}", file=sys.stderr)
        raise SystemExit(1)
