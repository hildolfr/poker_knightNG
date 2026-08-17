"""Security assertions for the staged direct-bind systemd deployment."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import subprocess

import pytest


DEPLOYMENT = Path(__file__).parents[1] / "deployment"


def _deployment_unit(name: str) -> str:
    return (DEPLOYMENT / "systemd" / name).read_text("utf-8")


def _parse_unit(text: str) -> defaultdict[str, list[tuple[str, str]]]:
    result: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";") or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            continue
        if section and "=" in stripped:
            key, value = stripped.split("=", 1)
            result[section].append((key.strip(), value.strip()))
    return result


def _copied_deployment(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    copied = root / "service" / "deployment"
    shutil.copytree(DEPLOYMENT, copied)
    return root, copied


def _validate(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", "service/deployment/validate-systemd.sh"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )


def test_deployment_scaffolding_cannot_be_enabled_or_started_manually() -> None:
    """Prevent an unreviewed unit edit from publishing the local IPC endpoint."""
    parsed = _parse_unit(_deployment_unit("poker-knight-ng.service"))
    assert [value for key, value in parsed["Unit"] if key == "RefuseManualStart"] == ["yes"]
    assert "Install" not in parsed


def test_staged_service_performs_its_own_bounded_local_bind() -> None:
    service_data = _parse_unit(_deployment_unit("poker-knight-ng.service"))
    assert [value for key, value in service_data["Service"] if key == "User"] == ["poker-knight-ng"]
    assert [value for key, value in service_data["Service"] if key == "Group"] == ["poker-knight-ng"]
    assert [value for key, value in service_data["Service"] if key == "RuntimeDirectory"] == ["poker-knight-ng"]
    assert [value for key, value in service_data["Service"] if key == "RuntimeDirectoryMode"] == ["0750"]
    assert [value for key, value in service_data["Service"] if key == "RestrictAddressFamilies"] == ["AF_UNIX"]
    assert not list((DEPLOYMENT / "systemd").glob("*.socket"))


@pytest.mark.parametrize("forbidden_value", ["no", "yes\nRefuseManualStart=no"])
def test_validator_rejects_non_exact_or_overridden_manual_start(
    tmp_path: Path, forbidden_value: str
) -> None:
    root, deployment = _copied_deployment(tmp_path)
    unit = deployment / "systemd" / "poker-knight-ng.service"
    unit.write_text(unit.read_text("utf-8").replace("RefuseManualStart=yes", f"RefuseManualStart={forbidden_value}"), "utf-8")
    result = _validate(root)
    assert result.returncode != 0
    assert "RefuseManualStart" in result.stderr


def test_validator_rejects_reintroduced_socket_activation_unit(tmp_path: Path) -> None:
    root, deployment = _copied_deployment(tmp_path)
    (deployment / "systemd" / "poker-knight-ng.socket").write_text(
        "[Socket]\nListenStream=/run/poker-knight-ng/service.sock\nService=attacker.service\n",
        "utf-8",
    )
    result = _validate(root)
    assert result.returncode != 0
    assert "socket activation units are forbidden" in result.stderr
