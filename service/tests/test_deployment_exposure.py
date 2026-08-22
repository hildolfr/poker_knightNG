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


def test_activated_unit_installs_and_carries_no_manual_start_guard() -> None:
    """Post-activation: exact [Install] target, no RefuseManualStart anywhere."""
    parsed = _parse_unit(_deployment_unit("poker-knight-ng.service"))
    assert [value for key, value in parsed["Install"] if key == "WantedBy"] == ["multi-user.target"]
    assert [key for key, _ in parsed["Unit"] if key == "RefuseManualStart"] == []


def test_staged_service_performs_its_own_bounded_local_bind() -> None:
    service_data = _parse_unit(_deployment_unit("poker-knight-ng.service"))
    assert [value for key, value in service_data["Service"] if key == "User"] == ["poker-knight-ng"]
    assert [value for key, value in service_data["Service"] if key == "Group"] == ["poker-knight-ng"]
    assert [value for key, value in service_data["Service"] if key == "RuntimeDirectory"] == ["poker-knight-ng"]
    assert [value for key, value in service_data["Service"] if key == "RuntimeDirectoryMode"] == ["0750"]
    assert [value for key, value in service_data["Service"] if key == "RestrictAddressFamilies"] == ["AF_UNIX"]
    assert not list((DEPLOYMENT / "systemd").glob("*.socket"))


@pytest.mark.parametrize("forbidden_install", ["", "WantedBy=default.target", "RequiredBy=multi-user.target\nWantedBy=multi-user.target"])
def test_validator_rejects_missing_or_wrong_or_overridden_install_target(
    tmp_path: Path, forbidden_install: str
) -> None:
    root, deployment = _copied_deployment(tmp_path)
    unit = deployment / "systemd" / "poker-knight-ng.service"
    text = unit.read_text("utf-8")
    if forbidden_install:
        text = text.replace(
            "[Install]\nWantedBy=multi-user.target",
            f"[Install]\n{forbidden_install}",
        )
    else:
        text = text.replace("[Install]\nWantedBy=multi-user.target", "")
    unit.write_text(text, "utf-8")
    result = _validate(root)
    assert result.returncode != 0
    assert "WantedBy" in result.stderr or "RequiredBy" in result.stderr


def test_validator_rejects_reintroduced_manual_start_guard(tmp_path: Path) -> None:
    """A RefuseManualStart guard must never silently return post-activation."""
    root, deployment = _copied_deployment(tmp_path)
    unit = deployment / "systemd" / "poker-knight-ng.service"
    unit.write_text(
        unit.read_text("utf-8").replace("[Service]", "RefuseManualStart=yes\n[Service]"),
        "utf-8",
    )
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


def test_validator_rejects_unit_with_restrict_address_families_missing(
    tmp_path: Path,
) -> None:
    root, deployment = _copied_deployment(tmp_path)
    unit = deployment / "systemd" / "poker-knight-ng.service"
    unit.write_text(
        unit.read_text("utf-8").replace("RestrictAddressFamilies=AF_UNIX\n", ""),
        "utf-8",
    )
    result = _validate(root)
    assert result.returncode != 0
    assert "RestrictAddressFamilies must be AF_UNIX" in result.stderr


@pytest.mark.parametrize("families", ["AF_UNIX AF_INET", "AF_UNIX AF_INET6", "AF_INET"])
def test_validator_rejects_restrict_address_families_containing_tcp(
    tmp_path: Path, families: str
) -> None:
    """Address families other than the exact AF_UNIX set are rejected."""
    root, deployment = _copied_deployment(tmp_path)
    unit = deployment / "systemd" / "poker-knight-ng.service"
    unit.write_text(
        unit.read_text("utf-8").replace(
            "RestrictAddressFamilies=AF_UNIX", f"RestrictAddressFamilies={families}"
        ),
        "utf-8",
    )
    result = _validate(root)
    assert result.returncode != 0
    assert "RestrictAddressFamilies must be AF_UNIX" in result.stderr


def test_validator_rejects_sockets_wiring_in_service_unit(tmp_path: Path) -> None:
    """A Sockets= directive would hand a net-listening fd to the service; rejected."""
    root, deployment = _copied_deployment(tmp_path)
    unit = deployment / "systemd" / "poker-knight-ng.service"
    unit.write_text(
        unit.read_text("utf-8").replace(
            "RestrictAddressFamilies=AF_UNIX",
            "RestrictAddressFamilies=AF_UNIX\nSockets=attacker.socket",
        ),
        "utf-8",
    )
    result = _validate(root)
    assert result.returncode != 0
    assert "must not define Sockets" in result.stderr


def test_shipped_unit_has_no_net_listening_directives() -> None:
    """The shipped service unit must not declare any listening/proxy surface."""
    parsed = _parse_unit(_deployment_unit("poker-knight-ng.service"))
    service_keys = {key for key, _ in parsed["Service"]}
    net_listening = {
        "ListenStream",
        "ListenDatagram",
        "ListenNetlink",
        "ListenSequentialPacket",
        "Sockets",
        "Bind",
        "FileDescriptorName",
    }
    assert net_listening.isdisjoint(service_keys)


def test_shipped_unit_lacks_socket_activation_and_aux_sections() -> None:
    parsed = _parse_unit(_deployment_unit("poker-knight-ng.service"))
    forbidden_sections = {"Socket", "Network", "Path", "Timer", "Mount"}
    assert forbidden_sections.isdisjoint(parsed)
    assert "Unit" in parsed
    assert "Service" in parsed


def test_diagnostics_snapshot_exposes_no_peer_request_card_seed_or_path_fields() -> None:
    """The operator diagnostics snapshot must stay a fixed, RAM-only aggregate."""
    from poker_knight_ng_service.runtime import ServiceRuntime

    snapshot = ServiceRuntime().diagnostics_snapshot()
    forbidden_substrings = ("peer", "request", "card", "seed", "path")
    for key in snapshot:
        assert not any(token in key for token in forbidden_substrings)
    assert set(snapshot) == {
        "schema_version",
        "readiness",
        "active_sessions",
        "max_sessions",
        "rejected_sessions",
    }
