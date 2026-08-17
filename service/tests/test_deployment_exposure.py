"""Security assertions for the staged systemd deployment scaffolding."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict


def _deployment_unit(name: str) -> str:
    return (
        Path(__file__).parents[1]
        / "deployment"
        / "systemd"
        / name
    ).read_text("utf-8")


def _parse_unit(text: str) -> DefaultDict[str, list[tuple[str, str]]]:
    result: DefaultDict[str, list[tuple[str, str]]] = defaultdict(list)
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


def test_deployment_scaffolding_cannot_be_enabled_or_started_manually() -> None:
    """Prevent an unreviewed unit edit from publishing the local IPC endpoint."""

    service = _deployment_unit("poker-knight-ng.service")
    socket = _deployment_unit("poker-knight-ng.socket")

    for unit_text in (service, socket):
        parsed = _parse_unit(unit_text)
        assert any(key == "RefuseManualStart" and value == "yes" for key, value in parsed["Unit"])
        assert "Install" not in parsed



def test_staged_socket_surface_is_local_and_permission_bounded() -> None:
    socket_text = _deployment_unit("poker-knight-ng.socket")
    service_text = _deployment_unit("poker-knight-ng.service")
    socket_data = _parse_unit(socket_text)
    service_data = _parse_unit(service_text)

    # Socket section must be exact and cannot be overridden by later entries.
    listen_stream = [value for key, value in socket_data["Socket"] if key == "ListenStream"]
    assert listen_stream == ["/run/poker-knight-ng/service.sock"]

    assert not any(key == "ListenDatagram" for key, _ in socket_data["Socket"])
    assert not any(key == "ListenSequentialPacket" for key, _ in socket_data["Socket"])

    assert [value for key, value in socket_data["Socket"] if key == "SocketMode"] == ["0660"]
    assert [value for key, value in socket_data["Socket"] if key == "DirectoryMode"] == ["0750"]

    # Hard-fail on socket ownership settings.
    assert [value for key, value in socket_data["Socket"] if key == "SocketUser"] == ["poker-knight-ng"]
    assert [value for key, value in socket_data["Socket"] if key == "SocketGroup"] == ["poker-knight-ng"]

    assert any(key == "Service" and value == "poker-knight-ng.service" for key, value in socket_data["Socket"])

    # Canonical service identity and hardening are not optional.
    assert [value for key, value in service_data["Service"] if key == "User"] == ["poker-knight-ng"]
    assert [value for key, value in service_data["Service"] if key == "Group"] == ["poker-knight-ng"]
    assert [value for key, value in service_data["Service"] if key == "RuntimeDirectory"] == ["poker-knight-ng"]
    assert [value for key, value in service_data["Service"] if key == "RuntimeDirectoryMode"] == ["0750"]
    assert [value for key, value in service_data["Service"] if key == "RestrictAddressFamilies"] == ["AF_UNIX"]
