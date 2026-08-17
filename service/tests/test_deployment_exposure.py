"""Security assertions for the staged systemd deployment scaffolding."""
from __future__ import annotations

from pathlib import Path


_DEPLOYMENT = Path(__file__).parents[1] / "deployment" / "systemd"


def _unit(name: str) -> str:
    return (_DEPLOYMENT / name).read_text("utf-8")


def test_deployment_scaffolding_cannot_be_enabled_or_started_manually() -> None:
    """Prevent an unreviewed unit edit from publishing the local IPC endpoint."""

    service = _unit("poker-knight-ng.service")
    socket = _unit("poker-knight-ng.socket")

    for text in (service, socket):
        assert "RefuseManualStart=yes" in text
        assert "[Install]" not in text
        assert "WantedBy=" not in text


def test_staged_socket_surface_is_local_and_permission_bounded() -> None:
    socket = _unit("poker-knight-ng.socket")
    service = _unit("poker-knight-ng.service")

    assert "ListenStream=/run/poker-knight-ng/service.sock" in socket
    assert "SocketMode=0660" in socket
    assert "DirectoryMode=0750" in socket
    assert "ListenDatagram=" not in socket
    assert "ListenSequentialPacket=" not in socket

    assert "User=poker-knight-ng" in service
    assert "Group=poker-knight-ng" in service
    assert "RuntimeDirectory=poker-knight-ng" in service
    assert "RuntimeDirectoryMode=0750" in service
    assert "RestrictAddressFamilies=AF_UNIX" in service
