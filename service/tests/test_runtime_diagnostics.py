"""Private RAM-only runtime diagnostics contract."""
from __future__ import annotations

import pytest

from poker_knight_ng_service import runtime
from poker_knight_ng_service.framing import AdmittedRequest, TransportFailure
from poker_knight_ng_service.routing import select_route


def test_runtime_diagnostics_are_fixed_schema_and_not_ready_before_listener() -> None:
    service = runtime.ServiceRuntime()

    assert service.diagnostics_snapshot() == {
        "schema_version": "poker-knight-ng-runtime-diagnostics-v1",
        "readiness": "not-ready",
        "active_sessions": 0,
        "max_sessions": 16,
        "rejected_sessions": 0,
    }


def test_runtime_diagnostics_are_ready_only_after_listener_construction() -> None:
    service = runtime.ServiceRuntime()
    service._listener = object()  # type: ignore[assignment]

    assert service.diagnostics_snapshot()["readiness"] == "ready"
    service._stopping = True
    assert service.diagnostics_snapshot()["readiness"] == "not-ready"


def test_runtime_diagnostics_track_only_bounded_aggregate_admission_outcomes() -> None:
    service = runtime.ServiceRuntime()

    for _ in range(16):
        assert service._admit_session()
    assert not service._admit_session()
    service._release_session()

    assert service.diagnostics_snapshot() == {
        "schema_version": "poker-knight-ng-runtime-diagnostics-v1",
        "readiness": "not-ready",
        "active_sessions": 15,
        "max_sessions": 16,
        "rejected_sessions": 1,
    }


def test_runtime_diagnostics_do_not_expose_identifiers_secrets_or_paths() -> None:
    service = runtime.ServiceRuntime()
    snapshot = service.diagnostics_snapshot()

    forbidden = {
        "correlation_id",
        "exception",
        "exception_text",
        "host_path",
        "request_body",
        "request_id",
        "socket_path",
        "token",
    }
    assert forbidden.isdisjoint(snapshot)


def test_diagnostics_have_no_http_route() -> None:
    request = AdmittedRequest(b"GET", b"/diagnostics", (), b"")

    with pytest.raises(TransportFailure, match="HTTP transport rejected") as failure:
        select_route(request)
    assert failure.value.status == 404
