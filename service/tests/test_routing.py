"""Exact service route and response-envelope tests."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from poker_knight_ng_service.framing import AdmittedRequest


def request(method: bytes, target: bytes, body: bytes = b"") -> AdmittedRequest:
    from poker_knight_ng_service.framing import AdmittedRequest

    return AdmittedRequest(
        method=method,
        target=target,
        headers=((b"host", b"local"),),
        body=body,
    )


def test_exact_health_route_is_selected_without_backend() -> None:
    from poker_knight_ng_service.routing import Route, select_route

    assert select_route(request(b"GET", b"/healthz")) is Route.HEALTH


def test_health_route_rejects_nonempty_entity_as_empty_400() -> None:
    from poker_knight_ng_service.framing import TransportFailure
    from poker_knight_ng_service.routing import select_route

    with pytest.raises(TransportFailure) as caught:
        select_route(request(b"GET", b"/healthz", b"x"))

    assert caught.value.status == 400
    assert caught.value.body == b""


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (b"/v1/solve", "CPU_SOLVE"),
        (b"/v1/solve-cuda", "CUDA_SOLVE"),
    ],
)
def test_exact_post_solve_routes_are_selected(target: bytes, expected: str) -> None:
    from poker_knight_ng_service.routing import Route, select_route

    assert select_route(request(b"POST", target, b"{}")) is getattr(Route, expected)


@pytest.mark.parametrize("target", [b"/v1/solve", b"/v1/solve-cuda"])
def test_solve_route_rejects_empty_entity_as_empty_400(target: bytes) -> None:
    from poker_knight_ng_service.framing import TransportFailure
    from poker_knight_ng_service.routing import select_route

    with pytest.raises(TransportFailure) as caught:
        select_route(request(b"POST", target))

    assert caught.value.status == 400
    assert caught.value.body == b""


@pytest.mark.parametrize(
    ("method", "target"),
    [
        (b"POST", b"/healthz"),
        (b"GET", b"/v1/solve"),
        (b"PUT", b"/v1/solve-cuda"),
    ],
)
def test_wrong_method_on_known_path_is_empty_405(method: bytes, target: bytes) -> None:
    from poker_knight_ng_service.framing import TransportFailure
    from poker_knight_ng_service.routing import select_route

    with pytest.raises(TransportFailure) as caught:
        select_route(request(method, target))

    assert caught.value.status == 405
    assert caught.value.body == b""


@pytest.mark.parametrize(
    "target",
    [b"/", b"/healthz?verbose=1", b"/Healthz", b"/v1/solve/", b"http://local/healthz"],
)
def test_non_exact_path_is_empty_404(target: bytes) -> None:
    from poker_knight_ng_service.framing import TransportFailure
    from poker_knight_ng_service.routing import select_route

    with pytest.raises(TransportFailure) as caught:
        select_route(request(b"GET", target))

    assert caught.value.status == 404
    assert caught.value.body == b""


def test_health_response_is_empty_204_without_json_or_correlation() -> None:
    from poker_knight_ng_service.responses import serialize_health_response

    wire = serialize_health_response()

    assert wire.startswith(b"HTTP/1.1 204 \r\n")
    assert b"Connection: close\r\n" in wire
    assert b"Content-Length: 0\r\n" in wire
    assert b"Content-Type:" not in wire
    assert b"X-Poker-Knight-Request-ID:" not in wire
    assert wire.endswith(b"\r\n\r\n")


def test_accepted_json_response_has_closed_headers_and_json_line() -> None:
    from poker_knight_ng_service.responses import serialize_json_response

    request_id = "pk_0123456789abcdef0123456789abcdef"
    body = b'{"backend":"cpu_reference"}\n'

    wire = serialize_json_response(status=200, body=body, request_id=request_id)

    assert wire.startswith(b"HTTP/1.1 200 \r\n")
    assert b"Content-Type: application/json\r\n" in wire
    assert b"Cache-Control: no-store\r\n" in wire
    assert b"X-Content-Type-Options: nosniff\r\n" in wire
    assert (
        b"X-Poker-Knight-Request-ID: " + request_id.encode("ascii") + b"\r\n"
        in wire
    )
    assert b"Connection: close\r\n" in wire
    assert f"Content-Length: {len(body)}\r\n".encode("ascii") in wire
    assert wire.endswith(b"\r\n" + body)


def test_problem_response_binds_body_and_header_correlation() -> None:
    from poker_knight_ng_service.responses import serialize_json_response

    request_id = "pk_0123456789abcdef0123456789abcdef"
    body = (
        b'{"code":"BACKEND_UNAVAILABLE","correlation_id":"'
        + request_id.encode("ascii")
        + b'","message":"backend unavailable"}\n'
    )

    wire = serialize_json_response(status=503, body=body, request_id=request_id)

    assert wire.endswith(b"\r\n" + body)


def test_problem_response_rejects_correlation_mismatch() -> None:
    from poker_knight_ng_service.responses import serialize_json_response

    request_id = "pk_0123456789abcdef0123456789abcdef"
    other_id = "pk_fedcba9876543210fedcba9876543210"
    body = (
        b'{"code":"INTERNAL_ERROR","correlation_id":"'
        + other_id.encode("ascii")
        + b'","message":"internal error"}\n'
    )

    with pytest.raises(ValueError, match="correlation"):
        serialize_json_response(status=500, body=body, request_id=request_id)


@pytest.mark.parametrize("status", [400, 404, 405, 408, 413, 415, 431])
def test_transport_failure_serializes_empty_closed_response(status: int) -> None:
    from poker_knight_ng_service.framing import TransportFailure
    from poker_knight_ng_service.responses import serialize_transport_failure

    wire = serialize_transport_failure(TransportFailure(status))

    assert wire.startswith(f"HTTP/1.1 {status} \r\n".encode("ascii"))
    assert b"Content-Length: 0\r\n" in wire
    assert b"Connection: close\r\n" in wire
    assert b"Content-Type:" not in wire
    assert b"X-Poker-Knight-Request-ID:" not in wire
    assert wire.endswith(b"\r\n\r\n")


def test_semantic_status_cannot_use_transport_failure_envelope() -> None:
    from poker_knight_ng_service.framing import TransportFailure
    from poker_knight_ng_service.responses import serialize_transport_failure

    with pytest.raises(ValueError, match="transport"):
        serialize_transport_failure(TransportFailure(500))


@pytest.mark.parametrize(
    "request_id",
    [
        "pk_0123456789abcdef0123456789abcde",
        "pk_0123456789ABCDEF0123456789ABCDEF",
        "pk_0123456789abcdef0123456789abcdeg",
        "pk_0123456789abcdef0123456789abcdef\r\nX: injected",
        "pk_é123456789abcdef0123456789abcde",
    ],
)
def test_json_response_rejects_invalid_request_id(request_id: str) -> None:
    from poker_knight_ng_service.responses import serialize_json_response

    with pytest.raises(ValueError, match="request ID"):
        serialize_json_response(status=200, body=b'{"ok":true}\n', request_id=request_id)


@pytest.mark.parametrize(
    "body",
    [
        b'{"z":0,"a":1}\n',
        b'{ "a":1}\n',
        b'{"a":NaN}\n',
        b'["not-an-object"]\n',
        '{"a":"é"}\n'.encode(),
        b'{"a":1}',
        b'{"a":1,"a":1}\n',
    ],
)
def test_json_response_rejects_noncanonical_body(body: bytes) -> None:
    from poker_knight_ng_service.responses import serialize_json_response

    with pytest.raises(ValueError, match="canonical JSON"):
        serialize_json_response(
            status=200,
            body=body,
            request_id="pk_0123456789abcdef0123456789abcdef",
        )


@pytest.mark.parametrize(
    "body",
    [
        b'{"a":"\\ud800"}\n',
        b'{"a":"\\udfff"}\n',
        b'{"a":[{"b":"\\ud800"}]}\n',
        b'{"\\udfff":"value"}\n',
    ],
)
def test_json_response_rejects_lone_unicode_surrogate(body: bytes) -> None:
    from poker_knight_ng_service.responses import serialize_json_response

    with pytest.raises(ValueError, match="canonical JSON"):
        serialize_json_response(
            status=200,
            body=body,
            request_id="pk_0123456789abcdef0123456789abcdef",
        )


def test_json_response_preserves_valid_unicode_surrogate_pair() -> None:
    from poker_knight_ng_service.responses import serialize_json_response

    body = b'{"a":"\\ud83d\\ude00"}\n'
    wire = serialize_json_response(
        status=200,
        body=body,
        request_id="pk_0123456789abcdef0123456789abcdef",
    )

    assert wire.endswith(b"\r\n" + body)


def test_success_response_rejects_correlation_field() -> None:
    from poker_knight_ng_service.responses import serialize_json_response

    request_id = "pk_0123456789abcdef0123456789abcdef"
    body = b'{"correlation_id":"' + request_id.encode("ascii") + b'"}\n'

    with pytest.raises(ValueError, match="success response"):
        serialize_json_response(status=200, body=body, request_id=request_id)


@pytest.mark.parametrize("status", [201, 204, 404, 405, 408, 413, 415, 431])
def test_json_response_rejects_nonsemantic_status(status: int) -> None:
    from poker_knight_ng_service.responses import serialize_json_response

    with pytest.raises(ValueError, match="solve response status"):
        serialize_json_response(
            status=status,
            body=b'{"ok":true}\n',
            request_id="pk_0123456789abcdef0123456789abcdef",
        )


@pytest.mark.parametrize("name", [b"Connection", b"connection", b"Content-Length"])
def test_generic_serializer_rejects_reserved_caller_headers(name: bytes) -> None:
    from poker_knight_ng_service.framing import serialize_response

    with pytest.raises(ValueError, match="reserved"):
        serialize_response(status=200, body=b"", headers=((name, b"bad"),))


@pytest.mark.parametrize(
    "headers",
    [
        ((b"Cache-Control", b"no-store"), (b"cache-control", b"private")),
        (
            (b"X-Poker-Knight-Request-ID", b"pk_" + b"1" * 32),
            (b"x-poker-knight-request-id", b"pk_" + b"2" * 32),
        ),
    ],
)
def test_generic_serializer_rejects_duplicate_caller_headers(
    headers: tuple[tuple[bytes, bytes], ...],
) -> None:
    from poker_knight_ng_service.framing import serialize_response

    with pytest.raises(ValueError, match="duplicate"):
        serialize_response(status=200, body=b"", headers=headers)


@pytest.mark.parametrize("name", [b"Transfer-Encoding", b"Upgrade", b"Trailer"])
def test_generic_serializer_owns_hop_by_hop_response_headers(name: bytes) -> None:
    from poker_knight_ng_service.framing import serialize_response

    with pytest.raises(ValueError, match="reserved"):
        serialize_response(status=200, body=b"", headers=((name, b"bad"),))


def test_request_id_generation_has_exact_closed_format() -> None:
    from poker_knight_ng_service.responses import generate_request_id

    generated = generate_request_id(lambda size: "ab" * size)

    assert generated == "pk_" + "ab" * 16


@pytest.mark.parametrize(
    "factory",
    [
        lambda size: (_ for _ in ()).throw(RuntimeError("entropy unavailable")),
        lambda size: "A" * (size * 2),
        lambda size: "0" * (size * 2 - 1),
    ],
)
def test_request_id_generation_failure_carries_emergency_id(factory: object) -> None:
    from poker_knight_ng_service.responses import (
        EMERGENCY_REQUEST_ID,
        RequestIdGenerationFailure,
        generate_request_id,
    )

    with pytest.raises(RequestIdGenerationFailure) as caught:
        generate_request_id(factory)  # type: ignore[arg-type]

    assert caught.value.request_id == EMERGENCY_REQUEST_ID
    assert EMERGENCY_REQUEST_ID == "pk_" + "0" * 32


def test_request_id_generation_does_not_convert_process_control_signal() -> None:
    from poker_knight_ng_service.responses import generate_request_id

    def interrupt(_: int) -> str:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        generate_request_id(interrupt)
