"""Bounded raw HTTP framing admission."""
from __future__ import annotations

import pytest


def test_valid_post_is_admitted_with_exact_body() -> None:
    from poker_knight_ng_service.framing import admit_request

    raw = (
        b"POST /v1/solve HTTP/1.1\r\n"
        b"Host: local\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 2\r\n"
        b"\r\n{}"
    )

    request = admit_request(raw)

    assert request.method == b"POST"
    assert request.target == b"/v1/solve"
    assert request.body == b"{}"
    assert request.headers == (
        (b"host", b"local"),
        (b"content-type", b"application/json"),
        (b"content-length", b"2"),
    )


def test_duplicate_equal_content_length_is_empty_400() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = (
        b"POST /v1/solve HTTP/1.1\r\n"
        b"Host: local\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 2\r\n"
        b"Content-Length: 2\r\n"
        b"\r\n{}"
    )

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 400
    assert caught.value.body == b""


def test_transfer_encoding_is_empty_400_before_h11() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = (
        b"POST /v1/solve HTTP/1.1\r\n"
        b"Host: local\r\n"
        b"Content-Type: application/json\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"Content-Length: 2\r\n"
        b"\r\n2\r\n{}\r\n0\r\n\r\n"
    )

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 400
    assert caught.value.body == b""


def test_aggregate_header_overflow_is_empty_431() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = b"GET /healthz HTTP/1.1\r\nHost: local\r\nX-Fill: " + b"a" * 8192 + b"\r\n\r\n"

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 431
    assert caught.value.body == b""


@pytest.mark.parametrize("raw_length", [b"16385", b"100000", b"9" * 1024])
def test_declared_payload_overflow_is_empty_413(raw_length: bytes) -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = (
        b"POST /v1/solve HTTP/1.1\r\n"
        b"Host: local\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length:" + raw_length + b"\r\n"
        b"\r\n"
    )

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 413
    assert caught.value.body == b""


def test_thirty_third_header_field_is_empty_431() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    fields = b"".join(f"X-{index}: value\r\n".encode("ascii") for index in range(32))
    raw = b"GET /healthz HTTP/1.1\r\nHost: local\r\n" + fields + b"\r\n"

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 431
    assert caught.value.body == b""


def test_header_name_over_128_bytes_is_empty_431() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = b"GET /healthz HTTP/1.1\r\nHost: local\r\n" + b"X" * 129 + b": value\r\n\r\n"

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 431
    assert caught.value.body == b""


def test_header_value_over_1024_bytes_is_empty_431() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = b"GET /healthz HTTP/1.1\r\nHost: local\r\nX-Fill: " + b"a" * 1025 + b"\r\n\r\n"

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 431
    assert caught.value.body == b""


def test_header_value_ows_over_1024_bytes_is_empty_431() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = (
        b"GET /healthz HTTP/1.1\r\nHost: local\r\nX-Fill:"
        + b" " * 1025
        + b"\r\n\r\n"
    )

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 431
    assert caught.value.body == b""


def test_header_value_exactly_1024_raw_bytes_is_admitted() -> None:
    from poker_knight_ng_service.framing import admit_request

    raw = (
        b"GET /healthz HTTP/1.1\r\nHost: local\r\nX-Fill:"
        + b"x" * 1024
        + b"\r\n\r\n"
    )

    admitted = admit_request(raw)

    assert (b"x-fill", b"x" * 1024) in admitted.headers


def test_malformed_header_line_is_empty_400() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = b"GET /healthz HTTP/1.1\r\nHost: local\r\nMalformed\r\n\r\n"

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 400
    assert caught.value.body == b""


def test_malformed_request_line_is_empty_400() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = b"BROKEN\r\nHost: local\r\n\r\n"

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 400
    assert caught.value.body == b""


def test_incomplete_declared_body_is_empty_400() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = (
        b"POST /v1/solve HTTP/1.1\r\n"
        b"Host: local\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 2\r\n"
        b"\r\n{"
    )

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 400
    assert caught.value.body == b""


def test_surplus_pipelined_bytes_are_empty_400() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = (
        b"POST /v1/solve HTTP/1.1\r\n"
        b"Host: local\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 2\r\n"
        b"\r\n{}"
        b"GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n"
    )

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 400
    assert caught.value.body == b""


def test_post_zero_content_length_is_empty_400() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = (
        b"POST /v1/solve HTTP/1.1\r\n"
        b"Host: local\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 0\r\n"
        b"\r\n"
    )

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 400
    assert caught.value.body == b""


def test_unsupported_post_media_type_is_empty_415() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = (
        b"POST /v1/solve HTTP/1.1\r\n"
        b"Host: local\r\n"
        b"Content-Type: text/plain\r\n"
        b"Content-Length: 2\r\n"
        b"\r\n{}"
    )

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 415
    assert caught.value.body == b""


def test_non_identity_content_encoding_is_empty_415() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = (
        b"POST /v1/solve HTTP/1.1\r\n"
        b"Host: local\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Encoding: gzip\r\n"
        b"Content-Length: 2\r\n"
        b"\r\n{}"
    )

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 415
    assert caught.value.body == b""


def test_empty_transport_response_closes_after_one_response() -> None:
    from poker_knight_ng_service.framing import serialize_response

    wire = serialize_response(status=431, body=b"")

    assert wire.startswith(b"HTTP/1.1 431 ")
    assert b"Connection: close\r\n" in wire
    assert b"Content-Length: 0\r\n" in wire
    assert wire.endswith(b"\r\n\r\n")


def test_http_1_0_request_is_empty_400() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = b"GET /healthz HTTP/1.0\r\nHost: local\r\n\r\n"

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 400
    assert caught.value.body == b""


def test_upgrade_attempt_is_empty_400() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = (
        b"GET /healthz HTTP/1.1\r\n"
        b"Host: local\r\n"
        b"Connection: upgrade\r\n"
        b"Upgrade: websocket\r\n"
        b"\r\n"
    )

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 400
    assert caught.value.body == b""


def test_oversized_surplus_is_empty_400_before_h11() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = (
        b"POST /v1/solve HTTP/1.1\r\n"
        b"Host: local\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 2\r\n"
        b"\r\n"
        + b"x" * 16385
    )

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 400
    assert caught.value.body == b""


def test_expect_continue_is_empty_400() -> None:
    from poker_knight_ng_service.framing import TransportFailure, admit_request

    raw = (
        b"POST /v1/solve HTTP/1.1\r\n"
        b"Host: local\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 2\r\n"
        b"Expect: 100-continue\r\n"
        b"\r\n{}"
    )

    with pytest.raises(TransportFailure) as caught:
        admit_request(raw)

    assert caught.value.status == 400
    assert caught.value.body == b""
