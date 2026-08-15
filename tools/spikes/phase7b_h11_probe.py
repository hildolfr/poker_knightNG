#!/usr/bin/env python3
"""Reproduce the Phase 7B h11 selection without opening a listener."""
from __future__ import annotations

import json
import sys
from importlib.metadata import version

import h11

BASE_COMMIT = "6c75cf7b260a1a83f2fbc8cba7cd81c5d1198d70"
WHEEL_SHA256 = "63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86"
H11_VERSION = version("h11")


def consume(raw: bytes, *, max_incomplete_event_size: int = 8192) -> tuple[h11.Connection, list[object]]:
    connection = h11.Connection(
        h11.SERVER,
        max_incomplete_event_size=max_incomplete_event_size,
    )
    connection.receive_data(raw)
    events: list[object] = []
    while True:
        event = connection.next_event()
        if event in (h11.NEED_DATA, h11.PAUSED):
            return connection, events
        events.append(event)


def rejects(raw: bytes) -> bool:
    try:
        consume(raw)
    except h11.RemoteProtocolError:
        return True
    return False


def main() -> int:
    if H11_VERSION != "0.16.0":
        raise RuntimeError("Phase 7B probe requires exact h11 0.16.0")

    valid = (
        b"POST /v1/solve HTTP/1.1\r\n"
        b"Host: local\r\nContent-Length: 3\r\n\r\nabc"
    )
    _, valid_events = consume(valid)
    valid_request_body_events = (
        len(valid_events) == 3
        and isinstance(valid_events[0], h11.Request)
        and valid_events[0].method == b"POST"
        and valid_events[0].target == b"/v1/solve"
        and isinstance(valid_events[1], h11.Data)
        and valid_events[1].data == b"abc"
        and isinstance(valid_events[2], h11.EndOfMessage)
    )

    duplicate = (
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\n"
        b"Content-Length: 3\r\nContent-Length: 3\r\n\r\nabc"
    )
    _, duplicate_events = consume(duplicate)
    duplicate_request = duplicate_events[0]
    if not isinstance(duplicate_request, h11.Request):
        raise RuntimeError("duplicate Content-Length request did not parse")
    duplicate_headers = [
        (name, value)
        for name, value in duplicate_request.headers
        if name == b"content-length"
    ]

    conflicting = (
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\n"
        b"Content-Length: 3\r\nContent-Length: 4\r\n\r\nabc"
    )
    chunked = (
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n"
    )
    chunked_with_length = (
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\n"
        b"Transfer-Encoding: chunked\r\nContent-Length: 3\r\n\r\n"
        b"3\r\nabc\r\n0\r\n\r\n"
    )
    malformed = b"GET /healthz HTTP/1.1\r\nHost: local\r\nBad Name: x\r\n\r\n"

    _, chunked_events = consume(chunked)
    _, te_cl_events = consume(chunked_with_length)

    first = b"GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n"
    pipelined, pipeline_events = consume(first + first)
    pipelined_bytes_exposed = (
        len(pipeline_events) == 2
        and isinstance(pipeline_events[0], h11.Request)
        and isinstance(pipeline_events[1], h11.EndOfMessage)
        and pipelined.their_state is h11.DONE
        and bool(pipelined.trailing_data[0])
    )

    response_connection, response_events = consume(first)
    if not (
        len(response_events) == 2
        and isinstance(response_events[0], h11.Request)
        and isinstance(response_events[1], h11.EndOfMessage)
    ):
        raise RuntimeError("response precondition failed")
    response_bytes = response_connection.send(
        h11.Response(
            status_code=204,
            headers=[(b"Connection", b"close"), (b"Cache-Control", b"no-store")],
        )
    ) + response_connection.send(h11.EndOfMessage())

    evidence = {
        "adapter_requirements": [
            "bounded-raw-header-admission-before-h11",
            "exactly-one-canonical-content-length",
            "reject-all-transfer-encoding",
            "header-and-body-monotonic-deadlines",
            "exact-body-and-surplus-byte-rejection",
            "inspect-h11-trailing-data",
            "never-call-start-next-cycle",
            "explicit-connection-close-and-socket-close",
        ],
        "base_commit": BASE_COMMIT,
        "candidate": {
            "name": "h11",
            "version": H11_VERSION,
            "wheel_sha256": WHEEL_SHA256,
        },
        "format_version": "poker-knight-ng-phase7b-runtime-spike-v1",
        "no_listener": True,
        "observations": {
            "conflicting_content_length_rejected": rejects(conflicting),
            "duplicate_equal_content_length_normalized": duplicate_headers == [(b"content-length", b"3")],
            "explicit_connection_close_serialized": b"Connection: close\r\n" in response_bytes,
            "malformed_header_rejected": rejects(malformed),
            "pipelined_bytes_exposed_as_trailing_data": pipelined_bytes_exposed,
            "transfer_encoding_chunked_accepted": any(
                isinstance(event, h11.Data) and event.data == b"abc"
                for event in chunked_events
            ),
            "transfer_encoding_with_content_length_accepted": any(
                isinstance(event, h11.Data) and event.data == b"abc"
                for event in te_cl_events
            ),
            "valid_request_body_events": valid_request_body_events,
        },
        "rejected_candidates": {"aiohttp": "3.14.3", "uvicorn": "0.52.3"},
        "selection": "direct-h11-with-bounded-raw-admission",
        "verdict": "VALIDATED",
    }
    if not all(evidence["observations"].values()):
        raise RuntimeError("one or more h11 observations did not reproduce")
    sys.stdout.write(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
