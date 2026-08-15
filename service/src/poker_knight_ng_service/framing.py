"""Bounded raw HTTP/1.1 framing admission."""
from __future__ import annotations

from dataclasses import dataclass

import h11


class TransportFailure(Exception):
    """Closed transport-only rejection with an empty response body."""

    def __init__(self, status: int) -> None:
        super().__init__("HTTP transport rejected")
        self.status = status
        self.body = b""


@dataclass(frozen=True, slots=True)
class AdmittedRequest:
    """One fully framed request admitted for semantic processing."""

    method: bytes
    target: bytes
    headers: tuple[tuple[bytes, bytes], ...]
    body: bytes


def serialize_response(*, status: int, body: bytes) -> bytes:
    """Serialize one bounded response that always closes the connection."""

    connection = h11.Connection(h11.SERVER)
    headers = [
        (b"Connection", b"close"),
        (b"Content-Length", str(len(body)).encode("ascii")),
    ]
    chunks = [
        connection.send(h11.Response(status_code=status, headers=headers, reason=b"")),
        connection.send(h11.Data(data=body)),
        connection.send(h11.EndOfMessage()),
    ]
    return b"".join(chunk for chunk in chunks if chunk is not None)


def _inspect_request_head(raw_head: bytes) -> int:
    """Validate one bounded request head and return its declared body length."""

    head, separator, trailing = raw_head.partition(b"\r\n\r\n")
    if len(head) + len(separator) > 8192:
        raise TransportFailure(431)
    if not separator or trailing:
        raise TransportFailure(400)
    lines = head.split(b"\r\n")
    if len(lines) - 1 > 32:
        raise TransportFailure(431)
    fields = [line.partition(b":") for line in lines[1:]]
    if any(not separator for _, separator, _ in fields):
        raise TransportFailure(400)
    if any(len(name) > 128 for name, _, _ in fields):
        raise TransportFailure(431)
    if any(len(value) > 1024 for _, _, value in fields):
        raise TransportFailure(431)
    names = [name.lower() for name, _, _ in fields]
    if any(name in names for name in (b"transfer-encoding", b"upgrade", b"expect")):
        raise TransportFailure(400)
    content_encodings = [
        value.strip(b" \t").lower()
        for name, _, value in fields
        if name.lower() == b"content-encoding"
    ]
    if any(value != b"identity" for value in content_encodings):
        raise TransportFailure(415)
    is_post = lines[0].startswith(b"POST ")
    if is_post and names.count(b"content-length") != 1:
        raise TransportFailure(400)
    if is_post:
        media_types = [
            value.strip(b" \t")
            for name, _, value in fields
            if name.lower() == b"content-type"
        ]
        if len(media_types) != 1 or media_types[0] not in {
            b"application/json",
            b"application/json; charset=utf-8",
        }:
            raise TransportFailure(415)
    declared_length: int | None = None
    if b"content-length" in names:
        raw_length = next(
            value.strip(b" \t")
            for name, separator, value in fields
            if separator and name.lower() == b"content-length"
        )
        if not raw_length.isdigit() or (
            len(raw_length) > 1 and raw_length.startswith(b"0")
        ):
            raise TransportFailure(400)
        if len(raw_length) > 5 or (
            len(raw_length) == 5 and raw_length > b"16384"
        ):
            raise TransportFailure(413)
        declared_length = int(raw_length)
        if is_post and declared_length == 0:
            raise TransportFailure(400)

    return declared_length or 0


def admit_request(raw: bytes) -> AdmittedRequest:
    """Parse one complete HTTP/1.1 request from already bounded bytes."""

    head, separator, raw_body = raw.partition(b"\r\n\r\n")
    declared_length = _inspect_request_head(head + separator)
    if len(raw_body) != declared_length:
        raise TransportFailure(400)

    try:
        connection = h11.Connection(h11.SERVER)
        connection.receive_data(raw)
        request: h11.Request | None = None
        body = bytearray()
        ended = False
        while True:
            event = connection.next_event()
            if event is h11.NEED_DATA or event is h11.PAUSED:
                break
            if isinstance(event, h11.Request):
                request = event
            elif isinstance(event, h11.Data):
                body.extend(event.data)
            elif isinstance(event, h11.EndOfMessage):
                ended = True
                break
    except h11.RemoteProtocolError:
        raise TransportFailure(400) from None

    if request is None or not ended:
        raise TransportFailure(400)
    if request.http_version != b"1.1":
        raise TransportFailure(400)
    trailing, _ = connection.trailing_data
    if trailing:
        raise TransportFailure(400)
    return AdmittedRequest(
        method=request.method,
        target=request.target,
        headers=tuple((bytes(name), bytes(value)) for name, value in request.headers),
        body=bytes(body),
    )
