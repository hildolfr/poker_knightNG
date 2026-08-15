"""Closed response envelopes for the private service."""
from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable

from .framing import TransportFailure, serialize_response

_REQUEST_ID = re.compile(r"pk_[0-9a-f]{32}")
_JSON_STATUSES = {200, 400, 422, 500, 503}
_TRANSPORT_STATUSES = {400, 404, 405, 408, 413, 415, 431}
EMERGENCY_REQUEST_ID = "pk_" + "0" * 32


class RequestIdGenerationFailure(Exception):
    """Force emergency-ID INTERNAL_ERROR handling without leaking entropy errors."""

    def __init__(self) -> None:
        super().__init__("request ID generation failed")
        self.request_id = EMERGENCY_REQUEST_ID


def generate_request_id(
    token_hex: Callable[[int], str] = secrets.token_hex,
) -> str:
    """Generate one exact request ID or force emergency internal-error handling."""

    try:
        request_id = "pk_" + token_hex(16)
        if _REQUEST_ID.fullmatch(request_id) is None:
            raise ValueError("malformed entropy output")
    except Exception:
        raise RequestIdGenerationFailure() from None
    return request_id


def _reject_constant(_: str) -> object:
    raise ValueError("non-finite JSON constant")


def _contains_surrogate(value: object) -> bool:
    if isinstance(value, str):
        return any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    if isinstance(value, list):
        return any(_contains_surrogate(item) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_surrogate(key) or _contains_surrogate(item)
            for key, item in value.items()
        )
    return False


def _canonical_json_object(body: bytes) -> dict[str, object]:
    try:
        text = body.decode("ascii")
        value = json.loads(
            text,
            parse_constant=_reject_constant,
        )
        canonical = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("invalid canonical JSON response") from exc
    if type(value) is not dict or _contains_surrogate(value) or canonical != body:
        raise ValueError("invalid canonical JSON response")
    return value


def serialize_health_response() -> bytes:
    """Serialize the exact empty health response without correlation metadata."""

    return serialize_response(status=204, body=b"")


def serialize_transport_failure(failure: TransportFailure) -> bytes:
    """Serialize one closed empty transport failure."""

    if failure.status not in _TRANSPORT_STATUSES or failure.body:
        raise ValueError("invalid transport failure")
    return serialize_response(status=failure.status, body=b"")


def serialize_json_response(*, status: int, body: bytes, request_id: str) -> bytes:
    """Serialize one accepted solve JSON line with fixed response headers."""

    if _REQUEST_ID.fullmatch(request_id) is None:
        raise ValueError("invalid request ID")
    if status not in _JSON_STATUSES:
        raise ValueError("invalid solve response status")
    value = _canonical_json_object(body)
    if status == 200:
        if "correlation_id" in value:
            raise ValueError("success response contains correlation ID")
    elif value.get("correlation_id") != request_id:
        raise ValueError("problem correlation ID mismatch")
    headers = (
        (b"Content-Type", b"application/json"),
        (b"Cache-Control", b"no-store"),
        (b"X-Content-Type-Options", b"nosniff"),
        (b"X-Poker-Knight-Request-ID", request_id.encode("ascii")),
    )
    return serialize_response(status=status, body=body, headers=headers)
