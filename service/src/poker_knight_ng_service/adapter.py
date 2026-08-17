"""Strict JSON adaptation into the frozen v1 request model."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from poker_knight_ng.contract.errors import problem
from poker_knight_ng.contract.models import EquityRequest

from .framing import AdmittedRequest
from .routing import Route, select_route

_MAX_REQUEST_BYTES = 16_384
_MAX_REQUESTED_TRIALS = 1_000_000


class _MalformedJson(Exception):
    """Private marker that never retains submitted request values."""


def _duplicate_free(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _MalformedJson
        value[key] = item
    return value


def _reject_constant(_: str) -> object:
    raise _MalformedJson


def _parse_integer(token: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise _MalformedJson from None


def _decode_json_object(body: bytes) -> dict[str, Any]:
    if type(body) is not bytes or not body or len(body) > _MAX_REQUEST_BYTES:
        raise problem("INTERNAL_ERROR")
    try:
        text = body.decode("utf-8")
        if text.startswith("\ufeff"):
            raise _MalformedJson
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_free,
            parse_constant=_reject_constant,
            parse_int=_parse_integer,
        )
    except _MalformedJson:
        raise problem("UNSUPPORTED_REQUEST") from None
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise problem("UNSUPPORTED_REQUEST") from None
    if type(value) is not dict:
        raise problem("UNSUPPORTED_REQUEST")
    return value


@dataclass(frozen=True, init=False)
class AdaptedSolveRequest:
    """One semantically parsed request bound to its explicit service route."""

    route: Route
    request: EquityRequest


def _adapted_request(route: Route, request: EquityRequest) -> AdaptedSolveRequest:
    adapted = object.__new__(AdaptedSolveRequest)
    object.__setattr__(adapted, "route", route)
    object.__setattr__(adapted, "request", request)
    return adapted


def _trusted_snapshot(admitted: AdmittedRequest) -> AdmittedRequest:
    if type(admitted) is not AdmittedRequest:
        raise problem("INTERNAL_ERROR")
    try:
        method = object.__getattribute__(admitted, "method")
        target = object.__getattribute__(admitted, "target")
        headers = object.__getattribute__(admitted, "headers")
        body = object.__getattribute__(admitted, "body")
    except Exception:
        raise problem("INTERNAL_ERROR") from None
    if (
        type(method) is not bytes
        or type(target) is not bytes
        or type(headers) is not tuple
        or type(body) is not bytes
        or len(headers) > 32
        or len(body) > _MAX_REQUEST_BYTES
    ):
        raise problem("INTERNAL_ERROR")
    for pair in headers:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not bytes
            or type(pair[1]) is not bytes
            or len(pair[0]) > 128
            or len(pair[1]) > 1024
        ):
            raise problem("INTERNAL_ERROR")
    return AdmittedRequest(method=method, target=target, headers=headers, body=body)


def adapt_solve_request(admitted: AdmittedRequest) -> AdaptedSolveRequest:
    """Parse and bind one framed solve request without invoking any engine."""

    trusted = _trusted_snapshot(admitted)
    route = select_route(trusted)
    if route is Route.HEALTH:
        raise problem("INTERNAL_ERROR")

    request = EquityRequest.parse(_decode_json_object(trusted.body))
    if route is Route.CUDA_SOLVE and request.backend != "cuda":
        raise problem("UNSUPPORTED_REQUEST")
    if request.requested_trials > _MAX_REQUESTED_TRIALS:
        raise problem("UNSUPPORTED_REQUEST")
    return _adapted_request(route, request)
