"""Listener-free coordination for exactly one admitted HTTP session."""
from __future__ import annotations

import json
from collections.abc import Callable
from threading import Lock
from typing import Protocol
from weakref import ReferenceType, ref

from poker_knight_ng.contract.errors import ContractProblem, PROBLEM_POLICIES, problem

from .adapter import adapt_solve_request
from .async_execution import execute_solve_async
from .connection import AsyncReader, read_admitted_request
from .framing import AdmittedRequest, TransportFailure
from .responses import (
    EMERGENCY_REQUEST_ID,
    RequestIdGenerationFailure,
    generate_request_id,
    serialize_health_response,
    serialize_json_response,
    serialize_transport_failure,
)
from .routing import Route, select_route


class AsyncSession(AsyncReader, Protocol):
    """One weak-referenceable reader plus response and close operations."""

    async def send_response(self, response: bytes) -> bool:
        """Send one complete response; return false when the peer is already gone."""
        ...

    def close(self) -> None:
        """Close the session without waiting."""
        ...

    async def wait_closed(self) -> bool | None:
        """Await graceful peer close and report peer-loss if applicable."""
        ...


def _build_session_claim_api():
    lock = Lock()
    claims: dict[int, ReferenceType[AsyncSession]] = {}

    def claim(session: AsyncSession) -> None:
        key = id(session)

        def forget(reference: ReferenceType[AsyncSession]) -> None:
            with lock:
                if claims.get(key) is reference:
                    claims.pop(key, None)

        try:
            reference = ref(session, forget)
        except TypeError:
            raise TypeError("sessions must support weak references") from None

        with lock:
            existing = claims.get(key)
            if existing is not None and existing() is session:
                raise RuntimeError("session is already owned")
            if existing is not None and existing() is not None:
                raise RuntimeError("session identity collision")
            claims[key] = reference

    return claim


_claim_session: Callable[[AsyncSession], None]
try:
    _claim_session  # pyright: ignore[reportUnboundVariable]
except NameError:
    _claim_session = _build_session_claim_api()
del _build_session_claim_api


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _problem_parts(failure: ContractProblem, request_id: str) -> tuple[int, dict[str, object]]:
    return PROBLEM_POLICIES[failure.code][2], failure.serialize(request_id)


def _json_response(status: int, payload: object, request_id: str) -> bytes:
    return serialize_json_response(
        status=status,
        body=_canonical_json(payload),
        request_id=request_id,
    )


def _internal_response(request_id: str) -> bytes:
    failure = problem("INTERNAL_ERROR")
    return _json_response(
        PROBLEM_POLICIES[failure.code][2],
        failure.serialize(request_id),
        request_id,
    )


async def _solve_response(admitted: AdmittedRequest) -> bytes:
    try:
        request_id = generate_request_id()
    except RequestIdGenerationFailure as id_failure:
        return _internal_response(id_failure.request_id)
    except Exception:
        return _internal_response(EMERGENCY_REQUEST_ID)

    try:
        try:
            adapted = adapt_solve_request(admitted)
            payload = await execute_solve_async(adapted)
            status = 200
        except ContractProblem as failure:
            status, payload = _problem_parts(failure, request_id)
        except Exception:
            status, payload = _problem_parts(problem("INTERNAL_ERROR"), request_id)
        return _json_response(status, payload, request_id)
    except Exception:
        return _internal_response(request_id)


async def handle_one_session(session: AsyncSession, on_admission: Callable[[], None] | None = None) -> None:
    """Handle exactly one request without owning a socket or listener."""

    _claim_session(session)
    try:
        try:
            admitted = await read_admitted_request(session)
            if on_admission is not None:
                on_admission()
            route = select_route(admitted)
            if route is Route.HEALTH:
                response = serialize_health_response()
            else:
                response = await _solve_response(admitted)
        except TransportFailure as failure:
            response = serialize_transport_failure(failure)
        await session.send_response(response)
    except BaseException:
        try:
            session.close()
        except BaseException:
            pass
        raise
    session.close()
