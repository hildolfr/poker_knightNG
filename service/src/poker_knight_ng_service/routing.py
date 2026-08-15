"""Closed HTTP route selection without engine invocation."""
from __future__ import annotations

from enum import Enum

from .framing import AdmittedRequest, TransportFailure


class Route(Enum):
    """The three routes accepted by service profile v1."""

    HEALTH = "health"
    CPU_SOLVE = "cpu-solve"
    CUDA_SOLVE = "cuda-solve"


_ROUTES: dict[bytes, tuple[bytes, Route]] = {
    b"/healthz": (b"GET", Route.HEALTH),
    b"/v1/solve": (b"POST", Route.CPU_SOLVE),
    b"/v1/solve-cuda": (b"POST", Route.CUDA_SOLVE),
}


def select_route(request: AdmittedRequest) -> Route:
    """Select one exact route or raise an empty transport failure."""

    route = _ROUTES.get(request.target)
    if route is None:
        raise TransportFailure(404)
    expected_method, selected = route
    if request.method != expected_method:
        raise TransportFailure(405)
    if selected is Route.HEALTH:
        if request.body:
            raise TransportFailure(400)
    elif not request.body:
        raise TransportFailure(400)
    return selected
