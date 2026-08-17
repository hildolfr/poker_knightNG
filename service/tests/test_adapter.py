"""Bounded service request-adapter tests."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from poker_knight_ng_service.framing import AdmittedRequest


def valid_body(*, backend: str = "cpu_reference", requested_trials: str = "1000") -> bytes:
    return (
        b'{"backend":"'
        + backend.encode("ascii")
        + b'","board_cards":[],"contract_version":"v1",'
        b'"hero_cards":["As","Kd"],"opponent_count":"1",'
        b'"requested_trials":"'
        + requested_trials.encode("ascii")
        + b'","rng":{"algorithm_id":"poker-knight-ng/philox4x32-10",'
        b'"algorithm_version":"1"},"seed":"0x0000000000000001"}'
    )


def admitted_request(body: bytes, *, target: bytes = b"/v1/solve") -> AdmittedRequest:
    from poker_knight_ng_service.framing import AdmittedRequest

    return AdmittedRequest(
        method=b"POST",
        target=target,
        headers=((b"content-type", b"application/json"),),
        body=body,
    )


def test_valid_cpu_route_adapts_through_frozen_equity_request_parser() -> None:
    from poker_knight_ng_service.adapter import adapt_solve_request
    from poker_knight_ng_service.routing import Route

    adapted = adapt_solve_request(admitted_request(valid_body()))

    assert adapted.route is Route.AUTO_SOLVE
    assert adapted.request.backend == "cpu_reference"
    assert adapted.request.requested_trials == 1000
    assert adapted.request.seed == 1


def test_adapted_request_cannot_be_constructed_outside_adapter() -> None:
    from poker_knight_ng.contract.models import EquityRequest
    from poker_knight_ng_service.adapter import AdaptedSolveRequest
    from poker_knight_ng_service.routing import Route

    request = EquityRequest.parse(json.loads(valid_body()))

    with pytest.raises(TypeError):
        AdaptedSolveRequest(route=Route.CPU_SOLVE, request=request)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "body",
    [
        b"{",
        b"{}{}",
        b"\xff",
        b"\xef\xbb\xbf{}",
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":1,"a":2}',
        b'{"a":{"b":1,"b":2}}',
        b"[]",
        b"null",
        (b'{' + b'"a":' * 600 + b"0" + b"}" * 600),
    ],
)
def test_malformed_accepted_json_maps_only_to_unsupported_request(body: bytes) -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service.adapter import adapt_solve_request

    with pytest.raises(ContractProblem) as caught:
        adapt_solve_request(admitted_request(body))

    assert caught.value.code == "UNSUPPORTED_REQUEST"


def test_noncanonical_but_valid_json_is_accepted() -> None:
    from poker_knight_ng_service.adapter import adapt_solve_request

    decoded = json.loads(valid_body())
    body = json.dumps(decoded, indent=2).encode("utf-8") + b" \n\t"

    adapted = adapt_solve_request(admitted_request(body))

    assert adapted.request.requested_trials == 1000


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda value: value.pop("rng"), "UNSUPPORTED_FIELD"),
        (lambda value: value.__setitem__("contract_version", "v2"), "INVALID_CONTRACT_VERSION"),
        (lambda value: value.__setitem__("hero_cards", ["XX", "Kd"]), "INVALID_CARD"),
        (lambda value: value.__setitem__("opponent_count", "7"), "INVALID_OPPONENT_COUNT"),
        (lambda value: value.__setitem__("seed", "1"), "INVALID_SEED"),
    ],
)
def test_frozen_equity_request_problem_codes_are_preserved(
    mutate: Callable[[dict[str, object]], object],
    expected: str,
) -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service.adapter import adapt_solve_request

    value = json.loads(valid_body())
    mutate(value)
    body = json.dumps(value, separators=(",", ":")).encode("ascii")

    with pytest.raises(ContractProblem) as caught:
        adapt_solve_request(admitted_request(body))

    assert caught.value.code == expected



def test_auto_route_allows_cpu_and_cuda_backends() -> None:
    from poker_knight_ng_service.adapter import adapt_solve_request
    from poker_knight_ng_service.routing import Route

    assert adapt_solve_request(admitted_request(valid_body())).route is Route.AUTO_SOLVE
    assert adapt_solve_request(admitted_request(valid_body(backend="cuda"))).route is Route.AUTO_SOLVE


def test_route_trial_cap_still_applies_for_auto_route() -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service.adapter import adapt_solve_request

    with pytest.raises(ContractProblem) as caught:
        adapt_solve_request(admitted_request(valid_body(requested_trials="1000001")))

    assert caught.value.code == "UNSUPPORTED_REQUEST"


def test_cuda_route_rejects_explicit_cpu_as_unsupported_request() -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service.adapter import adapt_solve_request

    with pytest.raises(ContractProblem) as caught:
        adapt_solve_request(
            admitted_request(valid_body(), target=b"/v1/solve-cuda")
        )

    assert caught.value.code == "UNSUPPORTED_REQUEST"


def test_cuda_route_accepts_explicit_cuda_without_engine_probe() -> None:
    from poker_knight_ng_service.adapter import adapt_solve_request
    from poker_knight_ng_service.routing import Route

    adapted = adapt_solve_request(
        admitted_request(valid_body(backend="cuda"), target=b"/v1/solve-cuda")
    )

    assert adapted.route is Route.CUDA_SOLVE
    assert adapted.request.backend == "cuda"


@pytest.mark.parametrize(
    ("trials", "accepted"),
    [("1000000", True), ("1000001", False)],
)
def test_service_trial_cap_is_exact(trials: str, accepted: bool) -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service.adapter import adapt_solve_request

    if accepted:
        adapted = adapt_solve_request(
            admitted_request(valid_body(requested_trials=trials))
        )
        assert adapted.request.requested_trials == 1_000_000
    else:
        with pytest.raises(ContractProblem) as caught:
            adapt_solve_request(
                admitted_request(valid_body(requested_trials=trials))
            )
        assert caught.value.code == "UNSUPPORTED_REQUEST"


def test_health_request_cannot_enter_solve_adapter() -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service.adapter import adapt_solve_request

    health = admitted_request(b"", target=b"/healthz")
    object.__setattr__(health, "method", b"GET")

    with pytest.raises(ContractProblem) as caught:
        adapt_solve_request(health)

    assert caught.value.code == "INTERNAL_ERROR"


def test_solve_adapter_rejects_admitted_request_subclass_before_field_access() -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service.adapter import adapt_solve_request
    from poker_knight_ng_service.framing import AdmittedRequest

    class Forged(AdmittedRequest):
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(name)

    forged = object.__new__(Forged)
    with pytest.raises(ContractProblem) as caught:
        adapt_solve_request(forged)

    assert caught.value.code == "INTERNAL_ERROR"


def test_json_parser_does_not_convert_process_control_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    import poker_knight_ng_service.adapter as adapter

    def interrupt(*args: object, **kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(adapter.json, "loads", interrupt)

    with pytest.raises(KeyboardInterrupt):
        adapter.adapt_solve_request(admitted_request(valid_body()))


@pytest.mark.parametrize("failure", [ValueError, RuntimeError])
def test_unexpected_adapter_fault_is_not_misclassified_as_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
    failure: type[Exception],
) -> None:
    import poker_knight_ng_service.adapter as adapter

    def fail(*_: object, **__: object) -> object:
        raise failure("unexpected adapter fault")

    monkeypatch.setattr(adapter.json, "loads", fail)

    with pytest.raises(failure, match="unexpected adapter fault"):
        adapter.adapt_solve_request(admitted_request(valid_body()))


@pytest.mark.parametrize("malformation", ["missing", "target-list", "headers-object"])
def test_forged_exact_admitted_request_maps_to_internal_error(
    malformation: str,
) -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service.adapter import adapt_solve_request
    from poker_knight_ng_service.framing import AdmittedRequest

    forged = object.__new__(AdmittedRequest)
    if malformation != "missing":
        object.__setattr__(forged, "method", b"POST")
        object.__setattr__(
            forged,
            "target",
            [] if malformation == "target-list" else b"/v1/solve",
        )
        object.__setattr__(
            forged,
            "headers",
            object() if malformation == "headers-object" else (),
        )
        object.__setattr__(forged, "body", valid_body())

    with pytest.raises(ContractProblem) as caught:
        adapt_solve_request(forged)

    assert caught.value.code == "INTERNAL_ERROR"
    assert caught.value.__cause__ is None
