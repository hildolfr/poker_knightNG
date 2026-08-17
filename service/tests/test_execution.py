"""Exact admitted engine-execution boundary tests."""
from __future__ import annotations

import pytest


def _adapted(*, backend: str = "cpu_reference", target: bytes = b"/v1/solve"):
    from poker_knight_ng_service.adapter import adapt_solve_request
    from poker_knight_ng_service.framing import AdmittedRequest

    body = (
        b'{"backend":"'
        + backend.encode("ascii")
        + b'","board_cards":[],"contract_version":"v1",'
        b'"hero_cards":["As","Kd"],"opponent_count":"1",'
        b'"requested_trials":"1",'
        b'"rng":{"algorithm_id":"poker-knight-ng/philox4x32-10",'
        b'"algorithm_version":"1"},"seed":"0x0000000000000001"}'
    )
    return adapt_solve_request(
        AdmittedRequest(
            method=b"POST",
            target=target,
            headers=((b"content-type", b"application/json"),),
            body=body,
        )
    )


def test_cpu_engine_is_constructed_only_after_global_admission(monkeypatch) -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng.engine.local import CPUReferenceEngine as RealCPUReferenceEngine
    from poker_knight_ng_service import execution
    from poker_knight_ng_service.admission import admit_solve

    clocks = iter((10, 11))
    real_engine = RealCPUReferenceEngine(clock_ns=lambda: next(clocks))
    observed: list[str] = []

    class FakeCPUReferenceEngine:
        def __init__(self) -> None:
            try:
                admit_solve()
            except ContractProblem as exc:
                observed.append(exc.code)
            else:
                pytest.fail("engine constructed before global solve admission")

        def solve(self, request):
            return real_engine.solve(request)

    monkeypatch.setattr(execution, "CPUReferenceEngine", FakeCPUReferenceEngine)

    payload = execution.execute_solve(_adapted())

    assert observed == ["RESOURCE_EXHAUSTED"]
    assert payload["backend"] == "cpu_reference"
    assert payload["completed_trials"] == "1"


def test_cuda_route_constructs_only_cuda_engine_while_admitted(monkeypatch) -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service import execution
    from poker_knight_ng_service.admission import admit_solve

    sentinel = object()
    calls: list[object] = []

    class FakeCUDAEngine:
        def __init__(self) -> None:
            with pytest.raises(ContractProblem) as caught:
                admit_solve()
            assert caught.value.code == "RESOURCE_EXHAUSTED"

        def solve(self, request):
            calls.append(request)
            return sentinel

    class ForbiddenCPU:
        def __init__(self) -> None:
            pytest.fail("CPU engine constructed for CUDA route")

    def serialize(result, request):
        assert result is sentinel
        assert request is calls[0]
        return {"backend": "cuda"}

    monkeypatch.setattr(execution, "CPUReferenceEngine", ForbiddenCPU)
    monkeypatch.setattr(execution, "CUDAEngine", FakeCUDAEngine)
    monkeypatch.setattr(execution, "serialize_equity_result", serialize)

    assert (
        execution.execute_solve(_adapted(backend="cuda", target=b"/v1/solve-cuda"))
        == {"backend": "cuda"}
    )
    assert len(calls) == 1


def test_auto_route_constructs_cuda_engine_while_admitted(monkeypatch) -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service import execution
    from poker_knight_ng_service.admission import admit_solve

    sentinel = object()
    calls: list[object] = []

    class FakeCUDAEngine:
        def __init__(self) -> None:
            with pytest.raises(ContractProblem) as caught:
                admit_solve()
            assert caught.value.code == "RESOURCE_EXHAUSTED"

        def solve(self, request):
            calls.append(request)
            return sentinel

    class ForbiddenCPU:
        def __init__(self) -> None:
            pytest.fail("CPU engine constructed for auto cuda backend")

    def serialize(result, request):
        assert result is sentinel
        assert object.__getattribute__(request, "backend") == "cuda"
        return {"backend": "cuda"}

    monkeypatch.setattr(execution, "CPUReferenceEngine", ForbiddenCPU)
    monkeypatch.setattr(execution, "CUDAEngine", FakeCUDAEngine)
    monkeypatch.setattr(execution, "serialize_equity_result", serialize)

    assert execution.execute_solve(_adapted(backend="cuda")) == {"backend": "cuda"}
    assert len(calls) == 1


def test_busy_rejection_occurs_before_engine_construction(monkeypatch) -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service import execution
    from poker_knight_ng_service.admission import admit_solve

    constructed = False

    class ForbiddenEngine:
        def __init__(self) -> None:
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(execution, "CPUReferenceEngine", ForbiddenEngine)
    held = admit_solve()
    try:
        with pytest.raises(ContractProblem) as caught:
            execution.execute_solve(_adapted())
        assert caught.value.code == "RESOURCE_EXHAUSTED"
        assert not constructed
    finally:
        held.release()


@pytest.mark.parametrize("forged", [False, True])
def test_internal_worker_rejects_inactive_lease_before_construction(
    monkeypatch,
    forged: bool,
) -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service import execution
    from poker_knight_ng_service.admission import SolveLease, admit_solve

    constructed = False

    class ForbiddenEngine:
        def __init__(self) -> None:
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(execution, "CPUReferenceEngine", ForbiddenEngine)
    if forged:
        lease = object.__new__(SolveLease)
    else:
        lease = admit_solve()
        lease.release()

    with pytest.raises(ContractProblem) as caught:
        execution._execute_admitted(_adapted(), lease)
    assert caught.value.code == "INTERNAL_ERROR"
    assert not constructed


@pytest.mark.parametrize(
    "code",
    [
        "BACKEND_UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "RNG_REJECTION_EXHAUSTED",
        "INTERNAL_ERROR",
    ],
)
def test_closed_engine_failure_codes_are_preserved_and_release_lease(monkeypatch, code: str) -> None:
    from poker_knight_ng.contract.errors import ContractProblem, problem
    from poker_knight_ng_service import execution
    from poker_knight_ng_service.admission import admit_solve

    failure = problem(code)

    class FailingEngine:
        def solve(self, request) -> None:
            raise failure

    monkeypatch.setattr(execution, "CPUReferenceEngine", FailingEngine)

    with pytest.raises(ContractProblem) as caught:
        execution.execute_solve(_adapted())
    assert caught.value is failure

    lease = admit_solve()
    lease.release()


def test_unexpected_engine_problem_and_ordinary_fault_map_internal(monkeypatch) -> None:
    from poker_knight_ng.contract.errors import ContractProblem, problem
    from poker_knight_ng_service import execution

    failures = (problem("INVALID_CARD"), RuntimeError("private engine detail"))
    for failure in failures:
        class FailingEngine:
            def solve(self, request) -> None:
                raise failure

        monkeypatch.setattr(execution, "CPUReferenceEngine", FailingEngine)
        with pytest.raises(ContractProblem) as caught:
            execution.execute_solve(_adapted())
        assert caught.value.code == "INTERNAL_ERROR"
        assert caught.value.__cause__ is failure


def test_result_serialization_failure_is_always_internal(monkeypatch) -> None:
    from poker_knight_ng.contract.errors import ContractProblem, problem
    from poker_knight_ng_service import execution

    class Engine:
        def solve(self, request):
            return object()

    failure = problem("BACKEND_UNAVAILABLE")
    monkeypatch.setattr(execution, "CPUReferenceEngine", Engine)
    monkeypatch.setattr(
        execution,
        "serialize_equity_result",
        lambda result, request: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(ContractProblem) as caught:
        execution.execute_solve(_adapted())
    assert caught.value.code == "INTERNAL_ERROR"
    assert caught.value.__cause__ is failure


def test_forged_adapted_route_fails_before_admission(monkeypatch) -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service import execution
    from poker_knight_ng_service.admission import admit_solve
    from poker_knight_ng_service.routing import Route

    adapted = _adapted()
    object.__setattr__(adapted, "route", Route.HEALTH)
    held = admit_solve()
    try:
        with pytest.raises(ContractProblem) as caught:
            execution.execute_solve(adapted)
        assert caught.value.code == "INTERNAL_ERROR"
    finally:
        held.release()


@pytest.mark.parametrize("stage", ["construct", "solve", "serialize"])
def test_base_exception_propagates_and_releases_lease(monkeypatch, stage: str) -> None:
    from poker_knight_ng_service import execution
    from poker_knight_ng_service.admission import admit_solve

    class Stop(BaseException):
        pass

    class Engine:
        def __init__(self) -> None:
            if stage == "construct":
                raise Stop

        def solve(self, request):
            if stage == "solve":
                raise Stop
            return object()

    def serialize(result, request):
        if stage == "serialize":
            raise Stop
        return {}

    monkeypatch.setattr(execution, "CPUReferenceEngine", Engine)
    monkeypatch.setattr(execution, "serialize_equity_result", serialize)

    with pytest.raises(Stop):
        execution.execute_solve(_adapted())

    lease = admit_solve()
    lease.release()
