import json
from pathlib import Path
import subprocess
import sys

import jsonschema
import pytest

import poker_knight_ng
from poker_knight_ng import ContractProblem, EquityRequest, EquityResult


ROOT = Path(__file__).parents[2]


def request(*, backend="cpu_reference", trials=2):
    return EquityRequest(
        ("As", "Ah"),
        ("2s", "3h", "Td"),
        2,
        trials,
        0x0123_4567_89AB_CDEF,
        backend,
    )


def test_root_exports_exact_phase6_surface_and_remains_cupy_inert(tmp_path):
    expected = {
        "CPUReferenceEngine",
        "CUDAEngine",
        "ContractProblem",
        "Engine",
        "EquityRequest",
        "EquityResult",
        "canonical_case_bytes",
        "canonical_case_hash",
        "serialize_equity_result",
        "solve",
        "solve_cuda",
        "__version__",
    }
    assert set(poker_knight_ng.__all__) == expected

    poison = tmp_path / "cupy"
    poison.mkdir()
    (poison / "__init__.py").write_text("raise RuntimeError('cupy imported')\n")
    code = (
        "import sys; import poker_knight_ng as p; "
        "assert 'cupy' not in sys.modules; "
        "assert all(hasattr(p, n) for n in p.__all__)"
    )
    environment = {"PYTHONPATH": str(tmp_path)}
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_solve_cuda_validates_route_before_lazy_engine_construction(monkeypatch):
    from poker_knight_ng import solve_cuda
    import poker_knight_ng.engine as engine

    calls = []

    class FakeEngine:
        def __init__(self):
            calls.append("constructed")

        def solve(self, value):
            calls.append(value)
            return "cuda-result"

    monkeypatch.setattr(engine, "CUDAEngine", FakeEngine)

    with pytest.raises(ContractProblem, match="UNSUPPORTED_REQUEST"):
        solve_cuda(request())
    assert calls == []

    cuda_request = request(backend="cuda")
    assert solve_cuda(cuda_request) == "cuda-result"
    assert calls == ["constructed", cuda_request]


def test_engine_protocol_accepts_both_public_engines_without_runtime_probe():
    from poker_knight_ng import CPUReferenceEngine, CUDAEngine, Engine

    assert isinstance(CPUReferenceEngine(), Engine)
    assert isinstance(CUDAEngine(runtime=object()), Engine)


def test_solve_cuda_rejects_nonexact_and_forged_requests_without_constructing(monkeypatch):
    from poker_knight_ng import solve_cuda
    import poker_knight_ng.engine as engine

    monkeypatch.setattr(
        engine,
        "CUDAEngine",
        lambda: pytest.fail("must not construct CUDA engine"),
    )

    class RequestSubclass(EquityRequest):
        pass

    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        solve_cuda(RequestSubclass(*request(backend="cuda").__dict__.values()))

    forged = request(backend="cuda")
    object.__setattr__(forged, "validate", lambda: None)
    object.__setattr__(forged, "backend", "cpu_reference")
    with pytest.raises(ContractProblem, match="UNSUPPORTED_REQUEST"):
        solve_cuda(forged)


def test_solve_remains_cpu_only_and_cuda_unavailable(monkeypatch):
    from poker_knight_ng import solve
    import poker_knight_ng.engine.local as local

    monkeypatch.setattr(
        local,
        "run_cpu_monte_carlo",
        lambda **_: pytest.fail("CUDA request must not execute CPU"),
    )
    with pytest.raises(ContractProblem, match="BACKEND_UNAVAILABLE"):
        solve(request(backend="cuda"))


def test_request_bound_serializer_emits_complete_schema_valid_wire():
    from poker_knight_ng import serialize_equity_result, solve

    source_request = request(trials=3)
    result = solve(source_request)
    raw = serialize_equity_result(result, source_request)

    schema = json.loads(
        (ROOT / "contracts/v1/equity-result.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(raw)
    assert raw["backend"] == "cpu_reference"
    assert raw["case_hash"] == result.case_hash
    assert raw["seed"] == f"0x{source_request.seed:016x}"
    assert raw["completed_trials"] == "3"
    assert raw["provenance"]["device_id"] is None
    assert raw["provenance"]["kernel_id"] is None


def test_request_bound_serializer_preserves_valid_cuda_provenance():
    from poker_knight_ng import serialize_equity_result
    from poker_knight_ng.engine.result import to_equity_result
    from poker_knight_ng.reference.monte_carlo import run_cpu_monte_carlo
    from poker_knight_ng.contract import canonical_case_hash

    source_request = request(backend="cuda", trials=2)
    aggregate = run_cpu_monte_carlo(
        seed=source_request.seed,
        hero_card_ids=(12, 25),
        board_card_ids=(0, 14, 34),
        opponent_count=source_request.opponent_count,
        requested_trials=source_request.requested_trials,
        replay_case_hash=bytes.fromhex(canonical_case_hash(source_request)),
    )
    result = to_equity_result(
        aggregate,
        source_request,
        7,
        provenance=(
            "cuda-deterministic-v1",
            "cuda-uuid:" + "ab" * 16,
            "cuda-source-sha256:" + "cd" * 32,
        ),
    )
    raw = serialize_equity_result(result, source_request)
    assert raw["backend"] == "cuda"
    assert raw["timing"] == {"total_duration_ns": "7"}
    assert raw["provenance"] == {
        "engine_build_id": "poker-knight-ng-0.1.0",
        "backend_qualification": "cuda-deterministic-v1",
        "device_id": "cuda-uuid:" + "ab" * 16,
        "kernel_id": "cuda-source-sha256:" + "cd" * 32,
    }


def test_serializer_fails_closed_on_wrong_type_request_mismatch_and_mutation():
    from poker_knight_ng import serialize_equity_result, solve

    source_request = request()
    result = solve(source_request)
    cases = (
        (object(), source_request),
        (result, object()),
        (result, request(backend="cuda")),
    )
    for candidate_result, candidate_request in cases:
        with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
            serialize_equity_result(candidate_result, candidate_request)

    object.__setattr__(result, "completed_trials", 99)
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        serialize_equity_result(result, source_request)


def test_serializer_maps_exception_but_preserves_baseexception():
    from poker_knight_ng import serialize_equity_result, solve

    source_request = request()
    ordinary = solve(source_request)
    object.__setattr__(ordinary, "seed", object())
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        serialize_equity_result(ordinary, source_request)

    class InterruptingSeed:
        def __format__(self, _format):
            raise KeyboardInterrupt

    interrupted = solve(source_request)
    object.__setattr__(interrupted, "seed", InterruptingSeed())
    with pytest.raises(KeyboardInterrupt):
        serialize_equity_result(interrupted, source_request)
