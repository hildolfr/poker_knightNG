"""Explicit CUDA engine integration without CUDA execution."""
import hashlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from poker_knight_ng import ContractProblem, EquityRequest
from poker_knight_ng.engine import solve
from poker_knight_ng.reference.monte_carlo import MonteCarloResult


def request(*, backend="cuda"):
    return EquityRequest(("Ah", "As"), ("Td", "3h", "2s"), 2, 1, 0x1234, backend)


class Runtime:
    def __init__(self, result=None, error=None):
        self.result = result or MonteCarloResult(1, 1, (0, 0, 0, 0, 0, 0), 0, 420, (1, 0, 0, 0, 0, 0, 0, 0, 0), 0)
        self.error = error
        self.calls = []
    def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.error: raise self.error
        return self.result
    def provenance(self):
        return ("cuda-uuid:" + "ab" * 16, "cuda-source-sha256:" + "cd" * 32)


def test_cuda_engine_is_explicit_inert_and_passes_host_authoritative_inputs():
    code = "from poker_knight_ng.engine import CUDAEngine; import sys; assert 'cupy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
    from poker_knight_ng.engine import CUDAEngine
    runtime = Runtime()
    result = CUDAEngine(runtime=runtime, clock_ns=iter((10, 20)).__next__).solve(request())
    assert runtime.calls == [{"hero": (12, 25), "board": (0, 14, 34), "opponents": 2, "key": (3440389639, 1616409312), "first_simulation_id": 0, "count": 1}]
    assert result.provenance == ("poker-knight-ng-0.1.0", "cuda-deterministic-v1", "cuda-uuid:" + "ab" * 16, "cuda-source-sha256:" + "cd" * 32)
    assert result.timing == 10


def test_default_solve_still_rejects_cuda():
    with pytest.raises(ContractProblem, match="BACKEND_UNAVAILABLE"):
        solve(request())


def test_cuda_engine_maps_typed_runtime_errors():
    from poker_knight_ng._cuda_runtime import CudaBackendUnavailable, CudaResourceExhausted, CudaRngExhausted
    from poker_knight_ng.engine import CUDAEngine
    for exc, code in ((CudaBackendUnavailable(), "BACKEND_UNAVAILABLE"), (CudaResourceExhausted(), "RESOURCE_EXHAUSTED"), (CudaRngExhausted(), "RNG_REJECTION_EXHAUSTED"), (ValueError(), "INTERNAL_ERROR")):
        with pytest.raises(ContractProblem, match=code):
            CUDAEngine(runtime=Runtime(error=exc), clock_ns=iter((1, 2)).__next__).solve(request())


def test_cuda_engine_rejects_cpu_request():
    from poker_knight_ng.engine import CUDAEngine
    with pytest.raises(ContractProblem, match="UNSUPPORTED_REQUEST"):
        CUDAEngine(runtime=Runtime()).solve(request(backend="cpu_reference"))


def test_cuda_engine_uses_fixed_request_validation_and_rejects_bad_types_before_runtime():
    from poker_knight_ng.engine import CUDAEngine

    runtime = Runtime()
    forged = request()
    object.__setattr__(forged, "validate", lambda: None)
    object.__setattr__(forged, "hero_cards", ("bad", "As"))
    with pytest.raises(ContractProblem, match="INVALID_CARD"):
        CUDAEngine(runtime=runtime).solve(forged)
    assert runtime.calls == []

    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        CUDAEngine(runtime=runtime).solve(object())  # type: ignore[arg-type]
    assert runtime.calls == []


def test_runtime_reprobes_abi_when_a_module_exists_but_qualification_is_incomplete(monkeypatch):
    from poker_knight_ng._cuda_runtime import CupyDeterministicRuntime

    class Module:
        def get_function(self, name):
            return name

    adapter = CupyDeterministicRuntime(_cp=SimpleNamespace(__version__="14.1.1"))
    adapter._module = Module()
    adapter._abi_verified = False
    calls = []
    monkeypatch.setattr(adapter, "_probe_abi", lambda: (calls.append("probe"), setattr(adapter, "_abi_verified", True)))
    assert adapter._kernels() == ("pkng_simulate_block_partials_kernel", "pkng_reduce_block_partials_kernel")
    assert calls == ["probe"]


def _qualified_runtime(properties, *, compute_capability="120"):
    from poker_knight_ng._cuda_runtime import APPROVED_SOURCE_SHA256, CupyDeterministicRuntime

    class Module:
        def get_function(self, name):
            return name

    device = SimpleNamespace(id=0, compute_capability=compute_capability)
    cp = SimpleNamespace(
        __version__="14.1.1",
        cuda=SimpleNamespace(
            Device=lambda: device,
            runtime=SimpleNamespace(getDeviceProperties=lambda _device_id: properties),
        ),
    )
    return CupyDeterministicRuntime(
        _cp=cp,
        _module=Module(),
        _source_digest=APPROVED_SOURCE_SHA256,
        _abi_verified=True,
    )


def test_runtime_provenance_uses_exact_production_shaped_cupy_properties():
    runtime = _qualified_runtime({"uuid": bytes.fromhex("ab" * 16), "major": 12, "minor": 0})
    assert runtime.provenance() == (
        "cuda-uuid:" + "ab" * 16,
        "cuda-source-sha256:d011e0f5c4db4d12fcb5240b5996f0911af7f153c7942f16772f871917ca5263",
    )


@pytest.mark.parametrize("properties,capability", [
    ({"uuid": b"bad", "major": 12, "minor": 0}, "120"),
    ({"uuid": b"a" * 16, "major": True, "minor": 0}, "10"),
    ({"uuid": b"a" * 16, "major": 1, "minor": 20}, "120"),
    ({"uuid": b"a" * 16, "major": -1, "minor": 0}, "-10"),
])
def test_runtime_provenance_rejects_malformed_device_properties(properties, capability):
    from poker_knight_ng._cuda_runtime import CudaBackendUnavailable
    with pytest.raises(CudaBackendUnavailable):
        _qualified_runtime(properties, compute_capability=capability).provenance()


def test_cuda_engine_closes_malformed_runtime_result_and_provenance():
    from poker_knight_ng.engine import CUDAEngine

    malformed_result = Runtime(result=object())
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        CUDAEngine(runtime=malformed_result, clock_ns=iter((1, 2)).__next__).solve(request())

    malformed_provenance = Runtime()
    malformed_provenance.provenance = lambda: ("raw gpu name", "bad")
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        CUDAEngine(runtime=malformed_provenance, clock_ns=iter((1, 2)).__next__).solve(request())


@pytest.mark.parametrize("clocks", [(True,), (-1,), (1 << 64,), (2, 1)])
def test_cuda_engine_closes_invalid_or_backward_clocks(clocks):
    from poker_knight_ng.engine import CUDAEngine
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        CUDAEngine(runtime=Runtime(), clock_ns=iter(clocks).__next__).solve(request())


class StopSignal(BaseException):
    pass


def test_cuda_engine_propagates_process_control_from_clock_run_and_provenance():
    from poker_knight_ng.engine import CUDAEngine

    with pytest.raises(StopSignal):
        CUDAEngine(runtime=Runtime(), clock_ns=lambda: (_ for _ in ()).throw(StopSignal())).solve(request())

    runtime = Runtime(error=StopSignal())
    with pytest.raises(StopSignal):
        CUDAEngine(runtime=runtime, clock_ns=iter((1, 2)).__next__).solve(request())

    runtime = Runtime()
    runtime.provenance = lambda: (_ for _ in ()).throw(StopSignal())
    with pytest.raises(StopSignal):
        CUDAEngine(runtime=runtime, clock_ns=iter((1, 2)).__next__).solve(request())


def test_typed_runtime_errors_from_provenance_keep_public_mapping():
    from poker_knight_ng._cuda_runtime import CudaBackendUnavailable, CudaResourceExhausted, CudaRngExhausted
    from poker_knight_ng.engine import CUDAEngine

    for exc, code in (
        (CudaBackendUnavailable(), "BACKEND_UNAVAILABLE"),
        (CudaResourceExhausted(), "RESOURCE_EXHAUSTED"),
        (CudaRngExhausted(), "RNG_REJECTION_EXHAUSTED"),
    ):
        runtime = Runtime()
        runtime.provenance = lambda exc=exc: (_ for _ in ()).throw(exc)
        with pytest.raises(ContractProblem, match=code):
            CUDAEngine(runtime=runtime, clock_ns=iter((1, 2)).__next__).solve(request())


@pytest.mark.parametrize("scenario", ["unreadable", "non_utf8", "closure", "digest", "unsupported_system"])
def test_every_source_admission_failure_is_backend_unavailable_publicly(scenario, tmp_path, monkeypatch):
    import poker_knight_ng._cuda_runtime as cuda_runtime
    from poker_knight_ng.engine import CUDAEngine

    source_dir = tmp_path / "cuda-sources"
    source_dir.mkdir()
    real_dir = cuda_runtime._source_directory()
    if scenario != "unreadable":
        for name in cuda_runtime.APPROVED_SOURCE_NAMES:
            (source_dir / name).write_bytes((real_dir / name).read_bytes())
    if scenario == "non_utf8":
        (source_dir / "philox.cuh").write_bytes(b"\xff")
    elif scenario == "closure":
        kernel = source_dir / "deterministic_kernels.cu"
        kernel.write_text(kernel.read_text().replace('#include "reduce.cuh"', '#include "not-approved.cuh"'))
    elif scenario == "digest":
        path = source_dir / "philox.cuh"
        path.write_bytes(path.read_bytes() + b"\n")
    elif scenario == "unsupported_system":
        path = source_dir / "deterministic_kernels.cu"
        path.write_bytes(path.read_bytes() + b"\n#include <vector>\n")
        digest = hashlib.sha256()
        for name in cuda_runtime.APPROVED_SOURCE_NAMES:
            digest.update(name.encode("ascii")); digest.update(b"\0"); digest.update((source_dir / name).read_bytes())
        monkeypatch.setattr(cuda_runtime, "APPROVED_SOURCE_SHA256", digest.hexdigest())

    monkeypatch.setattr(cuda_runtime, "_source_directory", lambda: source_dir)
    with pytest.raises(cuda_runtime.CudaBackendUnavailable) as caught:
        cuda_runtime._nvrtc_source_snapshot_with_digest()

    with pytest.raises(ContractProblem, match="BACKEND_UNAVAILABLE"):
        CUDAEngine(runtime=Runtime(error=caught.value), clock_ns=iter((1, 2)).__next__).solve(request())
