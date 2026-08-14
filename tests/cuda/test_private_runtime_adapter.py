"""Contract checks for the private, lazy CuPy CUDA runtime adapter."""
from __future__ import annotations

import hashlib
import importlib
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_runtime_import_is_cupy_inert_and_exposes_native_aggregate_abi() -> None:
    sys.modules.pop("poker_knight_ng._cuda_runtime", None)
    sys.modules.pop("cupy", None)
    runtime = importlib.import_module("poker_knight_ng._cuda_runtime")

    assert "cupy" not in sys.modules
    assert runtime.AGGREGATE_BYTES == 192
    assert runtime.AGGREGATE_DTYPE[0] == ("status", "u1")
    assert runtime.AGGREGATE_DTYPE[-2] == ("failure_draw_slot", "<u4")


def test_source_manifest_hashes_exact_pinned_local_source_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = importlib.import_module("poker_knight_ng._cuda_runtime")
    manifest = runtime.approved_source_manifest()

    assert tuple(path.name for path, _digest in manifest) == (
        "philox.cuh", "dealer.cuh", "cards.cuh", "evaluator.cuh", "simulate.cuh",
        "reduce.cuh", "deterministic_kernels.cu",
    )
    for path, digest in manifest:
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    expected_digest = hashlib.sha256(
        b"".join(path.name.encode("ascii") + b"\0" + path.read_bytes() for path, _ in manifest)
    ).hexdigest()
    assert runtime.APPROVED_SOURCE_SHA256 == "d011e0f5c4db4d12fcb5240b5996f0911af7f153c7942f16772f871917ca5263"
    assert runtime.approved_source_digest() == expected_digest == runtime.APPROVED_SOURCE_SHA256

    monkeypatch.setattr(runtime, "APPROVED_SOURCE_SHA256", "0" * 64)
    with pytest.raises(runtime.CudaRuntimeError, match="approved CUDA source digest"):
        runtime.nvrtc_source_snapshot()


def test_nvrtc_admission_reads_each_approved_source_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = importlib.import_module("poker_knight_ng._cuda_runtime")
    source_dir = runtime._source_directory()
    original_open = Path.open
    reads: dict[str, int] = {}

    def counted_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        mode = args[0] if args else kwargs.get("mode", "r")
        if path.parent == source_dir and "r" in str(mode):
            reads[path.name] = reads.get(path.name, 0) + 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    runtime.nvrtc_source_snapshot()
    assert reads == {name: 1 for name in runtime.APPROVED_SOURCE_NAMES}


def test_runtime_rejects_invalid_batch_and_vram_inputs_before_cupy_import() -> None:
    runtime = importlib.import_module("poker_knight_ng._cuda_runtime")

    with pytest.raises(ValueError, match="batch_blocks"):
        runtime.CupyDeterministicRuntime(batch_blocks=0)
    with pytest.raises(ValueError, match="vram_budget_bytes"):
        runtime.CupyDeterministicRuntime(vram_budget_bytes=0)
    with pytest.raises(ValueError, match="vram_budget_bytes"):
        runtime.CupyDeterministicRuntime(vram_budget_bytes=(2 << 30) + 1)


def test_exact_cupy_version_and_compiler_cache_key_are_closed() -> None:
    runtime = importlib.import_module("poker_knight_ng._cuda_runtime")
    wrong = runtime.CupyDeterministicRuntime(_cp=SimpleNamespace(__version__="14.1.0"))
    with pytest.raises(runtime.CudaRuntimeError, match="CuPy 14.1.1"):
        wrong._cupy()

    device = SimpleNamespace(compute_capability="120")
    cp = SimpleNamespace(__version__="14.1.1", cuda=SimpleNamespace(Device=lambda: device))
    digest, architecture, options = runtime.compiler_cache_key(cp)
    assert digest == runtime.approved_source_digest()
    assert architecture == "120"
    assert options == ("-std=c++17", "--gpu-architecture=compute_120")
    assert "--use_fast_math" not in options
    assert runtime._ABI_EXPECTED == (192, 8, 0, 8, 16, 24, 72, 80, 88, 160, 176, 184)


def test_nvrtc_source_is_freestanding_and_uses_direct_c_kernel_lookup() -> None:
    runtime = importlib.import_module("poker_knight_ng._cuda_runtime")
    source = runtime.nvrtc_source_snapshot()
    assert "#include <cstdint>" not in source
    assert "#include <cstddef>" not in source
    assert "offsetof(" not in source
    assert "&value.status" in source
    assert "using uint64_t = unsigned long long;" in source

    calls: dict[str, object] = {}

    class Module:
        def get_function(self, name: str) -> object:
            calls.setdefault("functions", []).append(name)  # type: ignore[union-attr]
            return object()

    class CP:
        __version__ = "14.1.1"
        cuda = SimpleNamespace(Device=lambda: SimpleNamespace(compute_capability="120"))

        @staticmethod
        def RawModule(**kwargs: object) -> Module:
            calls["raw_module"] = kwargs
            return Module()

    runtime._MODULE_CACHE.clear()
    adapter = runtime.CupyDeterministicRuntime(_cp=CP())
    adapter._compile()
    kwargs = calls["raw_module"]
    assert kwargs["backend"] == "nvrtc"  # type: ignore[index]
    assert "name_expressions" not in kwargs  # type: ignore[operator]
    assert calls["functions"] == [
        "pkng_simulate_block_partials_kernel",
        "pkng_reduce_block_partials_kernel",
        "pkng_aggregate_abi_probe",
    ]


def test_batch_admission_has_a_hard_2gib_cap_and_never_allocates_zero_blocks() -> None:
    runtime = importlib.import_module("poker_knight_ng._cuda_runtime")
    adapter = runtime.CupyDeterministicRuntime(batch_blocks=99, vram_budget_bytes=2 << 30)
    adapter._cp = type("CP", (), {"__version__": "14.1.1", "cuda": type("Cuda", (), {"runtime": type("Runtime", (), {
        "memGetInfo": staticmethod(lambda: ((3 << 30), 4 << 30)),
    })})})()
    assert adapter._batch_capacity() == 99

    adapter._cp.cuda.runtime.memGetInfo = staticmethod(lambda: ((2 << 30) - 1, 4 << 30))
    with pytest.raises(runtime.CudaRuntimeError, match="2 GiB"):
        adapter._batch_capacity()


def test_native_batch_and_final_aggregates_are_authoritatively_validated() -> None:
    runtime = importlib.import_module("poker_knight_ng._cuda_runtime")
    record = runtime.empty_aggregate_record()
    record["completed_trials"] = 1
    record["unique_wins"] = 2
    record["equity_share_units"] = 840
    record["hero_category_counts"][0] = 1
    with pytest.raises(runtime.CudaRuntimeError, match="aggregate validation"):
        runtime.validated_aggregate(record, opponents=1, requested_trials=1, board_count=0)


def test_run_validates_kernel_preconditions_before_cupy_import() -> None:
    runtime = importlib.import_module("poker_knight_ng._cuda_runtime")
    adapter = runtime.CupyDeterministicRuntime()
    with pytest.raises(ValueError, match="strictly increasing"):
        adapter.run(hero=(9, 2), board=(), opponents=1, key=(0, 0), first_simulation_id=0, count=1)
    with pytest.raises(ValueError, match="simulation range"):
        adapter.run(hero=(2, 9), board=(), opponents=1, key=(0, 0), first_simulation_id=(1 << 64) - 2, count=2)
    assert adapter._cp is None


def test_decode_aggregate_closes_native_status_and_exact_integer_fields() -> None:
    runtime = importlib.import_module("poker_knight_ng._cuda_runtime")
    value = runtime.empty_aggregate_record()
    value["status"] = 0
    value["completed_trials"] = 3
    value["unique_wins"] = 1
    value["tie_by_other_winners"][0] = 1
    value["losses"] = 1
    value["equity_share_units"] = 840
    value["hero_category_counts"][2] = 3
    value["rejection_lo"] = 9
    value["rejection_hi"] = 2
    value["failure_simulation_id"] = (1 << 64) - 1
    value["failure_draw_slot"] = (1 << 32) - 1

    decoded = runtime.decode_aggregate(value)
    assert decoded.completed_trials == 3
    assert decoded.tie_by_other_winners == (1, 0, 0, 0, 0, 0)
    assert decoded.rejection_count == (2 << 64) + 9

    value["status"] = 1
    value["failure_simulation_id"] = 4
    value["failure_draw_slot"] = 2
    with pytest.raises(runtime.CudaRngExhausted, match="simulation_id=4, draw_slot=2"):
        runtime.decode_aggregate(value)

    value["status"] = 99
    with pytest.raises(runtime.CudaRuntimeError, match="unknown aggregate status"):
        runtime.decode_aggregate(value)


def test_decode_aggregate_accepts_the_exact_raw_192_byte_device_abi() -> None:
    runtime = importlib.import_module("poker_knight_ng._cuda_runtime")
    raw = bytearray(runtime.AGGREGATE_BYTES)
    struct.pack_into("<B", raw, 0, 0)
    struct.pack_into("<Q", raw, 8, 1)
    struct.pack_into("<Q", raw, 16, 1)
    struct.pack_into("<Q", raw, 80, 840)
    struct.pack_into("<Q", raw, 88, 1)
    struct.pack_into("<Q", raw, 176, (1 << 64) - 1)
    struct.pack_into("<I", raw, 184, (1 << 32) - 1)
    decoded = runtime.decode_aggregate(raw)
    assert decoded.completed_trials == 1
    assert decoded.unique_wins == 1
    assert decoded.equity_share_units == 840


@pytest.mark.parametrize("field,value", [
    ("failure_simulation_id", -1),
    ("failure_simulation_id", 1 << 64),
    ("failure_simulation_id", True),
    ("failure_simulation_id", 1.5),
    ("failure_draw_slot", -1),
    ("failure_draw_slot", 1 << 32),
])
def test_corrupt_rng_failure_identity_is_rejected(field: str, value: object) -> None:
    runtime = importlib.import_module("poker_knight_ng._cuda_runtime")
    record = runtime.empty_aggregate_record()
    record["status"] = 1
    record["failure_simulation_id"] = 7
    record["failure_draw_slot"] = 2
    record[field] = value
    with pytest.raises(runtime.CudaRuntimeError, match="unsigned integer"):
        runtime.decode_aggregate(record)


def test_arbitrary_structured_dtype_is_not_an_aggregate_boundary() -> None:
    runtime = importlib.import_module("poker_knight_ng._cuda_runtime")

    class FakeStructured:
        shape = ()
        dtype = SimpleNamespace(fields={name: object() for name in runtime.empty_aggregate_record()})

        def __getitem__(self, _key: str) -> int:
            return 0

    with pytest.raises(runtime.CudaRuntimeError, match="wrong shape or dtype"):
        runtime.decode_aggregate(FakeStructured())
