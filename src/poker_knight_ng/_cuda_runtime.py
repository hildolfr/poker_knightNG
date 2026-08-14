"""Private, lazy, source-attested CuPy launcher for deterministic CUDA kernels.

This module is deliberately not an engine backend.  Importing it neither imports
CuPy nor touches a CUDA driver; construction validates only host configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import operator
from pathlib import Path
import re
import struct
from typing import Any, Final, cast

from .reference.monte_carlo import MonteCarloResult

UINT64_MAX: Final = (1 << 64) - 1
UINT32_MAX: Final = (1 << 32) - 1
THREADS: Final = 128
CUPY_VERSION: Final = "14.1.1"
MIN_FREE_VRAM_BYTES: Final = 2 << 30
MAX_ALLOCATION_BUDGET_BYTES: Final = 2 << 30
BATCH_OVERHEAD_BYTES: Final = 4096
AGGREGATE_BYTES: Final = 192
AGGREGATE_DTYPE: Final = [
    ("status", "u1"), ("_padding0", "u1", 7),
    ("completed_trials", "<u8"), ("unique_wins", "<u8"),
    ("tie_by_other_winners", "<u8", 6), ("losses", "<u8"),
    ("equity_share_units", "<u8"), ("hero_category_counts", "<u8", 9),
    ("rejection_lo", "<u8"), ("rejection_hi", "<u8"),
    ("failure_simulation_id", "<u8"), ("failure_draw_slot", "<u4"),
    ("_padding1", "u1", 4),
]
_INCLUDE = re.compile(r'^\s*#include\s+"([^"/]+)"\s*$', re.MULTILINE)
_SYSTEM_INCLUDE = re.compile(r'^\s*#include\s+<([^>]+)>\s*$', re.MULTILINE)
_MODULE_CACHE: dict[tuple[str, str, tuple[str, ...]], Any] = {}
APPROVED_SOURCE_NAMES: Final = (
    "philox.cuh",
    "dealer.cuh",
    "cards.cuh",
    "evaluator.cuh",
    "simulate.cuh",
    "reduce.cuh",
    "deterministic_kernels.cu",
)
APPROVED_SOURCE_SHA256: Final = "b3bc54c703bef1fb480462a41c5535eaa9ff56f82d5c34592d845553a12fdab1"
_NVRTC_PRELUDE: Final = r'''
namespace std {
using uint8_t = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;
}
#define UINT32_MAX 0xffffffffU
#define UINT64_MAX 0xffffffffffffffffULL
'''


class CudaRuntimeError(RuntimeError):
    """A CUDA runtime, source, or ABI invariant was not met."""


class CudaRngExhausted(CudaRuntimeError):
    """The deterministic native trial reported a bounded RNG exhaustion."""


def empty_aggregate_record() -> dict[str, object]:
    return {"status": 0, "completed_trials": 0, "unique_wins": 0,
            "tie_by_other_winners": [0] * 6, "losses": 0,
            "equity_share_units": 0, "hero_category_counts": [0] * 9,
            "rejection_lo": 0, "rejection_hi": 0,
            "failure_simulation_id": UINT64_MAX,
            "failure_draw_slot": UINT32_MAX}


def _as_record(value: object) -> Any:
    required = tuple(empty_aggregate_record())
    candidate: Any = value
    raw: bytes | None = None
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    else:
        try:
            if (
                getattr(candidate, "shape") == (AGGREGATE_BYTES,)
                and candidate.dtype.itemsize == 1
                and candidate.dtype.kind in ("u", "i")
            ):
                raw = candidate.tobytes(order="C")
        except (AttributeError, TypeError):
            pass
    if raw is not None:
        if len(raw) != AGGREGATE_BYTES:
            raise CudaRuntimeError("aggregate ABI byte buffer has wrong size")
        record = empty_aggregate_record()
        record["status"] = raw[0]
        record["completed_trials"] = struct.unpack_from("<Q", raw, 8)[0]
        record["unique_wins"] = struct.unpack_from("<Q", raw, 16)[0]
        record["tie_by_other_winners"] = list(struct.unpack_from("<6Q", raw, 24))
        record["losses"] = struct.unpack_from("<Q", raw, 72)[0]
        record["equity_share_units"] = struct.unpack_from("<Q", raw, 80)[0]
        record["hero_category_counts"] = list(struct.unpack_from("<9Q", raw, 88))
        record["rejection_lo"], record["rejection_hi"] = struct.unpack_from("<2Q", raw, 160)
        record["failure_simulation_id"] = struct.unpack_from("<Q", raw, 176)[0]
        record["failure_draw_slot"] = struct.unpack_from("<I", raw, 184)[0]
        return record
    if isinstance(value, dict):
        if set(value) == set(required):
            return value
    raise CudaRuntimeError("aggregate ABI record has wrong shape or dtype")


def _unsigned(value: Any, bits: int, name: str) -> int:
    if isinstance(value, bool):
        raise CudaRuntimeError(f"{name} must be a {bits}-bit unsigned integer")
    try:
        number = operator.index(value)
    except TypeError as exc:
        raise CudaRuntimeError(f"{name} must be a {bits}-bit unsigned integer") from exc
    if number < 0 or number >= 1 << bits:
        raise CudaRuntimeError(f"{name} must be a {bits}-bit unsigned integer")
    return number


def _unsigned_vector(value: Any, width: int, name: str) -> tuple[int, ...]:
    if type(value) not in (list, tuple) or len(value) != width:
        raise CudaRuntimeError(f"{name} must contain exactly {width} unsigned integers")
    return tuple(_unsigned(item, 64, f"{name}[{index}]") for index, item in enumerate(value))


def decode_aggregate(value: object) -> MonteCarloResult:
    record = _as_record(value)
    status = _unsigned(record["status"], 8, "status")
    completed = _unsigned(record["completed_trials"], 64, "completed_trials")
    wins = _unsigned(record["unique_wins"], 64, "unique_wins")
    ties = cast(tuple[int, int, int, int, int, int], _unsigned_vector(record["tie_by_other_winners"], 6, "tie_by_other_winners"))
    losses = _unsigned(record["losses"], 64, "losses")
    units = _unsigned(record["equity_share_units"], 64, "equity_share_units")
    categories = cast(tuple[int, int, int, int, int, int, int, int, int], _unsigned_vector(record["hero_category_counts"], 9, "hero_category_counts"))
    rejection_lo = _unsigned(record["rejection_lo"], 64, "rejection_lo")
    rejection_hi = _unsigned(record["rejection_hi"], 64, "rejection_hi")
    simulation_id = _unsigned(record["failure_simulation_id"], 64, "failure_simulation_id")
    draw_slot = _unsigned(record["failure_draw_slot"], 32, "failure_draw_slot")
    if status == 1:
        if simulation_id == UINT64_MAX or draw_slot > 16:
            raise CudaRuntimeError("invalid RNG exhaustion identity")
        raise CudaRngExhausted(f"RNG rejection exhausted: simulation_id={simulation_id}, draw_slot={draw_slot}")
    if status == 2:
        raise CudaRuntimeError("native aggregate reports invalid input")
    if status == 3:
        raise CudaRuntimeError("native aggregate counter overflow")
    if status != 0:
        raise CudaRuntimeError(f"unknown aggregate status: {status}")
    if simulation_id != UINT64_MAX or draw_slot != UINT32_MAX:
        raise CudaRuntimeError("successful aggregate contains failure identity")
    return MonteCarloResult(
        completed_trials=completed,
        unique_wins=wins,
        tie_by_other_winners=ties,
        losses=losses,
        equity_share_units=units,
        hero_category_counts=categories,
        rejection_count=(rejection_hi << 64) | rejection_lo,
    )


def validated_aggregate(
    value: object, *, opponents: int, requested_trials: int, board_count: int
) -> MonteCarloResult:
    """Decode and validate one authoritative batch/final native aggregate."""
    result = decode_aggregate(value)
    try:
        MonteCarloResult.validate(result, opponents, requested_trials, board_count)
    except Exception as exc:
        raise CudaRuntimeError("native aggregate validation failed") from exc
    return result


def _record_from_result(result: MonteCarloResult) -> dict[str, object]:
    rejection = object.__getattribute__(result, "rejection_count")
    record = empty_aggregate_record()
    for name in ("completed_trials", "unique_wins", "losses", "equity_share_units"):
        record[name] = object.__getattribute__(result, name)
    record["tie_by_other_winners"] = list(object.__getattribute__(result, "tie_by_other_winners"))
    record["hero_category_counts"] = list(object.__getattribute__(result, "hero_category_counts"))
    record["rejection_lo"], record["rejection_hi"] = rejection & UINT64_MAX, rejection >> 64
    return record


def _source_directory() -> Path:
    return Path(str(files("poker_knight_ng").joinpath("cuda-sources")))


def _read_approved_source_snapshot() -> tuple[tuple[Path, bytes], ...]:
    """Read and close the exact approved local include graph once."""
    directory = _source_directory()
    snapshot: dict[str, bytes] = {}
    for name in APPROVED_SOURCE_NAMES:
        path = directory / name
        try:
            snapshot[name] = path.read_bytes()
        except OSError as exc:
            raise CudaRuntimeError(f"approved CUDA source is unreadable: {name}") from exc

    seen: set[str] = set()
    ordered: list[str] = []

    def visit(name: str) -> None:
        if name in seen:
            return
        if Path(name).name != name or name not in snapshot:
            raise CudaRuntimeError("CUDA source contains a non-approved local include")
        seen.add(name)
        try:
            text = snapshot[name].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CudaRuntimeError(f"approved CUDA source is not UTF-8: {name}") from exc
        for child in _INCLUDE.findall(text):
            visit(child)
        ordered.append(name)

    visit("deterministic_kernels.cu")
    if tuple(ordered) != APPROVED_SOURCE_NAMES:
        raise CudaRuntimeError("CUDA source closure does not match the approved manifest")
    return tuple((directory / name, snapshot[name]) for name in APPROVED_SOURCE_NAMES)


def _approved_snapshot_digest(snapshot: tuple[tuple[Path, bytes], ...]) -> str:
    digest = hashlib.sha256()
    for path, data in snapshot:
        digest.update(path.name.encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
    actual = digest.hexdigest()
    if actual != APPROVED_SOURCE_SHA256:
        raise CudaRuntimeError("approved CUDA source digest does not match the pinned manifest")
    return actual


def approved_source_manifest() -> tuple[tuple[Path, str], ...]:
    """Return the pinned transitive source closure from one byte snapshot."""
    snapshot = _read_approved_source_snapshot()
    _approved_snapshot_digest(snapshot)
    return tuple((path, hashlib.sha256(data).hexdigest()) for path, data in snapshot)


def approved_source_digest() -> str:
    return _approved_snapshot_digest(_read_approved_source_snapshot())


def _compiler_environment(cp: Any) -> tuple[str, tuple[str, ...]]:
    if getattr(cp, "__version__", None) != CUPY_VERSION:
        raise CudaRuntimeError(f"exact CuPy {CUPY_VERSION} is required")
    try:
        architecture = cp.cuda.Device().compute_capability
    except Exception as exc:
        raise CudaRuntimeError("CUDA device compute capability is unavailable") from exc
    if type(architecture) is not str or re.fullmatch(r"[0-9]+", architecture) is None:
        raise CudaRuntimeError("CUDA compute capability has an unsupported shape")
    return architecture, ("-std=c++17", f"--gpu-architecture=compute_{architecture}")


def compiler_cache_key(cp: Any) -> tuple[str, str, tuple[str, ...]]:
    """Return the repository-owned semantic compiler cache key."""
    architecture, options = _compiler_environment(cp)
    return approved_source_digest(), architecture, options


def _merge(left: Any, right: Any) -> dict[str, object]:
    """Host mirror of the checked aggregate algebra used to close batches."""
    left, right = _as_record(left), _as_record(right)
    out = empty_aggregate_record()
    statuses = (int(left["status"]), int(right["status"]))
    if any(status not in (0, 1, 2, 3) for status in statuses):
        out["status"] = 2
        return out
    out["status"] = max(statuses)
    overflow = out["status"] == 3
    for name in ("completed_trials", "unique_wins", "losses", "equity_share_units"):
        total = int(left[name]) + int(right[name])
        if total > UINT64_MAX: total, overflow = UINT64_MAX, True
        out[name] = total
    for name, width in (("tie_by_other_winners", 6), ("hero_category_counts", 9)):
        values = out[name]
        assert isinstance(values, list)
        for i in range(width):
            total = int(left[name][i]) + int(right[name][i])
            if total > UINT64_MAX: total, overflow = UINT64_MAX, True
            values[i] = total
    rejection = ((int(left["rejection_hi"]) << 64) | int(left["rejection_lo"])) + ((int(right["rejection_hi"]) << 64) | int(right["rejection_lo"]))
    if rejection > (1 << 128) - 1: rejection, overflow = (1 << 128) - 1, True
    out["rejection_lo"], out["rejection_hi"] = rejection & UINT64_MAX, rejection >> 64
    failures = [(int(item["failure_simulation_id"]), int(item["failure_draw_slot"])) for item in (left, right)]
    failures = [item for item in failures if item[0] != UINT64_MAX and item[1] <= 16]
    if failures: out["failure_simulation_id"], out["failure_draw_slot"] = min(failures)
    if overflow: out["status"] = 3
    return out


_ABI_PROBE = r'''
using poker_knight_ng::cuda::AggregateResult;
extern "C" __global__ void pkng_aggregate_abi_probe(unsigned long long* out) {
 if (blockIdx.x || threadIdx.x || !out) return;
 AggregateResult value{};
 const unsigned char* base = reinterpret_cast<const unsigned char*>(&value);
 out[0] = sizeof(AggregateResult); out[1] = alignof(AggregateResult);
 out[2] = reinterpret_cast<const unsigned char*>(&value.status) - base;
 out[3] = reinterpret_cast<const unsigned char*>(&value.completed_trials) - base;
 out[4] = reinterpret_cast<const unsigned char*>(&value.unique_wins) - base;
 out[5] = reinterpret_cast<const unsigned char*>(&value.tie_by_other_winners) - base;
 out[6] = reinterpret_cast<const unsigned char*>(&value.losses) - base;
 out[7] = reinterpret_cast<const unsigned char*>(&value.equity_share_units) - base;
 out[8] = reinterpret_cast<const unsigned char*>(&value.hero_category_counts) - base;
 out[9] = reinterpret_cast<const unsigned char*>(&value.rejection_count) - base;
 out[10] = reinterpret_cast<const unsigned char*>(&value.failure_simulation_id) - base;
 out[11] = reinterpret_cast<const unsigned char*>(&value.failure_draw_slot) - base;
}'''
_ABI_EXPECTED: Final = (192, 8, 0, 8, 16, 24, 72, 80, 88, 160, 176, 184)


def _nvrtc_source_snapshot_with_digest() -> tuple[str, str]:
    byte_snapshot = _read_approved_source_snapshot()
    source_digest = _approved_snapshot_digest(byte_snapshot)
    snapshot = {path.name: data.decode("utf-8") for path, data in byte_snapshot}

    def inline(name: str) -> str:
        try:
            text = snapshot[name]
        except KeyError as exc:
            raise CudaRuntimeError("CUDA source closure changed during admission") from exc
        system_includes = set(_SYSTEM_INCLUDE.findall(text))
        if system_includes - {"cstdint"}:
            raise CudaRuntimeError("CUDA source contains an unsupported system include")
        text = _SYSTEM_INCLUDE.sub("", text)
        return _INCLUDE.sub(lambda match: inline(match.group(1)), text)

    expanded = inline("deterministic_kernels.cu")
    source = (
        _NVRTC_PRELUDE
        + f'\n#define PKNG_SOURCE_BUNDLE_SHA256 "{source_digest}"\n'
        + expanded
        + _ABI_PROBE
    )
    return source, source_digest


def nvrtc_source_snapshot() -> str:
    """Return the exact freestanding translation unit submitted to NVRTC."""
    return _nvrtc_source_snapshot_with_digest()[0]


@dataclass
class CupyDeterministicRuntime:
    batch_blocks: int = 256
    vram_budget_bytes: int | None = None
    _cp: Any = None
    _module: Any = None
    _source_digest: str | None = None
    _abi_verified: bool = False

    def __post_init__(self) -> None:
        if type(self.batch_blocks) is not int or self.batch_blocks < 1:
            raise ValueError("batch_blocks must be a positive built-in integer")
        if self.vram_budget_bytes is not None and (type(self.vram_budget_bytes) is not int or not 1 <= self.vram_budget_bytes <= MAX_ALLOCATION_BUDGET_BYTES):
            raise ValueError("vram_budget_bytes must be a positive built-in integer no greater than 2 GiB or None")

    def _cupy(self) -> Any:
        if self._cp is None:
            try:
                import cupy  # type: ignore[import-not-found]
            except Exception as exc:
                raise CudaRuntimeError("CuPy is unavailable") from exc
            self._cp = cupy
        if getattr(self._cp, "__version__", None) != CUPY_VERSION:
            raise CudaRuntimeError(f"exact CuPy {CUPY_VERSION} is required")
        return self._cp

    def _compile(self) -> Any:
        cp = self._cupy()
        source, source_digest = _nvrtc_source_snapshot_with_digest()
        architecture, options = _compiler_environment(cp)
        cache_key = (source_digest, architecture, options)
        cached = _MODULE_CACHE.get(cache_key)
        if cached is not None:
            self._source_digest = cache_key[0]
            return cached
        # The digest changes the submitted source bytes as well as our cache key,
        # preventing CuPy's own source cache from reusing stale included headers.
        names = (
            "pkng_simulate_block_partials_kernel",
            "pkng_reduce_block_partials_kernel",
            "pkng_aggregate_abi_probe",
        )
        try:
            module = cp.RawModule(code=source, options=cache_key[2], backend="nvrtc")
            for name in names:
                module.get_function(name)
        except Exception as exc:
            raise CudaRuntimeError("failed to compile source-attested deterministic CUDA source") from exc
        _MODULE_CACHE[cache_key] = module
        self._source_digest = source_digest
        return module

    def _probe_abi(self) -> None:
        if self._abi_verified:
            return
        cp = self._cupy()
        try:
            output = cp.empty(len(_ABI_EXPECTED), dtype=cp.uint64)
            self._module.get_function("pkng_aggregate_abi_probe")((1,), (1,), (output,))
            cp.cuda.runtime.deviceSynchronize()
            actual = tuple(int(x) for x in cp.asnumpy(output))
        except Exception as exc:
            raise CudaRuntimeError("failed to probe CUDA aggregate ABI") from exc
        if actual != _ABI_EXPECTED:
            raise CudaRuntimeError("CUDA aggregate ABI does not match the host layout")
        self._abi_verified = True

    def _kernels(self) -> tuple[Any, Any]:
        if self._module is None:
            self._module = self._compile()
            self._probe_abi()
        return (self._module.get_function("pkng_simulate_block_partials_kernel"), self._module.get_function("pkng_reduce_block_partials_kernel"))

    def _batch_capacity(self) -> int:
        free, _total = self._cupy().cuda.runtime.memGetInfo()
        free = int(free)
        if free < MIN_FREE_VRAM_BYTES:
            raise CudaRuntimeError("CUDA admission requires at least 2 GiB free VRAM")
        allowed = min(
            free,
            MAX_ALLOCATION_BUDGET_BYTES,
            self.vram_budget_bytes
            if self.vram_budget_bytes is not None
            else MAX_ALLOCATION_BUDGET_BYTES,
        )
        capacity = (allowed - BATCH_OVERHEAD_BYTES - AGGREGATE_BYTES) // AGGREGATE_BYTES
        if capacity < 1:
            raise CudaRuntimeError("insufficient VRAM for one deterministic partial")
        return min(self.batch_blocks, capacity)

    @staticmethod
    def _validate_run_arguments(hero: tuple[int, int], board: tuple[int, ...], opponents: int, key: tuple[int, int], first_simulation_id: int, count: int) -> None:
        if type(hero) is not tuple or len(hero) != 2 or type(board) is not tuple or len(board) not in (0, 3, 4, 5):
            raise ValueError("hero must contain two cards and board must contain 0, 3, 4, or 5 cards")
        cards = hero + board
        if any(type(card) is not int or not 0 <= card <= 51 for card in cards) or len(set(cards)) != len(cards):
            raise ValueError("hero and board must be unique card ids")
        if tuple(sorted(hero)) != hero or tuple(sorted(board)) != board:
            raise ValueError("hero and board card ids must be strictly increasing")
        if type(opponents) is not int or not 1 <= opponents <= 6:
            raise ValueError("opponents must be an integer from 1 through 6")
        if type(key) is not tuple or len(key) != 2 or any(type(word) is not int or not 0 <= word <= UINT32_MAX for word in key):
            raise ValueError("key must contain two uint32 words")
        if type(first_simulation_id) is not int or type(count) is not int or count < 1 or first_simulation_id < 0 or first_simulation_id + count - 1 >= UINT64_MAX:
            raise ValueError("simulation range is invalid")

    def run(self, *, hero: tuple[int, int], board: tuple[int, ...], opponents: int, key: tuple[int, int], first_simulation_id: int, count: int) -> MonteCarloResult:
        self._validate_run_arguments(hero, board, opponents, key, first_simulation_id, count)
        cp = self._cupy()
        # Admission occurs before compilation/context work and is rechecked after
        # compilation plus before every batch allocation.
        self._batch_capacity()
        simulate, reduce = self._kernels()
        self._batch_capacity()
        hero_d, board_d = cp.asarray(hero, dtype=cp.uint8), cp.asarray(board if board else (0,), dtype=cp.uint8)
        aggregate, offset = empty_aggregate_record(), 0
        while offset < count:
            capacity = self._batch_capacity()
            trials = min(count - offset, capacity * THREADS)
            blocks = (trials + THREADS - 1) // THREADS
            partials = cp.empty(blocks * AGGREGATE_BYTES, dtype=cp.uint8)
            final = cp.empty(AGGREGATE_BYTES, dtype=cp.uint8)
            simulate((blocks,), (THREADS,), (hero_d, board_d, cp.uint32(len(board)), cp.uint32(opponents), cp.uint32(key[0]), cp.uint32(key[1]), cp.uint64(first_simulation_id + offset), cp.uint64(trials), partials))
            reduce((1,), (THREADS,), (partials, cp.uint64(blocks), final))
            cp.cuda.runtime.deviceSynchronize()
            batch_result = validated_aggregate(
                cp.asnumpy(final),
                opponents=opponents,
                requested_trials=trials,
                board_count=len(board),
            )
            aggregate = _merge(aggregate, _record_from_result(batch_result))
            offset += trials
        return validated_aggregate(
            aggregate,
            opponents=opponents,
            requested_trials=count,
            board_count=len(board),
        )
