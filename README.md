# Poker Knight NG

Poker Knight NG is a contract-first, deterministic Texas Hold'em equity engine for Python 3.13. The package has zero base runtime dependencies and provides two explicit synchronous engines:

- `CPUReferenceEngine` is the default public executor and the authoritative deterministic CPU implementation.
- `CUDAEngine` is an explicit opt-in executor backed by the qualified CuPy/CUDA runtime.

`solve()` remains CPU-only. A request for `backend="cuda"` sent through
`solve()` fails with `BACKEND_UNAVAILABLE`; it is never silently downgraded
or automatically routed to a GPU. Use `solve_cuda(request)` or
`CUDAEngine().solve(request)` only when CUDA execution is deliberately
selected.

## Python API

```python
from poker_knight_ng import (
    EquityRequest,
    serialize_equity_result,
    solve,
    solve_cuda,
)

cpu_request = EquityRequest(
    ("As", "Ah"), ("2s", "3h", "Td"), 2, 100_000,
    0x0123_4567_89AB_CDEF, "cpu_reference",
)
cpu_result = solve(cpu_request)
cpu_wire = serialize_equity_result(cpu_result, cpu_request)

cuda_request = EquityRequest(
    ("As", "Ah"), ("2s", "3h", "Td"), 2, 100_000,
    0x0123_4567_89AB_CDEF, "cuda",
)
cuda_result = solve_cuda(cuda_request)
cuda_wire = serialize_equity_result(cuda_result, cuda_request)
```

`serialize_equity_result()` is request-bound: it revalidates the complete
normalized result against the originating request before emitting the exact v1
wire object. Root import and engine construction remain CuPy-inert; CUDA is
loaded only when an explicit CUDA request executes.

## Command line

The package installs `poker-knight-ng` and also supports
`python -m poker_knight_ng`. The routes are separate and exact:

```sh
poker-knight-ng solve < request.json
poker-knight-ng solve-cuda < cuda-request.json
poker-knight-ng --help
```

The CLI reads one UTF-8 JSON object from stdin, bounded to 16,384 bytes. It
rejects a BOM, duplicate members, non-finite constants, trailing documents,
and non-object roots. Successful execution writes one canonical v1 result JSON
line to stdout. Failures write one canonical closed v1 problem JSON line to
stderr without raw exception text, submitted cards, or host paths.

| Exit | Meaning |
|---:|---|
| 0 | Result or `--help` |
| 1 | Terminal write failure |
| 2 | Invocation or JSON framing failure |
| 3 | Non-retryable request/contract failure |
| 4 | Retryable backend/resource failure |
| 5 | Internal or RNG-exhaustion failure |

There is no CLI auto-routing or fallback. `solve` with a CUDA request returns
`BACKEND_UNAVAILABLE`; `solve-cuda` with a CPU request returns
`UNSUPPORTED_REQUEST`.

## Development and verification

```sh
uv sync --frozen --group dev
uv run pytest -q
uv run python tools/generate_oracle_fixtures.py --verify
uv run python tools/qualify_seven_card_corpus.py --verify
uv run python tools/generate_rng_seed_bank.py --verify
uv run python tools/verify_cuda_release_qualification.py
uv run python tools/verify_cuda_statistical_release_qualification.py
```

Before declaring the published roadmap ready for merge, run the local,
network-free completion gate. It checks the roadmap and service contracts, CI
and release-documentation readiness, plus the checked-in deployment artifacts;
it does not start or deploy the service:

```sh
python tools/verify_roadmap_completion.py
```

The exhaustive seven-card release qualification remains deliberately opt-in:

```sh
RUN_SEVEN_CARD_RELEASE_QUALIFICATION=1 uv run python tools/qualify_seven_card_corpus.py --release
```

For an explicit CUDA environment, install the pinned optional extra:

```sh
uv sync --frozen --group dev --extra gpu-cu13
```

The approved GPU dependency is exactly `cupy-cuda13x==14.1.1`. CUDA compilation is lazy; importing the root package does not import CuPy or initialize CUDA.

## Current CUDA qualification

The qualified source checkpoint is `7fb617b900c06102caafe240ff95afe7fef2aa58`. Its exact wheel-installed run passed cold, warm-cache, and forced-PTX execution for all committed seed-bank vectors, the full 799-test JUnit run with zero failures/errors and eight skips, and Compute Sanitizer `memcheck`, `racecheck`, `initcheck`, and `synccheck` with zero findings.

The compact public record is `validation/holdem/v1/cuda_release_qualification.json`; verify it with `tools/verify_cuda_release_qualification.py`. The record binds the source closure, qualification tool, seed authority, lockfile, wheel/sdist/JUnit/sanitizer hashes, and the private canonical evidence hash without publishing host paths, process inventories, or raw device names.

Phase 5C statistical qualification passed on executed source `7fb617b900c06102caafe240ff95afe7fef2aa58`: all four preregistered batch geometries produced the exact frozen integer aggregate, exact planned kernel-call traces, passing fixed-stream Wilson/Hoeffding gates, and zero rejection count. The privacy-safe public record is `validation/holdem/v1/cuda_statistical_release_qualification.json`; verify it CPU-only with `tools/verify_cuda_statistical_release_qualification.py`.

CUDA result provenance is restricted to `cuda-uuid:<UUID.hex()>` and `cuda-source-sha256:<approved digest>`. Host-side validation, canonical case hashing, and Philox key derivation remain authoritative.

## Scope

The binding v1 authorities are `docs/adr/`, `contracts/v1/`, and `validation/holdem/v1/SPEC.md`. The repository contains a deterministic CPU engine and an explicitly selected qualified CUDA engine. It does not yet provide a network service, automatic CUDA routing, adaptive sampling, multi-GPU execution, or silent backend fallback.
