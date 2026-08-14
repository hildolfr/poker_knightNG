# Poker Knight NG

Poker Knight NG is a contract-first, deterministic Texas Hold'em equity engine for Python 3.13. The package has zero base runtime dependencies and provides two explicit synchronous engines:

- `CPUReferenceEngine` is the default public executor and the authoritative deterministic CPU implementation.
- `CUDAEngine` is an explicit opt-in executor backed by the qualified CuPy/CUDA runtime.

solve() remains CPU-default. A request for `backend="cuda"` sent through `solve()` fails with `BACKEND_UNAVAILABLE`; it is never silently downgraded or automatically routed to a GPU. Use `CUDAEngine().solve(request)` only when CUDA execution is deliberately selected.

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

The qualified source checkpoint is `c2b3eb96413d17194a85144491c71539a4818452`. Its exact wheel-installed run passed cold, warm-cache, and forced-PTX execution for all committed seed-bank vectors, the full 607-test JUnit run with zero failures/errors and two skips, and Compute Sanitizer `memcheck`, `racecheck`, `initcheck`, and `synccheck` with zero findings.

The compact public record is `validation/holdem/v1/cuda_release_qualification.json`; verify it with `tools/verify_cuda_release_qualification.py`. The record binds the source closure, qualification tool, seed authority, lockfile, wheel/sdist/JUnit/sanitizer hashes, and the private canonical evidence hash without publishing host paths, process inventories, or raw device names.

Phase 5C statistical qualification passed on executed source `f5b31a0cb94e6139cb81be2f013d9d6c44017d98`: all four preregistered batch geometries produced the exact frozen integer aggregate, exact planned kernel-call traces, passing fixed-stream Wilson/Hoeffding gates, and zero rejection count. The privacy-safe public record is `validation/holdem/v1/cuda_statistical_release_qualification.json`; verify it CPU-only with `tools/verify_cuda_statistical_release_qualification.py`.

CUDA result provenance is restricted to `cuda-uuid:<UUID.hex()>` and `cuda-source-sha256:<approved digest>`. Host-side validation, canonical case hashing, and Philox key derivation remain authoritative.

## Scope

The binding v1 authorities are `docs/adr/`, `contracts/v1/`, and `validation/holdem/v1/SPEC.md`. The repository contains a deterministic CPU engine and an explicitly selected qualified CUDA engine. It does not yet provide a network service, automatic CUDA routing, adaptive sampling, multi-GPU execution, or silent backend fallback.
