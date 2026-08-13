# Poker Knight NG

## Phase 1 package boundary

Poker Knight NG now has a reproducible **CPU-only installation and contract boundary** for Python 3.13. Base installation has no CuPy or NumPy dependency and importing `poker_knight_ng` does not load CUDA or compile kernel sources. The v1 request/result contract and schema resources are included in both wheel and source distributions.

```bash
uv sync --frozen --group dev
uv run pytest -q
```

CUDA is deliberately optional. Install the CUDA 13 extra only on a qualified CUDA 13 host:

```bash
uv sync --extra gpu-cu13
```

A request with `backend: "cuda"` is never silently downgraded; until a new backend is qualified it fails with the stable `BACKEND_UNAVAILABLE` contract problem. This checkpoint provides parsing, canonical case encoding, result-invariant validation, and packaging only—it does **not** implement a CPU solver or CUDA solver.

The superseded legacy Python solver, server, CuPy wrapper, card utilities, validator, adaptive/OOM fallback implementation, and their tests and benchmarks are preserved at tag `legacy/pre-revival-2026-08-12` and are deliberately excluded from the revival branch and distributions. CUDA `.cu`/`.cuh` files remain inert package data under `poker_knight_ng/cuda-sources/` for later replacement and qualification work; there is no importable `poker_knight_ng.cuda` namespace, and importing the package never compiles or executes them.

## Contract authority

The binding v1 definitions are in [`docs/adr/`](docs/adr/), [`contracts/v1/`](contracts/v1/), and [`validation/holdem/v1/SPEC.md`](validation/holdem/v1/SPEC.md). Cards use canonical ASCII tokens, work is fixed-count and seed-explicit, and canonical case hashes follow ADR 0003. Unsupported historical analytics and fallback behavior are explicitly rejected.

## Revival status

The preserved legacy implementation is evidence and a possible source of test vectors, **not** a trusted oracle; its unsafe CUDA kernel will be replaced rather than incrementally repaired. The independent evaluator/reference oracle, deterministic dealer, production CUDA backend, and service remain later phases.

- Preserved legacy tag: [`legacy/pre-revival-2026-08-12`](../../tree/legacy/pre-revival-2026-08-12)
- Preserved baseline SHA: `618f6a505590c195b75b76b027f0eea0e771ebda`
