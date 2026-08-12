# Poker Knight NG

## Revival status

Poker Knight NG is being revived as a correctness-first Texas Hold'em equity engine. The preserved legacy implementation is evidence and a possible source of test vectors, **not** a trusted oracle; its unsafe CUDA kernel will be replaced rather than incrementally repaired.

At this checkpoint, no new CPU-only reference backend, CUDA production backend, or public service is claimed complete. Do not treat the legacy package's calculations or advanced metrics as trustworthy.

- Preserved legacy tag: [`legacy/pre-revival-2026-08-12`](../../tree/legacy/pre-revival-2026-08-12)
- Preserved baseline SHA: `618f6a505590c195b75b76b027f0eea0e771ebda`

## Intended equity v1 scope

v1 is limited to standard 52-card Texas Hold'em equity for exactly two known hero cards, boards of length 0/3/4/5, and 1–6 uniformly random opponents. Each request will specify a fixed trial count and explicit `uint64` seed. The results will be limited to raw W/T/L counts, tie multiplicity, exact pot-share units (denominator 420), hero category counts, and host-derived probabilities, timing, and provenance.

Unsupported poker analytics and product surface—including ranges, ICM, strategy/advice, side pots, other variants, caching, streaming, multi-GPU balancing, and Camelot changes—will be explicitly rejected, never ignored or zero-filled. See [ADR 0001](docs/adr/0001-equity-v1-scope.md) for the binding scope and compatibility boundary.

## Development order and gates

The revival proceeds in this order:

1. preserve the legacy snapshot and freeze the v1 ADR, contracts, and validation specification;
2. establish the reproducible CPU-only package boundary;
3. build and qualify the independent evaluator/reference oracle and exact validation fixtures;
4. freeze the deterministic CPU dealer and Monte Carlo reference stream;
5. build and qualify the CUDA production backend against those reference contracts and fixtures;
6. add only the narrow service boundary and later integrations justified by conformance evidence.

Correctness gates come before performance or deployment claims: fixed requested trial counts, explicit seeds, exact integer accounting, deterministic replay, explicit rejection of unsupported features, and CPU/CUDA conformance. CPU-only and CUDA execution are planned as separate boundaries; that separation is not implemented by this README.

## Planned contract and validation locations

Later phases are expected to add versioned request/result contracts under [`contracts/v1/`](contracts/v1/) and validation material under [`validation/holdem/v1/`](validation/holdem/v1/). These locations do not exist yet and are linked here as roadmap targets, not as available artifacts. Later ADRs are likewise planned under [`docs/adr/`](docs/adr/).
