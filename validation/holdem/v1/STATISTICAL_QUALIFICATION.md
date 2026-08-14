# Phase 5C CUDA statistical qualification preregistration

**Status:** frozen harness contract; no GPU result is claimed by this document.

Phase 5C qualifies one already-committed deterministic Monte Carlo stream across four real CUDA batch geometries. It does not change the public API, default `solve()` routing, request semantics, RNG stream, integer reducer, or Phase 5A/5B evidence identities.

The machine-readable private-evidence wire contract is [`cuda_statistical_qualification.schema.json`](cuda_statistical_qualification.schema.json). Evidence is canonical ASCII JSON with closed objects and decimal-string counters. Passed evidence includes a whole-harness closure digest binding the imported Phase 5A bounded-process/admission helper, Phase 5C tool, schema, preregistration, seed authorities, runtime, and CUDA sources. Failed evidence contains only its closed failure identity; passed evidence contains the full source, artifact, gate, case, geometry, aggregate, statistical, provenance, and bounded environment records.

## Frozen case

Exactly one row is admitted from `validation/holdem/v1/rng_seed_bank.json`:

- ID: `river-high-card-wtl-2000`
- seed: `0x0000000000000001`
- hero card IDs: `[12, 37]`
- board card IDs: `[0, 18, 33, 47, 48]`
- opponent count: `1`
- requested trials: `2000`
- canonical case hash: the exact committed row value

The worker fails if the bank has zero, two, substituted, or modified statistical rows. The parent must run the exact bounded command `generate_rng_seed_bank.py --verify` successfully before GPU admission or worker dispatch. A manifest hash check alone is insufficient.

## Frozen geometries

`THREADS=128`, aggregate record size `192` bytes, and batch overhead `4096` bytes are fixed authorities.

| Evidence name | Configuration | Actual capacity | Exact `(first simulation ID, trials, partial blocks)` plan |
|---|---|---:|---|
| `capacity_1` | `batch_blocks=1`, default budget | 1 | fifteen rows `(0..1792 step 128, 128, 1)`; final `(1920, 80, 1)` |
| `budget_capacity_3` | `batch_blocks=256`, `vram_budget_bytes=4864` | 3 | `(0,384,3)`, `(384,384,3)`, `(768,384,3)`, `(1152,384,3)`, `(1536,384,3)`, `(1920,80,1)` |
| `capacity_7` | `batch_blocks=7`, default budget | 7 | `(0,896,7)`, `(896,896,7)`, `(1792,208,2)` |
| `capacity_256` | `batch_blocks=256`, default budget | 256 | `(0,2000,16)` |

The budget-clamped capacity is exact: `(4864 - 4096 - 192) / 192 = 3`. Runtime admission still requires at least 2 GiB free VRAM before compilation and before every batch.

Qualification wraps the actual private `simulate` and `reduce` kernel callables. Each batch must record and match, in order:

- ordinal, first simulation ID, trials, and partial block count;
- simulate grid `(partial_blocks,)` and block `(128,)`;
- reduce grid `(1,)`, block `(128,)`, and reducer partial count.

A calculated plan without a matching actual call trace does not pass.

## Exact and statistical gates

For every geometry, all scalar counters, six tie bins, nine category bins, equity-share units, and rejection count must satisfy the complete chain:

```text
CPU raw aggregate == frozen aggregate
CUDA raw aggregate == frozen aggregate
CUDA raw aggregate == CPU raw aggregate
```

Exact equality is authoritative for CUDA conformance. Statistical intervals never excuse an integer mismatch.

The separate calibration gates use the committed population metadata:

- each W/T/L Wilson interval is built from observed Monte Carlo successes and the fixed observed `N=2000`, then must cover the independently enumerated exact population probability;
- normalized pot-share equity uses the fixed Hoeffding radius `sqrt(log(2/alpha)/(2*N))` against the independently enumerated exact population mean;
- `alpha=0.000001` is **per-gate** and the frozen Wilson value is exactly `z=4.891638475698591`;
- 16 gate evaluations across four geometries have maximum union-bound error `0.000016`;
- the result is a **fixed-stream calibration** gate over deterministic Philox output, not a realized frequentist coverage guarantee.

The frozen `z` is a normative numerical constant, not a claim of exact symbolic equivalence to the normal quantile. Sampling is with replacement, so no finite-population correction is introduced.

## Fail-closed execution

No retry, adaptive trial count, alternate seed, alternate row, geometry substitution, tolerance widening, or fallback is permitted. Failure of seed verification, admission, compilation, allocation, launch tracing, exact equality, or any statistical gate ends the run with one stable error code.

The harness checkpoint must be independently reviewed, committed as one clean child, published, and remote-verified before the GPU run. The live run targets that exact published SHA in a clean checkout. A later public closeout records the successful private evidence hash for that prior source SHA and must not relabel itself as the qualified source.
