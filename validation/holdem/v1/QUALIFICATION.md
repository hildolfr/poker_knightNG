# Hold'em v1 qualification history

## Current status

**PASS — deterministic CPU and explicit CUDA engines qualified.** The current qualified CUDA source identity is `7fb617b900c06102caafe240ff95afe7fef2aa58`; this closeout records that already-completed run and does not relabel itself as the wheel's source.

Fast verification commands:

```sh
uv run python tools/generate_oracle_fixtures.py --verify
uv run python tools/qualify_seven_card_corpus.py --verify
uv run python tools/generate_rng_seed_bank.py --verify
uv run python tools/verify_cuda_release_qualification.py
```

## Phase 5 CUDA release qualification

The compact public record is `cuda_release_qualification.json`, bound by `manifests/cuda_release_qualification.sha256` and verified by `tools/verify_cuda_release_qualification.py`. The canonical private evidence for run `phase6c-opt-7fb617b-5b3` has SHA-256:

```text
295fb7629dc53d956f933b2dbc2cd37a1142d52d0c4bf71762e534d79272132d
```

The approved CUDA source closure has SHA-256:

```text
8da8349bed65e782a18d29f83de884341b3838f40c1e83904d07860c2c4ade5a
```

The wheel-installed qualification used Python 3.13.15, `cupy-cuda13x==14.1.1`, CUDA runtime 13.2, CUDA toolkit 13.3, and compute capability 12.0. Its admission policy required at least 2 GiB free device memory; volatile free-memory snapshots remain private.

Acceptance gates:

- exact clean source SHA and qualification-branch identity;
- wheel, sdist, `METADATA`, `RECORD`, installed-byte closure, source closure, lockfile, seed bank, and seed-manifest binding;
- all three committed seed-bank vectors equal across authoritative CPU, CUDA, and frozen aggregates;
- distinct cold, warm-cache, and forced-PTX worker paths agree exactly;
- JUnit: 799 total, 791 passed, 8 skipped, 0 failures, 0 errors;
- Compute Sanitizer `memcheck`: `ERROR SUMMARY: 0 errors`;
- Compute Sanitizer `racecheck`: `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`;
- Compute Sanitizer `initcheck`: `ERROR SUMMARY: 0 errors`;
- Compute Sanitizer `synccheck`: `ERROR SUMMARY: 0 errors`;
- canonical evidence passed the CPU-only verifier on the GPU host and again after transfer.

The public record deliberately contains no host paths, process IDs, workload inventory, raw GPU name, or unbounded logs. It retains safe device/source provenance, exact artifact hashes and sizes, environment versions, fixed summaries, and the private evidence hash. Wheel/sdist/JUnit/sanitizer files remain operator evidence rather than repository release assets.

This qualification does **not** change routing: `solve()` remains CPU-default and an explicit CUDA request through it fails `BACKEND_UNAVAILABLE`. Qualified CUDA execution is opt-in through `CUDAEngine`. There is no network service or silent fallback.

## Phase 5C statistical qualification

**PASS — executed source `7fb617b900c06102caafe240ff95afe7fef2aa58`.** The identity-bound private evidence SHA-256 is `27bf23106e44e399fb5157d1c61891852f9d9ce0f32d2504dc6f490271a24017`. All four frozen batch geometries matched their exact kernel-call plans and complete CPU/frozen integer aggregate; every preregistered Wilson/Hoeffding gate passed and rejection count was zero.

The privacy-safe public record is `cuda_statistical_release_qualification.json`. Verify it without CuPy or GPU access:

```sh
uv run python tools/verify_cuda_statistical_release_qualification.py
```

The record omits host paths, process inventory, memory snapshots, raw device names, and timing data. This qualification does not change routing: `solve()` remains CPU-default; CUDA remains explicit through `CUDAEngine`.

## Phase 2 oracle fixture qualification

Repeatable commands:

```sh
RUN_ORACLE_RELEASE_QUALIFICATION=1 uv run python tools/generate_oracle_fixtures.py --release
RUN_SEVEN_CARD_RELEASE_QUALIFICATION=1 uv run python tools/qualify_seven_card_corpus.py --release
```

**PASS — 2026-08-13.** The final checkpoint-C `uv run python tools/generate_oracle_fixtures.py --release` completed successfully in **196.236 seconds**. It regenerated canonical bytes only after the transparent reference and pinned `treys==0.1.8` agreed on every exact/tie counter, category bin, and 420-unit total. The checked-in manifest is bound to the final generator source.

The release corpus covers unknown river (990 deals), unknown turn (45,540), selected unknown flop `As Kd` / `2s 7h 9d` (1,070,190), fixed-hole turn (44), AA-vs-KK preflop `C(48,5)=1,712,304`, terminal loss, and all six shared-board tie bins. Treys royal class 0 is normalized to v1 `straight_flush`.

The full `C(52,7)=133,784,560` seven-card release-candidate corpus is also complete. `seven_card_release_qualification.json` records exact nine-category totals, deterministic command, implementation SHA-256 bindings, and the differential scope. The exact C category function used by the full-space counter agrees per hand with both the transparent 21-subset evaluator and pinned Treys 0.1.8 on a deterministic 10,000-hand all-category/wheel sample; canonical totals independently constrain every full-space category.

`generate_oracle_fixtures.py --verify` and `qualify_seven_card_corpus.py --verify` are fast, CPU-only checks. The expensive release gates remain explicitly opt-in and must not be inferred from the fast verifiers alone.
