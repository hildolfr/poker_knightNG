# Hold'em v1 qualification history

## Current status

**PASS — deterministic CPU and explicit CUDA engines qualified.** The qualified source identity for the CUDA run is `c2b3eb96413d17194a85144491c71539a4818452`; this later documentation checkpoint records that already-completed run and does not relabel itself as the wheel's source.

Fast verification commands:

```sh
uv run python tools/generate_oracle_fixtures.py --verify
uv run python tools/qualify_seven_card_corpus.py --verify
uv run python tools/generate_rng_seed_bank.py --verify
uv run python tools/verify_cuda_release_qualification.py
```

## Phase 5 CUDA release qualification

The compact public record is `cuda_release_qualification.json`, bound by `manifests/cuda_release_qualification.sha256` and verified by `tools/verify_cuda_release_qualification.py`. The canonical private evidence for run `phase5-c2b3eb9-final2` has SHA-256:

```text
9e2edef60ec2a890b970ef83a8c114af0f56e74f7f0bf1c7e20c21e84ae5178d
```

The approved CUDA source closure has SHA-256:

```text
d011e0f5c4db4d12fcb5240b5996f0911af7f153c7942f16772f871917ca5263
```

The wheel-installed qualification used Python 3.13.15, `cupy-cuda13x==14.1.1`, CUDA runtime 13.2, CUDA toolkit 13.3, and compute capability 12.0. It required at least 2 GiB free device memory and recorded 13,529,776,128 bytes before and after the run.

Acceptance gates:

- exact clean source SHA and qualification-branch identity;
- wheel, sdist, `METADATA`, `RECORD`, installed-byte closure, source closure, lockfile, seed bank, and seed-manifest binding;
- all three committed seed-bank vectors equal across authoritative CPU, CUDA, and frozen aggregates;
- distinct cold, warm-cache, and forced-PTX worker paths agree exactly;
- JUnit: 607 total, 605 passed, 2 skipped, 0 failures, 0 errors;
- Compute Sanitizer `memcheck`: `ERROR SUMMARY: 0 errors`;
- Compute Sanitizer `racecheck`: `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`;
- Compute Sanitizer `initcheck`: `ERROR SUMMARY: 0 errors`;
- Compute Sanitizer `synccheck`: `ERROR SUMMARY: 0 errors`;
- canonical evidence passed the CPU-only verifier on the GPU host and again after transfer.

The public record deliberately contains no host paths, process IDs, workload inventory, raw GPU name, or unbounded logs. It retains safe device/source provenance, exact artifact hashes and sizes, environment versions, fixed summaries, and the private evidence hash. Wheel/sdist/JUnit/sanitizer files remain operator evidence rather than repository release assets.

This qualification does **not** change routing: `solve()` remains CPU-default and an explicit CUDA request through it fails `BACKEND_UNAVAILABLE`. Qualified CUDA execution is opt-in through `CUDAEngine`. There is no network service or silent fallback.

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
