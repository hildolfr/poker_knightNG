# Phase 6C performance-characterization method

**Status:** normative protocol and implemented private harness; **no baseline has
been recorded**. This document preregisters how the first CUDA baseline is to be
collected and published. The harness is an operator-only tool, not a public
package command, and this document makes no measured performance claim.

**Historical provenance note:** `TODO.md` is qualification-bound legacy history
retained byte-for-byte under immutable Phase 5 authority. Its dated 2025 phrases
such as `current: 256` and `Current Config (256 threads)` describe a superseded
historical profile and are not a current runtime or performance claim. The
qualified runtime and this protocol authoritatively fix 128 threads per block.

## 1. Scope and boundary

Phase 6C characterizes the latency and throughput of the existing explicit CUDA
path for the fixed Hold'em v1 workload below. It does not change the public v1
contract, routing, result schema, or CLI. In particular, `solve()` remains
CPU-only and `solve_cuda()` remains the only normal public Python route for a
CUDA request, as required by [ADR 0004](adr/0004-explicit-cuda-routing-and-cli.md).
The benchmark execution path is private and must not add a public command,
option, endpoint, field, fallback, retry, or adaptive-sampling behavior.

All normal (uninstrumented) measurements use exactly one explicit CUDA solve of
one valid request. The **end-to-end host latency** begins immediately before the
call to the normal explicit CUDA solve route and ends immediately after it
returns or raises, using one host monotonic clock in nanoseconds. It includes
normal request-to-result work performed by that route: lazy runtime work when
applicable, admission, CUDA context/compiler/cache work when applicable, input
transfer, kernel launch/execution, reduction, result transfer, host validation,
aggregation, and construction of the returned result. It excludes controller
setup, scenario-file parsing, process creation, artifact hashing, evidence
serialization, and printing. A failed call has no latency sample.

The one **instrumented stage pass** is a separate private observer pass. It does
not supply end-to-end latency or throughput values. With CUDA events on the
active stream, it reports aggregate GPU elapsed durations (nanoseconds) for:

- `h2d`: event immediately before the observer's asynchronous prepared-input
  copy through the event immediately after that copy;
- `simulate`: for every batch, event immediately before the simulation-kernel
  launch through the event immediately after that launch, summed across batches;
- `reduction`: for every batch, event immediately before the reduction-kernel
  launch through the event immediately after that launch, summed across batches;
- `d2h`: for every batch, event immediately before the asynchronous final
  aggregate copy to its pinned host buffer through the event immediately after
  that copy, summed across batches.

Stage launch geometry is preregistered at exactly **256 blocks per batch**, the
immutable qualified runtime default, with 128 threads per block. The fixed trial
counts `10,000`, `100,000`, `500,000`, and `1,000,000` therefore require exactly
`1`, `4`, `16`, and `31` batches, respectively. Each batch contributes one
simulate, reduction, and D2H event pair. A device unable to admit this exact
plan aborts; the plan is never reduced or tuned dynamically.

The observer must preallocate CUDA events, device input buffers, pinned H2D
input buffers, and pinned D2H aggregate buffers before its first measured
boundary. It must synchronize only
at the final D2H event before resolving elapsed event times. The observed stage
sequence and aggregate must match the benchmark-only `_run_private_stage` loop;
that loop is neither a public API nor a production execution route and does not
modify the source-bound qualified CUDA runtime.

A resolved CUDA-event interval may be `0` nanoseconds at the event timer's
resolution. Zero is retained as a measured stage value; a missing event,
negative elapsed time, malformed boundary, or absent stage remains an abort.

## 2. Fixed workload matrix

The canonical scenario authority is `benchmarks/scenarios/v1/`, format
`phase6c-scenario-v1`. Its twelve base scenarios are the Cartesian product:

| Street | Known board cards | Opponents |
|---|---:|---:|
| preflop | 0 | 1, 3, 6 |
| flop | 3 | 1, 3, 6 |
| turn | 4 | 1, 3, 6 |
| river | 5 | 1, 3, 6 |

Each committed scenario fixes canonical card order, hero cards, board, seed,
street, opponent count, and ID (`v1-<street>-o<opponents>`). The trial dimension
is exactly `10,000`, `100,000`, `500,000`, and `1,000,000`. Thus the controller
expands the twelve scenarios into exactly **48 cells**, with ID
`v1-<street>-o<opponents>-n<trials>`, and forces `backend: "cuda"` for every
cell. No scenario, seed, trial count, batching geometry, or matrix member may be
added, removed, substituted, or tuned after a run begins.

## 3. Run classes and order

The four run classes answer different questions and must never be pooled:

1. **Cold start with empty cache:** one fresh process measures the fixed startup canary
   `v1-preflop-o1-n10000` with an empty run-owned compiler/cache directory. It
   measures one normal explicit CUDA solve. Compilation and cache population
   may occur, but this class does not claim that compilation is the only measured
   work. It is reported separately and is not a steady sample.
2. **Warm cache / fresh process:** after the cold process has exited, one fresh
   process measures the same fixed startup canary using the same populated
   run-owned disk cache. It measures one normal explicit CUDA solve. It is
   reported separately and is not a steady sample.
3. **Steady same-process:** one already-admitted process measures all 48 cells.
   For each cell it performs one unmeasured normal explicit CUDA-solve warmup,
   then exactly 30 normal, successful repetitions of that same cell in that
   process. The 30 durations are the only population for steady percentiles and
   throughput.
4. **Instrumented stage pass:** one separate observer-enabled private pass for
   each of the 48 cells after the normal measurements. It reports stage
   durations only and must not replace a startup or steady normal solve.

After cold succeeds, the controller recursively records every regular cache
file's sorted relative path, byte size, and SHA-256 in a canonical seal. It
rejects empty caches, links, non-files, additions, removals, or changed bytes.
Immediately before the warm worker, it independently rescans the shared cache;
private evidence retains both the cold seal and warm verification manifest, and
the verifier requires them to be byte-for-byte equal and bound to the cold
analytical digest.

A run record must state process identity only in private evidence, run class,
cell ID, ordinal, and whether a value is warmup, valid sample, or failure.
Warmups are unmeasured: they are retained as execution-status evidence but are
not latency samples. There is no additional burn-in. A failed warmup, failed
normal solve, missing observer boundary, failed stage pass, or incomplete
sample population aborts publication for the affected benchmark execution; no
replacement, retry-as-sample, or partial percentile is allowed.

## 4. Statistics

For a steady cell, let the 30 valid uninstrumented durations be positive integer
nanoseconds `d[1]..d[30]`. Sort them ascending to `s[1]..s[30]`. For percentile
`p` in `{5, 50, 95}`, report nearest-rank:

```text
rank(p) = ceil(p * 30 / 100)
p<p>_duration_ns = s[rank(p)]
```

Accordingly, the ranks are 2, 15, and 29 for p5, p50, and p95. All valid steady
samples are retained: **outlier deletion, clipping, winsorization, and selective
repetition are prohibited**. A controller may validate that a duration is a
positive integer, but it must not alter it.

Steady throughput is derived only from p50 end-to-end host latency:

```text
samples_per_second = requested_trials * 1,000,000,000 / p50_duration_ns
reported = Decimal(samples_per_second).quantize(Decimal("0.001"),
                                                rounding=ROUND_HALF_UP)
```

The serialized throughput is a fixed-scale decimal string with exactly three
fractional digits. Do not average per-repetition throughput, derive throughput
from GPU events, or use cold/warm/stage durations in this calculation.

## 5. Admission and analytical equality gates

Before any sample, the controller must establish that each normal call is an
explicit CUDA request and is admitted by the normal runtime. The runtime's
existing admission remains authoritative: source/CuPy/device/ABI qualification,
free-VRAM checks before compilation/context work and again after compilation and
before every batch allocation, and bounded allocation behavior. The private
controller must not suppress or reinterpret `BACKEND_UNAVAILABLE`,
`RESOURCE_EXHAUSTED`, RNG exhaustion, invariant failure, or any other execution
failure. It must never stop other workloads to manufacture admission.

Every successful normal or stage execution must satisfy all v1 exact analytical
invariants, including completed trials equal requested trials; outcome and
nine-category partitions; legal tie bins; exact equity-unit algebra; and exact,
unreduced fraction reconstruction. Phases 5B and 5C remain the independent
CUDA correctness and statistical-qualification authorities; Phase 6C does not
repeat or replace those proofs.

For each cell, an isolated-cache fresh process produces one uninstrumented
public `solve_cuda()` analytical reference digest. This reference tests exact
repeatability across the warmup, all 30 steady calls, and the separately
instrumented stage pass. It is deliberately not described as an independent
implementation or new correctness proof. Equality covers completed trials,
wins, every tie bin, losses, equity-share units, every hero-category count,
rejection count, and all request-bound analytical result fields after removing
backend timing and provenance. Equality is exact; no statistical interval,
rounding tolerance, or matching displayed equity value substitutes for it.
Private evidence retains the separate-process reference digest per cell, the
warmup digest, all 30 normal-run digests, and the instrumented-stage digest.
The stdlib verifier requires every retained digest for that cell to equal the
reference digest; a self-attested equality boolean is not evidence. Failure of
admission, equality, or any semantic invariant is an abort gate, not a
performance datum.

## 6. Reproducibility and identity binding

A proposed execution is admissible only when its private evidence binds all of
the following exact identities before results are analyzed:

- clean source checkout: repository commit SHA and `git status --porcelain` with
  no output;
- exact source closure: the clean Git commit binds every tracked protocol,
  controller, verifier, method, CUDA-source, and scenario byte; direct SHA-256
  records additionally bind the executed benchmark tool and qualified CUDA
  runtime source;
- exact dependency authority: SHA-256 of `uv.lock`, the Python version, and the
  installed package/distribution metadata; `pyproject.toml` is bound by the
  clean Git commit and the built wheel/sdist artifacts;
- exact installation artifacts: wheel filename and SHA-256, sdist filename and
  SHA-256, and SHA-256 of the installed package-file manifest. The tested
  installation must be made from the recorded wheel; the sdist is independently
  recorded as a release-source binding, not silently substituted for the wheel;
- exact scenario authority: SHA-256 of each canonical scenario file and a
  deterministic manifest SHA-256 over the complete 12-file set;
- immutable correctness authorities: exact SHA-256 bindings for the published
  Phase 5B and Phase 5C qualification records, their historical source-closure
  manifests, both historical stdlib verifiers, and qualified
  `_cuda_runtime.py`; both published verifiers must pass against the live clean
  checkout before Phase 6C evidence can be accepted;
- CUDA identity: resolved CUDA driver/runtime information, exact CuPy version
  (`14.1.1`), device UUID, compute capability, and the approved CUDA source
  digest; and
- protocol identity: benchmark ID `holdem-v1-cuda-baseline-1`, scenario format,
  benchmark-tool hash, fixed six-worker sequence, and the clean commit that
  binds this document and the controller/verifier sources.

The source checkout must be clean at collection and publication. A mismatch,
unreadable identity, symlink substitution, source-digest mismatch, wrong CuPy
version, wrong scenario matrix, or artifact/install discrepancy aborts the run.
A new commit, lockfile, dependency, scenario byte, device, driver, runtime, or
controller requires a newly bound execution; results may not be relabeled as
comparable by prose alone.

## 7. Evidence and privacy

Raw evidence is **private**. It contains the complete ordered duration vectors,
six canonical worker payloads and capture hashes, the cold/warm cache-seal
binding, installed wheel closure, bounded raw GPU-admission snapshots, private
host/CUDA environment values, and hashes needed for audit. Stable generic
failures do not become evidence records. Store successful private evidence in
access-controlled project evidence storage; do not commit or publish it by
default.

A public projection may contain only the following allowlisted fields:

- benchmark/protocol ID and format version;
- clean source commit SHA and hashes of the lockfile, scenario manifest,
  controller/method, wheel, sdist, installed manifest, Phase 5B/5C authority
  records and manifests, and private-evidence canonical digest;
- scenario ID, street, opponent count, requested trials, run class, and the
  p5/p50/p95 steady durations and p50-derived throughput where applicable;
- stage names and their aggregate GPU nanoseconds for the instrumented pass;
- CUDA provenance restricted to `cuda-uuid:<lowercase-hex>` and
  `cuda-source-sha256:<lowercase-hex>`; and
- protocol outcome (`complete` or a closed abort category) without raw errors.

The public projection must not contain hostnames, usernames, paths, IP/MAC
addresses, process IDs or inventories, timestamps, raw GPU product names,
driver diagnostic strings, cache locations, environment variables, command
lines, raw duration vectors, raw exceptions, submitted card values, or seeds.
No field outside this allowlist is public merely because it seems harmless.

## 8. Abort gates and non-claims

Abort without a successful evidence record and publish no baseline performance
claim if any of these occurs: a dirty or mismatched source/install/artifact
binding; matrix or canonical-scenario failure; unavailable CUDA/qualification;
failed resource admission; competing GPU compute process observed in either the
before or after collection snapshot; an execution/error/invariant/equality
failure; fewer than
exactly 30 valid steady samples; omitted warmup; cache-mode contamination;
missing/invalid stage boundaries; prohibited evidence disclosure; or a
controller/harness change during collection. The operator must not terminate,
pause, deprioritize, or otherwise stop competing workloads. Instead, abort and
reschedule under an uncontended condition.

This protocol makes no claim about absolute speed, speedup, regression,
scalability, energy, fairness across hardware, CPU performance, multi-GPU
behavior, service performance, or production SLA. It creates no performance
threshold. Until at least one complete, variance-bearing baseline exists, no
optimization is justified as a measured improvement.

After a complete baseline, an optimization is eligible for consideration only
when it repeats this exact protocol on an exactly bound comparable environment,
preserves every analytical equality gate, and improves the relevant
predeclared steady p50 metric without regressing the corresponding predeclared
p95 metric. Any changed workload, identity, or uncontended condition creates a
new baseline rather than proof of improvement.

## 9. Reproduction status and command forms

The implemented controller and its six workers remain private tools. Do not
present a public `benchmark` package subcommand as available. From a clean
checkout, use absolute paths and an output path outside the checkout:

```sh
uv sync --frozen --group dev --extra gpu-cu13
uv run pytest -q tests/benchmarks \
  tests/cuda/test_private_benchmark_runtime_observer.py \
  tests/cuda/test_private_runtime_adapter.py
uv run python -m build
PRIVATE_ROOT="/absolute/private/path/phase6c-run"
uv venv --python 3.13 "$PRIVATE_ROOT/venv"
uv export --frozen --extra gpu-cu13 --no-dev --no-emit-project \
  --output-file "$PRIVATE_ROOT/runtime-requirements.txt"
uv pip sync --python "$PRIVATE_ROOT/venv/bin/python" \
  "$PRIVATE_ROOT/runtime-requirements.txt"
uv pip install --python "$PRIVATE_ROOT/venv/bin/python" --no-deps \
  "$PWD/dist/<exact-wheel>.whl"
"$PRIVATE_ROOT/venv/bin/python" tools/run_cuda_benchmark.py \
  --repo-root "$PWD" \
  --wheel "$PWD/dist/<exact-wheel>.whl" \
  --sdist "$PWD/dist/<exact-sdist>.tar.gz" \
  --lockfile "$PWD/uv.lock" \
  --scenario-dir "$PWD/benchmarks/scenarios/v1" \
  --scenario-manifest "$PWD/benchmarks/scenarios-v1.json" \
  --output "$PRIVATE_ROOT/phase6c-private.json"
"$PRIVATE_ROOT/venv/bin/python" tools/verify_cuda_benchmark_evidence.py \
  "$PRIVATE_ROOT/phase6c-private.json" \
  --repo-root "$PWD" \
  --wheel "$PWD/dist/<exact-wheel>.whl" \
  --sdist "$PWD/dist/<exact-sdist>.tar.gz" \
  --lockfile "$PWD/uv.lock" \
  --scenario-dir "$PWD/benchmarks/scenarios/v1" \
  --scenario-manifest "$PWD/benchmarks/scenarios-v1.json"
```

The output filename must not already exist. The controller refuses dirty source,
symlinked authorities, malformed or contested GPU admission, worker stderr,
digest disagreement, incomplete populations, invalid cache handoff, and any
record rejected by the bound verifier.
