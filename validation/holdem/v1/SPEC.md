# Hold'em v1 validation and conformance specification

**Status:** binding v1 contract/validation specification. The schema/semantic contract, exact oracle, deterministic CPU stream/engine, explicit CUDA engine, deterministic GPU qualification/publication checkpoints, and explicit local Python API/CLI are complete. A network service and automatic CUDA routing remain unimplemented.

## 1. Authority, scope, and future artifacts

The normative authorities are [ADR 0001](../../../docs/adr/0001-equity-v1-scope.md) (scope and no-fallback rule), [ADR 0002](../../../docs/adr/0002-card-rank-and-tie-semantics.md) (cards, rank keys, outcomes, categories, and equity), [ADR 0003](../../../docs/adr/0003-deterministic-rng-and-deal-order.md) (case bytes, SHA-256, Philox, and dealing), [ADR 0004](../../../docs/adr/0004-explicit-cuda-routing-and-cli.md) (explicit local API/CLI routing and bounded command-line behavior), and the v1 [request](../../../contracts/v1/equity-request.schema.json), [result](../../../contracts/v1/equity-result.schema.json), and [problem](../../../contracts/v1/problem.schema.json) schemas. If wording conflicts, the ADR/schema that owns the subject controls.

The preserved legacy code and tests are evidence and may suggest vectors; they are **not an oracle**. Phase 2's independent transparent evaluator/exact oracle, committed fixture corpus, independent-engine qualification, and complete `C(52,7)=133,784,560` release-candidate category corpus are available. The hash-bound `seven_card_release_qualification.json` records exact canonical totals and its honest differential scope: the exact C category function used by the exhaustive counter, transparent evaluator, and pinned Treys 0.1.8 agree per hand on a deterministic 10,000-hand all-category/wheel sample, while every full-space category is checked against canonical totals. Phase 3 supplies the deterministic Philox/deal stream, CPU Monte Carlo engine, and generated seed bank. Phase 4 supplies the explicit CUDA engine and source-hashed runtime. Phase 5 supplies the fail-closed exact-SHA GPU qualification/statistical harnesses and public hash-bound qualification records. Phase 6 supplies the explicit no-fallback local Python API and bounded JSON CLI.

The committed Phase 2 fixture materials are:

* `canonical_rank_vectors.jsonl`
* `exact_holdem_cases.jsonl`
* `tie_and_split_cases.jsonl`
* `manifests/sha256sums.txt`
* `QUALIFICATION.md`

`rng_seed_bank.json` and `manifests/rng_seed_bank.sha256` are the committed Phase 3 deterministic replay authority. `cuda_release_qualification.json` and `manifests/cuda_release_qualification.sha256` are the Phase 5 public qualification record and binding manifest. Qualification records document reproducible commands and exact source/evidence identities. Fixture generation is reviewed separately from any production evaluator/CUDA path. No generated fixture may use legacy code or production CUDA as its sole oracle: every asserted fixture needs at least one independently derived path (transparent five-card/subset enumeration, independently reviewed arithmetic, or a separately pinned differential engine). A manifest authenticates bytes, not truth.

## 2. Layers and evidence plan

| Layer | What is checked | Planned evidence | Status |
|---|---|---|---|
| Schema/structural | JSON shape, closed objects, lexical wire formats | schema positives/negatives | complete |
| Semantic | cross-field meanings and integer conservation | contract semantic tests | complete |
| Exact evaluator/oracle | five-card keys, best 5 of 7, exact Hold'em counts | committed rank/exact/tie JSONL fixtures, SHA-256 manifest, Treys differential qualification | Phase 2 complete |
| Deterministic stream replay | case bytes/hash, Philox, selection and deals | KATs, deal vectors, seed bank | Phase 3 complete |
| CUDA conformance | CPU/CUDA exact integer-result equality, fixed reduction geometry, cold/warm/PTX paths, sanitizers | exact-SHA qualification harness and public record | Phases 4–5 complete |
| Statistical characterization | predeclared interval checks against exact cases | deterministic seed-bank jobs | Phase 5C complete |

### Implemented checkpoint status

- **Phase 3 — deterministic CPU stream and engine: COMPLETE.**
- **Phase 4 — explicit CUDA engine: COMPLETE.** CUDA remains explicitly selected and never silently substituted.
- **Phase 5A — deterministic GPU qualification harness: COMPLETE.**
- **Phase 5B — qualification publication: COMPLETE.** The source checkpoint is `c2b3eb96413d17194a85144491c71539a4818452`; the later publication checkpoint does not replace that source identity.
- **Phase 5C-A — CUDA statistical qualification harness: COMPLETE.** The evidence schema, fixed geometry plans, corrected observed-sample Wilson direction, Hoeffding gate, exact aggregate equality, and single-attempt worker/orchestrator are frozen.
- **Phase 5C-B — CUDA statistical execution and publication: COMPLETE.** Executed source `f5b31a0cb94e6139cb81be2f013d9d6c44017d98` passed all four frozen geometries; the privacy-safe public record identity-binds the canonical private evidence digest and historical harness closure.
- **Phase 6 — explicit public API and CLI: COMPLETE.** `solve()` remains CPU-only; `solve_cuda()` and `solve-cuda` are explicit CUDA routes with no fallback, bounded strict JSON input, canonical v1 output, and closed problem/exit behavior.

Structural schema validity is necessary but never proves semantic validity. Exact oracle comparison is not a substitute for deterministic replay, and statistical agreement never waives an exact invariant.

## 3. Result-field coverage matrix

All result fields are required by the result schema. `u64` means canonical decimal unsigned-64-bit JSON string; `fraction` means the closed object `{numerator,denominator}` of such strings. CPU and CUDA reducers implement the exact integer algebra; statistical tolerance never applies to reducer equality.

| Wire path | Semantics / units | Provenance | Invariant/test |
|---|---|---|---|
| `contract_version` | literal `v1` | accepted request/contract | schema const; request/result coherence |
| `backend` | actual `cpu_reference` or `cuda`, never fallback | requested/actual executor | enum; equals request; unavailable requested backend errors |
| `rng.algorithm_id` | `poker-knight-ng/philox4x32-10` | accepted request / ADR 0003 | exact const and replay KAT |
| `rng.algorithm_version` | string literal `1` | accepted request / ADR 0003 | exact const and replay KAT |
| `case_hash` | lowercase hex SHA-256 of ADR 0003 bytes; seed/trials excluded | canonicalizer | recompute from validated cards/O; two implementations agree |
| `seed` | `0x` + 16 lowercase hex digits, explicit uint64 | accepted request | schema format; LE64 key/deal replay |
| `requested_trials` | positive bounded canonical decimal, trials | accepted request | schema bound; equals `completed_trials` on success |
| `completed_trials` | `u64`, completed trials | final reducer | equals requested; nonzero |
| `unique_wins` | `u64` hero-only maximum events, trials | evaluator/reducer | outcome partition; adds 420 units each |
| `ties` | `u64` hero tie events, not equity, trials | evaluator/reducer | equals six-bin sum |
| `tie_by_other_winners.1` | `u64` ties with 1 other equal-key winner, trials | evaluator/reducer | allowed only O>=1; unit 210 |
| `tie_by_other_winners.2` | `u64`, 2 other winners, trials | evaluator/reducer | zero when O<2; unit 140 |
| `tie_by_other_winners.3` | `u64`, 3 other winners, trials | evaluator/reducer | zero when O<3; unit 105 |
| `tie_by_other_winners.4` | `u64`, 4 other winners, trials | evaluator/reducer | zero when O<4; unit 84 |
| `tie_by_other_winners.5` | `u64`, 5 other winners, trials | evaluator/reducer | zero when O<5; unit 70 |
| `tie_by_other_winners.6` | `u64`, 6 other winners, trials | evaluator/reducer | zero when O<6; unit 60 |
| `losses` | `u64` events with a strictly higher opponent key, trials | evaluator/reducer | outcome partition; 0 units |
| `equity_share_units` | `u64` accumulated exact 1/420 pot shares | reducer | exact tie-unit formula |
| `hero_category_counts.high_card` | `u64` final hero category events, trials | max 5-of-7 evaluator/reducer | nine-category partition |
| `hero_category_counts.one_pair` | `u64` final hero category events, trials | evaluator/reducer | nine-category partition |
| `hero_category_counts.two_pair` | `u64` final hero category events, trials | evaluator/reducer | nine-category partition |
| `hero_category_counts.three_of_a_kind` | `u64` final hero category events, trials | evaluator/reducer | nine-category partition |
| `hero_category_counts.straight` | `u64` final hero category events, trials | evaluator/reducer | wheel/six-high vectors; partition |
| `hero_category_counts.flush` | `u64` final hero category events, trials | evaluator/reducer | flush-kicker vectors; partition |
| `hero_category_counts.full_house` | `u64` final hero category events, trials | evaluator/reducer | double-trips vector; partition |
| `hero_category_counts.four_of_a_kind` | `u64` final hero category events, trials | evaluator/reducer | quads-kicker vector; partition |
| `hero_category_counts.straight_flush` | `u64` final hero category events, trials | evaluator/reducer | board/royal-display vectors; no `royal_flush` field |
| `probabilities.unique_win` | unreduced `{unique_wins, completed_trials}`, unitless event probability | host from counts | exact numerator/denominator recomputation |
| `probabilities.tie` | unreduced `{ties, completed_trials}`, unitless event probability | host from counts | exact recomputation |
| `probabilities.loss` | unreduced `{losses, completed_trials}`, unitless event probability | host from counts | exact recomputation |
| `probabilities.showdown_equity` | unreduced `{equity_share_units,420*completed_trials}`, unitless pot share | host from integer units | exact recomputation; not event-tie probability |
| `timing.total_duration_ns` | `u64` host monotonic end-to-end elapsed nanoseconds | host monotonic clock | canonical u64 format; no correctness threshold |
| `provenance.engine_build_id` | safe printable build/revision ID | build system | regex/bounds/injection rejection |
| `provenance.backend_qualification` | safe printable actual backend qualification ID/status | qualification record | regex/bounds and actual-backend presence |
| `provenance.device_id` | safe printable CUDA device ID, or `null` for CPU | runtime discovery | CPU null / CUDA non-null schema coherence |
| `provenance.kernel_id` | safe printable CUDA source/kernel ID, or `null` for CPU | CUDA build | CPU null / CUDA non-null schema coherence |

The complete semantic equations are:

```text
ties = Σ bins[k]                         (k=1..6)
unique_wins + ties + losses = completed_trials
Σ nine hero_category_counts = completed_trials
equity_share_units = 420*unique_wins + Σ (420/(k+1))*bins[k]
completed_trials = requested_trials       (success only)
```

No success may contain partial counters. The four fraction representations are exact and **unreduced** as listed in the matrix. `royal_flush`, analytics, fallback, adaptive sampling, streaming, and multi-GPU result fields do not exist in v1.

## 4. Exact rank, tie, and equity gates

Rank fixtures MUST reproduce ADR 0002’s wheel `(4,5,0,0,0,0)`, six-high `(4,6,0,0,0,0)`, flush-kicker, two-pair kicker, double-trips, quads-kicker, and royal-display-as-`straight_flush` examples. Seven-card evaluation is the lexicographic maximum of all 21 five-card subsets.

| k other winners | total winners | units/event `420/(k+1)` | expected bin/fraction behavior |
|---:|---:|---:|---|
| 1 | 2 | 210 | bin 1 increments once; tie event numerator increments once |
| 2 | 3 | 140 | bin 2 increments once; tie event numerator increments once |
| 3 | 4 | 105 | bin 3 increments once; tie event numerator increments once |
| 4 | 5 | 84 | bin 4 increments once; tie event numerator increments once |
| 5 | 6 | 70 | bin 5 increments once; tie event numerator increments once |
| 6 | 7 | 60 | bin 6 increments once; tie event numerator increments once |

### Independent documented derivations (Phase 0 evidence retained)

These two auditable derivations predate the executable Phase 2 oracle and remain useful review evidence; checkpoint C additionally supplies executable fixture reproduction and differential qualification.

| ADR 0002 example | Method A: rank-key / evaluator comparison | Method B: independent subset or winner-set arithmetic | Required result |
|---|---|---|---|
| Board `As Ks Qs Js Ts`; hero `2c 3d`; O1 `4h 5c` | The board subset is `(8,14,0,0,0,0)` for both; no higher subset exists. | Direct board-playing reasoning: the five board cards are a shared royal-display straight flush, hence winner set `{hero,O1}` and equal halves. | bin1=1, ties=1, wins=losses=0, units=210, hero `straight_flush`=1. |
| Same board; hero; O1; O2 `6h 7c` | All three maximum keys equal `(8,14,0,0,0,0)`. | Shared board gives winner set of cardinality 3; each gets 1/3, independently of hole cards. | bin2=1, ties=1, units=140. |
| Board `As Ks Qs Js 2d`; hero `Tc 3c`; O1 `Th 4c`; O2 `8h 8d` | Hero/O1 each select `T,J,Q,K,A` straight `(4,14,0,0,0,0)`; O2’s pair loses. | Exhaustive candidate reasoning: board plus each T makes broadway; O2 has no T and cannot make that straight. Winner set is `{hero,O1}`, not all opponents. | bin1=1, ties=1, units=210; O2 does not affect k. |
| Board `As Ah Ad Kc Qd`; hero `Ks Kd`; O `Ac Qc` | Hero key `(6,14,13,0,0,0)`; opponent key `(7,14,13,0,0,0)` is lexicographically higher. | Five-card subset check: hero uses AAAKK (full house); opponent uses AAAAK (quads); strict opponent winner set, no split. | losses=1, units=0, hero `full_house`=1. |
| Four trials: win, k=1, k=2, loss | Apply ADR keys/outcome predicates once per trial. | Pot arithmetic: `420 + 420/2 + 420/3 + 0 = 770`; winner/tie/loss events are 1/2/1. | W/T/L=1/2/1, bins=(1,1,0,0,0,0), N=4, units=770, equity=`770/(420*4)=11/24`. |

For every row fixtures/tests MUST independently repeat both derivation styles: transparent rank/subset comparison and either direct winner-set/pot arithmetic or separately enumerated five-card subsets. Impossible bins `k > opponent_count` MUST be zero. In any individual trial exactly one of win, one bin, or loss occurs; category classification is independent of outcome.

## 5. Canonical serialization and hash gate

Canonical JSON is **not** the hash preimage. First validate request fields, map ADR 0002 cards to IDs, sort hero/board IDs, then concatenate ADR 0003 bytes. `seed` and `requested_trials` are excluded.

The following equivalent accepted wire requests intentionally swap hero and board order (note string `opponent_count` and string RNG version):

```json
{"contract_version":"v1","hero_cards":["As","Ah"],"board_cards":["2s","3h","Td"],"opponent_count":"2","requested_trials":"4","seed":"0x0123456789abcdef","backend":"cpu_reference","rng":{"algorithm_id":"poker-knight-ng/philox4x32-10","algorithm_version":"1"}}
{"contract_version":"v1","hero_cards":["Ah","As"],"board_cards":["Td","2s","3h"],"opponent_count":"2","requested_trials":"99","seed":"0x0000000000000000","backend":"cuda","rng":{"algorithm_id":"poker-knight-ng/philox4x32-10","algorithm_version":"1"}}
```

Both validate structurally and map to sorted IDs hero `[12,25]`, board `[0,14,34]`, then exactly:

```text
bytes hex = 17706f6b65722d6b6e696768742d6e672f636173652f7631020c1903000e2202
sha256    = fb3c0fa3e41cdd7f89e45b458f17f14174d51f285723c5178c68bd2756fec3eb
```

A distinct accepted case changing O to string `"1"` maps to bytes ending `...2201`, i.e.

```text
17706f6b65722d6b6e696768742d6e672f636173652f7631020c1903000e2201
sha256    = 02c1bd696bf6b3b2f723d597d7f3f88c04db3eb1988679638bb9f52fa998d2bd
```

This exact SHA-256 digest is normative. Future tests MUST recompute and assert exactly `02c1bd696bf6b3b2f723d597d7f3f88c04db3eb1988679638bb9f52fa998d2bd`; checking only that it differs from the equivalent-case digest is insufficient.

Phase 0 has two documented derivations for the equivalent case: **A**, direct field-boundary concatenation `0x17 || ASCII(...) || 0x02 || 0c19 || 0x03 || 000e22 || 0x02`; **B**, an independently written validated-wire transformation (temporary, untracked conformance script) produced the same hex/hash. The Phase 0 gate requires two independent tests to produce identical bytes/hash.

Replay tests MUST reproduce ADR 0003’s three Random123 Philox KAT rows and its end-to-end known-case key/deal vector: case hash above, seed `0x0123456789abcdef`, key digest `88e7ce310c8febab77f7362ee4951b9b74aa958f8005c084bad9f433ce6162ef`, key `(0x31cee788,0xabeb8f0c)`, and simulation 7 dealt IDs `[7,8,37,42,15,10]`, plus its nonzero-attempt counter row. Those ADR vectors, not legacy output, are normative.

## 6. Error and negative matrix

Schema rejection identifies malformed/closed wire data; semantic/runtime mapping selects the stable `problem.schema.json` code and never returns a success result. Details/type/status/retryability are exactly schema-owned. No error may unsafe-echo cards, seed, hashes, payload, paths, or backend exception text; no partial counters, retry/fallback, reseed, or adaptive sample count is allowed.

| Negative class / representative case | Layer | stable problem code |
|---|---|---|
| `contract_version != v1` | structural/semantic | `INVALID_CONTRACT_VERSION` |
| malformed/case/Unicode card | structural | `INVALID_CARD` |
| duplicate within/across known cards | structural then semantic cross-set | `DUPLICATE_CARD` |
| board length 1,2,6 | structural | `INVALID_BOARD_LENGTH` |
| numeric, zero, 7, or malformed opponent count | structural | `INVALID_OPPONENT_COUNT` |
| zero, leading-zero, over-bound trials | structural | `INVALID_TRIAL_COUNT` |
| malformed/noncanonical/out-of-range seed | structural | `INVALID_SEED` |
| extra closed-object/historical analytic field | structural mapping | `UNSUPPORTED_FIELD` |
| unsupported but recognizable request shape/capability | semantic | `UNSUPPORTED_REQUEST` |
| RNG ID/version other than ADR 0003 | structural/semantic | `UNSUPPORTED_RNG` |
| requested backend unavailable | pre-execution runtime | `BACKEND_UNAVAILABLE` |
| rejection at attempt UINT32_MAX | execution terminal | `RNG_REJECTION_EXHAUSTED` |
| admission/resource exhaustion | runtime | `RESOURCE_EXHAUSTED` |
| invariant breach/unclassified internal failure | runtime | `INTERNAL_ERROR` |

Negative tests MUST schema-validate both representative valid problem documents and invalid mutations, including `field_errors` only for allowed input/unsupported codes and never runtime/internal branches.

## 7. Statistical boundaries

Phase 5C statistical qualification uses fixed named seeds and trial counts from the committed seed bank. It MUST NOT use stale legacy tolerance claims as a correctness oracle. Exact rank, deal, counter, W/T/L, category, equity-unit, fraction, CPU/CUDA, and geometry invariants have zero tolerance. Statistical intervals/tolerances are preregistered before the run, fixed rather than adaptive, and appropriate to their estimand (Wilson only for named Bernoulli events; a separately documented bounded-mean interval for pot-share equity). CPU/CUDA deterministic conformance is exact integer equality, never interval agreement.

## 8. Phase 3 checkpoint D seed bank

`rng_seed_bank.json` is a generated, committed, hash-bound preregistration for deterministic CPU Monte Carlo verification. `tools/generate_rng_seed_bank.py --release` independently implements canonical encoding, Philox, rejection sampling, swap-with-tail dealing, and outcome accumulation to generate every authoritative integer counter; it then replays the generated rows through the production CPU reference before atomically publishing the bank together with `manifests/rng_seed_bank.sha256`. The dedicated manifest binds the bank and every authority input byte. `--verify` regenerates and independently verifies this complete bundle, rejecting stale, hand-edited, or unbound artifacts.

Every exact row fixes the canonical bytes/hash, seed, topology, requested trial count, and every authoritative integer counter. Verification recomputes the bytes/hash and supplies that exact hash as `replay_case_hash`; a one-bit mutation fails before a deal starts. Publication stages fsync'd files and rolls back replaced destinations if a later replacement fails; it is recoverable bundle atomicity, rather than an unsupported claim of filesystem-wide crash atomicity.

The named statistical row is also replayed exactly as a regression guard, then checked against its preregistered estimands using a fixed two-sided Wilson score interval (`alpha=0.000001`, fixed `z=4.891638475698591`). The test never adapts trials, tolerance, seeds, or interval after execution. For each W/T/L Bernoulli event, the interval is built from the observed Monte Carlo successes and fixed observed trial count, then must cover the independently enumerated exact population probability. Pot-share equity is separately qualified with fixed two-sided Hoeffding bounded-mean radius `sqrt(log(2/alpha)/(2*N))`, alpha `0.000001`, range `[0,1]`, against exact population equity `exact_units/(420*population_N)`; it is never a Wilson estimand.

## 9. Phase 0 close gate

- [x] This SPEC gives two documented derivations agreeing on every ADR 0002 tie/equity example.
- [x] This SPEC gives two canonicalization/hash derivations for the equivalent wire case; executable temporary verification repeats them.
- [x] Every required result and nested result field appears in the matrix with semantics, units/wire representation, provenance, and planned test.
- [x] Schema meta-validation and positive/negative examples are required; no analytics fields, fallback, adaptive sampling, or parallel-GPU promises enter v1.
- [x] Project-owner human approval was explicitly recorded after independent specification and quality reviews; the Phase 0 contract is approved for Phase 1.
- [x] Phase 2 checkpoint C supplies an executable independent reference oracle, generated rank/exact/tie fixtures, their SHA-256 manifest, and a separately pinned differential-engine qualification record.
- [x] Deterministic CPU stream/engine and explicit qualified CUDA engine are implemented with no fallback.
- [x] Explicit user-facing local Python API and CLI routing are implemented with no fallback.
- [ ] A network service and automatic CUDA routing remain future work.
