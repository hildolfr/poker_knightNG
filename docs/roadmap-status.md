# Roadmap status

This document is the maintained project-status index for Poker Knight NG. It is
derived from the binding authorities in [`docs/adr/`](adr/),
[`contracts/v1/`](../contracts/v1/), and
[`validation/holdem/v1/SPEC.md`](../validation/holdem/v1/SPEC.md). If this index
conflicts with a binding authority, the binding authority wins.

`TODO.md` is retained as non-authoritative legacy evidence. Its historical
checkboxes, architecture, dependencies, and performance claims are not the
current roadmap.

**Status baseline:** promoted default-branch commit
`15e49a5e8d88bcca6395ec07c02aacf388996ac4` on `main`.

## Status meanings

| Status | Meaning |
|---|---|
| **Complete** | Implemented, tested, independently reviewed, integrated, pushed, and evidence-bound where required |
| **Active** | Implemented and currently enabled in the live repository or service state, with bounded evidence where applicable |
| **Implemented** | The maintained mechanism or procedure exists and is tested, but does not imply an external release or deployment |
| **Not started** | Authoritative or proposed checkpoint has no approved implementation checkpoint yet |
| **Untouched** | No implementation or qualification work has begun |
| **Deferred outside v1** | Deliberately excluded from the v1 contract; a future versioned decision is required |
| **Prohibited in v1** | v1 must reject or fail rather than silently provide this behavior |
| **Superseded** | Legacy checklist item replaced by the binding revival contract |

## Authoritative revival roadmap

| Phase / checkpoint | Status | Delivered or remaining |
|---|---|---|
| Phase 0 — contract and scope | **Complete** | Binding v1 request/result/problem schemas, exact integer semantics, closed errors, no-fallback policy, and ADRs |
| Phase 1 — package boundary | **Complete** | Python 3.13 package, zero base runtime dependencies, canonical serialization, immutable validated models, and wheel/sdist verification |
| Phase 2A — transparent evaluator | **Complete** | Independent five-card and best-five-of-seven reference evaluator |
| Phase 2B — exact fixtures | **Complete** | Canonical rank, exact-case, and tie/equity fixtures with hash-bound manifests |
| Phase 2C — exhaustive qualification | **Complete** | Complete `C(52,7) = 133,784,560` category enumeration and differential qualification |
| Phase 3A — deterministic RNG | **Complete** | Philox4x32-10 implementation, upstream KATs, and project derivation vectors |
| Phase 3B — deterministic dealer | **Complete** | Canonical case-hash enforcement, rejection-sampled dealing, fixed draw order, and replay integrity |
| Phase 3C — CPU Monte Carlo | **Complete** | Deterministic integer-only CPU engine with exact counters and equity units |
| Phase 3D — seed authority | **Complete** | Generated, replay-verified, hash-bound RNG seed bank |
| Phase 4 — CUDA engine | **Complete** | Explicit CUDA executor, geometry-independent deterministic results, and exact CPU/CUDA equality |
| Phase 5A — GPU qualification harness | **Complete** | Exact-source, exact-wheel, fail-closed qualification with cold/warm/PTX and sanitizer gates |
| Phase 5B — qualification publication | **Complete** | Privacy-safe public CUDA qualification authority |
| Phase 5C-A — statistical harness | **Complete** | Frozen geometries, deterministic seed jobs, Wilson/Hoeffding gates, and exact aggregate checks |
| Phase 5C-B — statistical execution | **Complete** | Live qualified GPU execution and privacy-safe public statistical authority |
| Phase 6 — local API and CLI | **Complete** | `solve()`, `solve_cuda()`, `CUDAEngine`, `solve`, and `solve-cuda`; bounded canonical JSON and closed exits |
| Phase 6C-A — benchmark harness | **Complete** | Uncontested, identity-bound, privacy-safe benchmark protocol |
| Phase 6C-B — baseline | **Complete** | First valid public performance baseline |
| Phase 6C-C — evaluator optimization | **Complete** | Direct seven-card CUDA evaluator, requalification, comparison, and accepted replacement baseline |
| Phase 7A — network service contract | **Complete** | Accepted [ADR 0005](adr/0005-private-unix-http-service.md) and the hash-bound service profile are on `main` at `cd994d45516b257515c03f6225b4242babde0cc5`; [default-branch verification](https://github.com/hildolfr/poker_knightNG/actions/runs/31862456329) passed. No service listener is implemented or deployed |
| Phase 7B — bounded network service | **Active** | No listener is implemented. [Spike 001](../spikes/001-http-runtime-selection/README.md) selected direct `h11==0.16.0` and is on `main` at `08f97ac10337f945eaf1bf993be3b0faaa5a8955`; [default-branch verification](https://github.com/hildolfr/poker_knightNG/actions/runs/31864664096) passed. The isolated raw framing package is active under accepted [ADR 0006](adr/0006-isolated-http-service-package.md); Uvicorn and aiohttp remain rejected |
| Phase 7C — automatic CUDA routing contract | **Not started** | Requires a new ADR/versioned compatibility decision because v1 prohibits implicit routing and fallback |
| Phase 7D — automatic routing implementation | **Untouched** | Define backend admission, deterministic selection, resource policy, and explicit failure behavior |
| Phase 7E — network/runtime qualification | **Untouched** | Validate load, cancellation, resource exhaustion, privacy, security boundaries, and deployment behavior |
| Phase 7F — deployment | **Untouched** | No production service, container/systemd deployment, monitoring, or operator runbook exists |

The binding SPEC currently has one unchecked combined item: a network service
and automatic CUDA routing. Phase 7A–7F is the proposed gated decomposition of
that item; these subphase names are not binding until approved in an ADR or SPEC
amendment.

## Deferred product surface

These capabilities are explicitly outside v1. They are not partially
implemented requirements and must not be accepted, ignored, zero-filled, or
silently approximated through a v1 interface.

| Capability | Status | Contract position |
|---|---|---|
| ICM | **Deferred outside v1** | Requires a future versioned contract |
| Tournament stacks and payouts | **Deferred outside v1** | Outside v1 |
| Ranges | **Deferred outside v1** | Outside v1 |
| Weighted opponents | **Deferred outside v1** | Outside v1 |
| Position modelling | **Deferred outside v1** | Outside v1 |
| Fold equity | **Deferred outside v1** | Outside v1 |
| GTO, strategy, or advice | **Deferred outside v1** | Outside v1 |
| Pot odds and MDF | **Deferred outside v1** | Outside v1 |
| Board texture and draw interpretation | **Deferred outside v1** | Outside v1 |
| Vulnerability analysis | **Deferred outside v1** | Outside v1 |
| Percentiles | **Deferred outside v1** | Outside v1 |
| Side pots | **Deferred outside v1** | Outside v1 |
| Betting/action trees | **Deferred outside v1** | Outside v1 |
| Multiple poker variants | **Deferred outside v1** | Hold'em only in v1 |
| Public internet exposure | **Deferred outside v1** | No public service exists |
| Accounts, billing and multitenancy | **Deferred outside v1** | Outside v1 |
| Multi-GPU balancing | **Deferred outside v1** | Must not alter deterministic identity |
| Streaming progress | **Deferred outside v1** | Outside v1 |
| WebSockets | **Deferred outside v1** | Outside v1 |
| Result caching | **Deferred outside v1** | Outside v1 |
| Camelot changes | **Deferred outside v1** | Outside v1 |
| Adaptive trial counts | **Prohibited in v1** | A success must complete the exact requested trial count |
| Timeout-based convergence | **Prohibited in v1** | Cannot return a shortened successful result |
| Implicit reseeding | **Prohibited in v1** | Breaks replay identity |
| Silent CPU/GPU fallback | **Prohibited in v1** | An unavailable requested backend must fail explicitly |
| Advanced analytics | **Prohibited in v1** | Cannot enter the closed v1 result schema |

## Release and operational status

| Item | Status | Detail |
|---|---|---|
| Accepted default branch | **Complete** | `main` at `15e49a5e8d88bcca6395ec07c02aacf388996ac4` |
| Full maintained test suite | **Complete** | 801 passed and 3 skipped at the promoted checkpoint |
| CUDA qualification | **Complete** | Exact-wheel GPU and sanitizer evidence published |
| Performance acceptance | **Complete** | 64.742% lower geometric-mean p50 with no p50 or p95 regressions |
| Wheel and sdist verification | **Complete** | Fresh artifact installation and content checks passed |
| Promotion to default `main` | **Complete** | Legacy `main` was fast-forwarded without squash, rebase, or merge commit to exact reviewed SHA `15e49a5e8d88bcca6395ec07c02aacf388996ac4`; [default-branch verification passed](https://github.com/hildolfr/poker_knightNG/actions/runs/31859196286) |
| Main branch protection | **Active** | Strict required `verify` status, linear history, force-push/deletion denial, and disabled administrator enforcement are captured in the canonical [protection evidence](evidence/main-branch-protection.json) and SHA-256 manifest |
| PyPI publication | **Untouched** | No package release has been published |
| GitHub release | **Untouched** | No GitHub release exists |
| Revival/release tag | **Untouched** | Only the preserved legacy tag exists |
| Automated CPU CI | **Active** | Pinned read-only verification passed on the source candidate, final closeout, pull request, and promoted default branch; it covers full tests, authorities, package build, and isolated wheel smoke testing |
| Release procedure | **Implemented** | `docs/release-process.md` freezes approval, history-preservation, privacy, publication, and rollback gates |
| Automated release pipeline | **Untouched** | No sign or publish workflow exists |
| Network deployment | **Untouched** | No long-running API service has been built or deployed |
| Production observability | **Untouched** | Network-service health, metrics, and operator alerts remain future work |
| Security exposure review | **Not started** | Service threat modelling begins with Phase 7A |

## Legacy `TODO.md` reconciliation

| Legacy entry | Current disposition |
|---|---|
| Comprehensive public docstrings | **Not formally audited** as a dedicated checkpoint |
| Usage examples for all API features | **Partially superseded** by current CPU, CUDA, and CLI README examples |
| Update `CLAUDE.md` | **Superseded** and not a binding revival requirement |
| Full API compliance suite | **Complete under the v1 contract** through maintained schema, semantic, CPU, CUDA, CLI, and artifact tests |
| Package and test pip installation | **Artifact verification complete**; public PyPI release remains untouched |
| Historical latency target | **Superseded**; qualified Phase 6C evidence is the only current performance authority |

## Next checkpoint order

Work advances one reviewed checkpoint at a time:

1. keep this roadmap index synchronized with binding authorities;
2. audit and execute promotion/release readiness;
3. specify Phase 7A before writing network-service code;
4. implement and qualify the bounded service;
5. decide automatic routing separately through a new compatibility ADR; and
6. consider deferred product capabilities only through explicit future contracts.

Every completed checkpoint requires executable verification, independent
specification and quality review where applicable, a clean commit, remote SHA
verification, and updated evidence here.
