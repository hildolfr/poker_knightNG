# ADR 0001: Equity v1 scope

- **Status:** Accepted
- **Date:** 2026-08-12
- **Decision:** Freeze a small, deterministic Texas Hold'em equity contract before replacing implementation code.

## Context

Poker Knight NG's legacy snapshot is preserved as evidence and as a source of possible test vectors; it is **not** an oracle. In particular, the current unsafe CUDA kernel must not be incrementally repaired. The revival will replace the simulation, evaluator, RNG, reduction, and result boundary behind a narrow, versioned contract.

## Decision

### Supported v1 input and execution surface

Equity v1 supports only all of the following:

- standard 52-card Texas Hold'em;
- exactly two known hero hole cards;
- board lengths `0`, `3`, `4`, or `5`;
- `1..6` uniformly random opponents;
- a fixed caller-requested trial count; and
- an explicit unsigned 64-bit (`uint64`) seed.

A CPU reference backend remains available for deterministic replay. Its exact-enumeration mode is a verification/oracle mechanism for fixtures and exhaustive cases; the caller-facing stochastic solve contract remains a fixed requested trial count plus seed. A CUDA backend is the production backend. These names describe the intended contract boundary, not a claim that either new implementation is complete at this checkpoint.

### Result surface

v1 results are limited to these families:

1. raw integer win/tie/loss (W/T/L) outcome counts;
2. exact tie multiplicity;
3. exact accumulated pot-share units with denominator `420`;
4. hero final-hand category counts; and
5. host-derived probabilities, timing, and provenance.

No other result family is part of v1. In particular, probabilities are derived on the host from integer counts; they are not a substitute for raw counts.

For each completed trial:

- `unique_wins` means hero is the only player with maximum hand rank.
- `tie_by_other_winners[k]` means hero has maximum rank with exactly `k` other co-winners, for `k = 1..6`.
- `ties = sum(tie_by_other_winners)` is an event count, not pot-share equity.
- `losses` means at least one opponent has strictly higher rank.
- The outcome partition is `unique_wins + ties + losses == completed_trials`. A successful v1 execution has `completed_trials == requested_trials`.
- With denominator `420`, a unique win contributes `420` units; a tie with `k` other winners (`k + 1` total winners) contributes `420/(k + 1)` units; and a loss contributes `0`.
- `showdown_equity = equity_share_units/(420*completed_trials)`.

Pot-share units therefore represent split-pot equity exactly while ties remain outcome events.

### Explicitly rejected/deferred surface

The following roadmap surface is outside v1 and must be rejected explicitly if requested or supplied through a legacy-shaped interface. It must not be ignored, accepted and zero-filled, or silently approximated:

- ICM;
- tournament stacks/payouts;
- ranges;
- weighted opponents;
- position;
- fold equity;
- GTO/strategy/advice;
- pot odds/MDF;
- board texture/draw interpretation;
- vulnerability;
- percentiles;
- side pots;
- betting/action trees;
- multiple poker variants;
- public internet exposure;
- accounts/billing/multitenancy;
- multi-GPU balancing;
- streaming progress;
- WebSockets;
- result caching; and
- Camelot changes.

### Reliability constraints

A successful v1 execution completes exactly the requested number of trials. v1 forbids:

- silent fallback;
- sample adaptation (including adaptive sample reduction);
- timeout convergence or timeout-based stopping;
- implicit reseeding; and
- advanced analytics.

An unavailable or unsupported requested capability is an explicit error, not a degraded calculation. Backend availability and qualification are reported through provenance rather than concealed by fallback behavior.

## Consequences

The initial implementation is intentionally narrower than the legacy package and may reject historical arguments and fields. This makes conformance testable: CPU verification and CUDA production must implement the same versioned integer result contract. Timing and probability presentation remain host responsibilities, and raw deterministic outcomes remain inspectable.

This ADR does not freeze card encoding, deal order, exact hand-rank comparison representation, schema bytes, RNG selection, CUDA reduction details, service topology, or validation fixtures. Those compatibility details belong to later ADRs and contract/validation artifacts. Until they exist, this document is the v1 scope boundary rather than an implementation promise.

## Future compatibility boundary

Future versions may add capabilities only through an explicit versioned contract and compatibility decision. They must preserve the meaning of v1 result families for v1 requests, or reject incompatible requests explicitly. Deferred surface must not enter v1 through optional fields, undocumented defaults, zero-valued output, or implicit backend behavior.
