# Poker Knight NG

## Phase 2 exact-oracle checkpoint

Poker Knight NG has a reproducible Python 3.13 package boundary with zero base runtime dependencies. This checkpoint adds a transparent five-card/subset **reference evaluator** and exact Hold'em fixture oracle, plus canonical JSONL validation artifacts qualified against pinned `treys==0.1.8`. These are validation materials—not a production CPU solve backend.

```sh
uv sync --frozen --group dev
uv run pytest -q
uv run python tools/generate_oracle_fixtures.py --verify
```

The exhaustive seven-card release qualification is deliberately opt-in; it traverses every `C(52,7)=133,784,560` unordered hand and records signed evidence. The exact C category function used by that exhaustive counter is directly differentially checked, per hand, against both the transparent 21-subset evaluator and pinned `treys==0.1.8` on a deterministic 10,000-hand all-category/wheel sample; canonical full-space totals provide the complete-corpus conservation gate.

```sh
RUN_SEVEN_CARD_RELEASE_QUALIFICATION=1 uv run python tools/qualify_seven_card_corpus.py --release
uv run python tools/qualify_seven_card_corpus.py --verify
```

There is no production CPU solver, deterministic deal stream, CUDA backend, or service in this checkpoint. CUDA remains an optional future capability; a CUDA request is never silently downgraded. The binding v1 authorities are `docs/adr/`, `contracts/v1/`, and `validation/holdem/v1/SPEC.md`.
