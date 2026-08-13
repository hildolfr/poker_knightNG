# Poker Knight NG

## Phase 2 exact-oracle checkpoint

Poker Knight NG has a reproducible Python 3.13 package boundary with zero base runtime dependencies. This checkpoint adds a transparent five-card/subset **reference evaluator** and exact Hold'em fixture oracle, plus canonical JSONL validation artifacts qualified against pinned `treys==0.1.8`. These are validation materials—not a production CPU solve backend.

```sh
uv sync --frozen --group dev
uv run pytest -q
uv run python tools/generate_oracle_fixtures.py --verify
```

The exhaustive release qualification is deliberately opt-in because it evaluates the selected flop and full AA-vs-KK preflop corpus with both engines:

```sh
RUN_ORACLE_RELEASE_QUALIFICATION=1 uv run pytest tests/oracle/test_oracle_fixture_corpus.py -q
```

There is no production CPU solver, deterministic deal stream, CUDA backend, or service in this checkpoint. CUDA remains an optional future capability; a CUDA request is never silently downgraded. The binding v1 authorities are `docs/adr/`, `contracts/v1/`, and `validation/holdem/v1/SPEC.md`.
