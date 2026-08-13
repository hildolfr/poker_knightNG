# Phase 2 checkpoint C — oracle fixture qualification

## Status

**PASS — 2026-08-13.** The final checkpoint-C `uv run python tools/generate_oracle_fixtures.py --release` completed successfully in **196.236 seconds**. It regenerated all canonical bytes only after the transparent reference and independently pinned **treys==0.1.8** agreed on every exact/tie counter, category bin and 420-unit total. The checked-in manifest is bound to this final generator source.

The release corpus covers unknown river (990 deals), unknown turn (45,540), selected unknown flop `As Kd` / `2s 7h 9d` (1,070,190), fixed-hole turn (44), AA-vs-KK preflop `C(48,5)=1,712,304`, terminal loss, plus all six shared-board tie bins. Treys royal class 0 is normalized to v1 `straight_flush`.

## Commands

```sh
uv run python tools/generate_oracle_fixtures.py --verify
RUN_ORACLE_RELEASE_QUALIFICATION=1 uv run pytest tests/oracle/test_oracle_fixture_corpus.py -q
# or: uv run python tools/generate_oracle_fixtures.py --release --output /tmp/v1
```

`--verify` is deliberately fast: it validates strict canonical JSONL structure, frozen qualified semantics, conservation rules, manifest grammar/order, and every checked-in SHA-256 binding without rerunning exhaustive enumeration. `--release` is the explicit exhaustive transparent+Treys gate. It constructs outputs in memory and stages every destination before publication. Publication is **failure-atomic with rollback for reported publication failures; individual files use atomic replace** (it does not claim impossible cross-file crash atomicity). The manifest binds corpus and authority source bytes, not an impossible self-referential commit identity.

This is an exact-reference oracle and fixture qualification, **not** a production CPU solve backend, deterministic stream, CUDA implementation, or service. Phase 2's full `C(52,7)=133,784,560` seven-card release-candidate corpus is complete: `seven_card_release_qualification.json` records its exact nine-category totals, deterministic command, committed first-successful runtime for the unchanged qualification/source state, implementation SHA-256 bindings, and differential scope. Repeat runs still report their current measured runtime in release stdout, but retain the validated committed elapsed value so timing alone cannot rewrite hash-bound evidence or its manifest. Run `RUN_SEVEN_CARD_RELEASE_QUALIFICATION=1 uv run python tools/qualify_seven_card_corpus.py --release` to repeat the expensive gate; `uv run python tools/qualify_seven_card_corpus.py --verify` validates the committed evidence and bindings quickly. The exact C category function used by the full-space counter is directly checked per hand against both the transparent 21-subset authority and Treys 0.1.8 on the deterministic 10,000-hand all-category/wheel sample; exact canonical totals independently constrain every full-space category.
