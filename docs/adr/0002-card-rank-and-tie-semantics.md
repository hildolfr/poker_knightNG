# ADR 0002: Card, rank, outcome, tie, and equity semantics

- **Status:** Accepted
- **Date:** 2026-08-12
- **Decision:** Freeze the v1 card notation and identity mapping, five- and seven-card rank ordering, result accounting, and exact showdown-equity semantics.

## Context

ADR 0001 freezes v1 scope and the meanings of summary tie and pot-share results, including `unique_wins`, `tie_by_other_winners`, `losses`, `ties`, and the denominator-`420` equity accounting. This ADR expands those established meanings into normative card, evaluator, category, outcome, width, and example rules; it does not redefine ADR 0001.

The preserved implementation and its tests are evidence only, not an oracle. In particular, they use a different card-ID mapping, permit non-canonical display/input forms, and expose a ten-category output that includes `royal_flush`. Those behaviors conflict with this ADR and are superseded for v1.

## Decision

All requirements in this section are binding for v1. `MUST` and `MUST NOT` have their usual RFC-style normative meanings.

### 1. External cards and known-card validity

A public v1 card is exactly a two-character ASCII token: rank followed by suit.

- Rank characters are exactly `2 3 4 5 6 7 8 9 T J Q K A`.
- Suit characters are exactly `s h d c`.
- Examples: `As`, `Th`, and `2c`.
- Input is case-sensitive and exact. Lowercase ranks, uppercase suits, Unicode suit glyphs, and any other spellings are **not** canonical v1 inputs.
- Whitespace is not part of a card token. A request representation that separates tokens may define its own surrounding structural syntax later, but it MUST NOT trim, normalize, or reinterpret whitespace inside a token.

All known cards across hero and board MUST be distinct. A duplicate known card is invalid and MUST cause rejection before execution. The hero has exactly two known cards; the inherited board lengths are only `0`, `3`, `4`, or `5`; and the inherited opponent range is `1..6`.

### 2. Internal card IDs

The canonical internal card ID range is `0..51`:

```text
card_id = suit_index * 13 + rank_index
rank_index: 2..A = 0..12
suit_index: s=0, h=1, d=2, c=3
```

Thus `2s = 0`, `As = 12`, `2h = 13`, `Th = 21`, and `2c = 39`. Implementations MUST use this mapping wherever an internal card ID is required by the v1 contract. Card IDs are contract-visible only in deterministic fixtures and traces; public requests use canonical ASCII cards, not numeric IDs.

### 3. Five-card rank key

Every five-card hand MUST have one fixed-length, lexicographically comparable rank key:

```text
(category, k1, k2, k3, k4, k5)
```

All six members are integers. Higher tuples always win under ordinary lexicographic comparison. Unused kicker positions MUST be zero-filled. Category integers are exactly:

| Category name | Integer |
| --- | ---: |
| `high_card` | 0 |
| `one_pair` | 1 |
| `two_pair` | 2 |
| `three_of_a_kind` | 3 |
| `straight` | 4 |
| `flush` | 5 |
| `full_house` | 6 |
| `four_of_a_kind` | 7 |
| `straight_flush` | 8 |

Ranks in keys use natural poker values `2..14` (`A = 14`), except that an ace-low wheel straight (`A-2-3-4-5`) and wheel straight flush encode high card `5`. A royal flush is a display-only subtype of `straight_flush` whose high card is `14`; it is not another category and MUST NOT have a separate category-count bucket.

The exact key layouts are:

| Category | Key layout |
| --- | --- |
| `high_card` | `(0, high, next, next, next, low)` — five distinct ranks descending |
| `one_pair` | `(1, pair, kicker1, kicker2, kicker3, 0)` — kickers descending |
| `two_pair` | `(2, high_pair, low_pair, kicker, 0, 0)` |
| `three_of_a_kind` | `(3, trips, kicker1, kicker2, 0, 0)` — kickers descending |
| `straight` | `(4, high, 0, 0, 0, 0)` |
| `flush` | `(5, high, next, next, next, low)` — five flush ranks descending |
| `full_house` | `(6, trips, pair, 0, 0, 0)` |
| `four_of_a_kind` | `(7, quads, kicker, 0, 0, 0)` |
| `straight_flush` | `(8, high, 0, 0, 0, 0)` |

Where more than one candidate exists within seven cards, the key is determined by the best five-card subset, not by a shortcut category label. In particular, with two trip ranks, the higher trip rank is the `trips` component and the lower trip rank is the `pair` component of the best full house.

### 4. Seven-card evaluation and hero categories

A completed Texas Hold'em hand has seven cards: two hole cards plus the complete five-card board. Its key MUST be the maximum lexicographic key among all `C(7,5) = 21` five-card subsets.

For every completed trial, the hero final-hand category count MUST classify that maximum hero key. It is independent of whether the hero wins, ties, or loses, and exactly one of the nine category counters MUST increment per completed trial.

### 5. Outcomes and tie multiplicity

For a completed trial, compare the hero's seven-card key with every opponent's seven-card key.

- `unique_wins` increments **iff** the hero key is strictly greater than every opponent key.
- `ties` increments **iff** no opponent key is greater than the hero key and at least one opponent key equals it.
- `losses` increments **iff** at least one opponent key is greater than the hero key.

For a tie, define `k` as the number of **other winners**: the number of opponent keys equal to the hero key. Thus total winners are `k + 1`, including hero. `tie_by_other_winners[k]` increments only for a hero tie with exactly `k` equal-key opponent co-winners, for `k = 1..6`. Bins impossible for the request's opponent count MUST remain zero. This definition does not count opponents that lose to the tied winning key.

Exactly one of `unique_wins`, one `tie_by_other_winners[k]` bin, or `losses` MUST increment in each completed trial. Therefore:

```text
ties = sum(k=1..6, tie_by_other_winners[k])
unique_wins + ties + losses = completed_trials
sum(all nine hero category counts) = completed_trials
```

### 6. Exact showdown equity and numeric widths

The common pot-share denominator is `420`, the least common multiple of possible total-winner counts `1..7`. For each completed trial:

- a unique win adds `420` equity-share units;
- a tie with `k` other winners adds `420 / (k + 1)` units; and
- a loss adds `0` units.

Consequently:

```text
equity_share_units = 420 * unique_wins
                   + sum(k=1..6, (420 / (k + 1)) * tie_by_other_winners[k])
showdown_equity = equity_share_units / (420 * completed_trials)
```

This is exact integer pot-share accounting before the final host-derived division. Event probabilities are different quantities: `unique_wins / completed_trials`, `ties / completed_trials`, and `losses / completed_trials` describe win/tie/loss events and MUST NOT be presented as pot-share equity.

Authoritative outcome counters, all tie bins, hero category counters, `completed_trials`, and `equity_share_units` MUST be unsigned 64-bit values. Before execution, an implementation MUST require `1 <= requested_trials <= floor((2^64 - 1) / 420)` and MUST reject any request outside that range, or any request for which another required accumulation would overflow. Because every completed trial can be a unique win, `equity_share_units` can reach `420 * requested_trials`. A successful execution has `completed_trials = requested_trials`, so the equity denominator is nonzero. This agrees with ADR 0001 and intentionally does not set a lower operational maximum-trials constant; schema validation may impose a lower positive maximum later.

## Normative examples

The examples use canonical ASCII cards. A listed five-card key is the required result for that five-card hand; a listed seven-card key is the maximum over its 21 subsets.

### Card identity and input examples

| Token / ID | Required interpretation |
| --- | --- |
| `As` | canonical public card; ID `12` |
| `Th` | canonical public card; ID `21` |
| `2c` | canonical public card; ID `39` |
| `A♠`, `as`, `AS`, `T h` | invalid canonical v1 card tokens |
| hero `As Ah`; board `As 7d 4c` | invalid request: duplicate known `As` |

### Rank-key examples

| Cards | Required key | Why |
| --- | --- | --- |
| `As 2h 3d 4c 5s` | `(4, 5, 0, 0, 0, 0)` | wheel: ace plays low |
| `2s 3h 4d 5c 6s` | `(4, 6, 0, 0, 0, 0)` | six-high straight; it beats the wheel |
| `Ah Kh 9h 5h 2h` | `(5, 14, 13, 9, 5, 2)` | flush kickers descend; this beats `Ah Qh Jh 8h 3h`, `(5, 14, 12, 11, 8, 3)` |
| `As Ad Kc Kd Qh` | `(2, 14, 13, 12, 0, 0)` | two-pair kicker is `Q` |
| `As Ah Ad Kc Kd 2s 2h` | `(6, 14, 13, 0, 0, 0)` | double trips: aces full of kings, not kings full of aces |
| `As Ah Ad Ac Kd` | `(7, 14, 13, 0, 0, 0)` | quads kicker is `K`; it beats `As Ah Ad Ac Qd`, `(7, 14, 12, 0, 0, 0)` |
| `Ts Js Qs Ks As` | `(8, 14, 0, 0, 0, 0)` | displayable as royal flush, counted only as `straight_flush` |

### Seven-card, board-playing equality, and accounting examples

1. **Board-playing equality.** Board `As Ks Qs Js Ts`; hero `2c 3d`; one opponent `4h 5c`. Both seven-card keys are `(8, 14, 0, 0, 0, 0)` because the board alone plays. The trial increments `tie_by_other_winners[1]` and `ties`, not `unique_wins` or `losses`; it adds `420 / 2 = 210` equity-share units. The hero category increment is `straight_flush`.
2. **Two equal-key opponent co-winners.** Board `As Ks Qs Js Ts`; hero `2c 3d`; opponents `4h 5c` and `6h 7c`. All three players have the same board-playing key. The trial increments only `tie_by_other_winners[2]` and `ties`; it adds `420 / 3 = 140` units. The total winner count is three, while `k` is two.
3. **An opponent can lose while a tie is recorded.** Board `As Ks Qs Js 2d`; hero `Tc 3c`; opponent 1 `Th 4c`; and opponent 2 `8h 8d`. Hero and opponent 1 tie with `(4, 14, 0, 0, 0, 0)` (broadway straight), while opponent 2 loses with a pair. This increments only `tie_by_other_winners[1]`, because `k` counts equal-key co-winners, not all opponents.
4. **Loss has no equity credit.** Board `As Ah Ad Kc Qd`; hero `Ks Kd` has `(6, 14, 13, 0, 0, 0)` (aces full of kings); an opponent holding `Ac Qc` has `(7, 14, 13, 0, 0, 0)` (aces quads, king kicker). The trial increments only `losses`, contributes zero equity-share units, and still increments the hero `full_house` category.

For a concrete aggregate accounting check over four completed trials—one unique win, one `k=1` tie, one `k=2` tie, and one loss—the required counters are:

```text
unique_wins = 1
tie_by_other_winners[1] = 1
tie_by_other_winners[2] = 1
all other tie bins = 0
ties = 2
losses = 1
completed_trials = 4
equity_share_units = 420 + 210 + 140 = 770
showdown_equity = 770 / (420 * 4) = 11 / 24
```

The hero category counters for that aggregate remain a separate partition and must sum to four regardless of those outcome counters.

## Consequences

Future request, fixture, trace, CPU-reference, and CUDA-production work MUST conform to these meanings. Implementations must not retain legacy rank-major/club-first card IDs, Unicode-or-case-insensitive public parsing, `royal_flush` as category `9`, or a ten-category output. Legacy expectations with those semantics are explicitly superseded rather than grandfathered into v1.

This ADR does not choose deal order, RNG, schema bytes, APIs, implementation algorithms, or a maximum trials constant. Those remain later compatibility and validation decisions.
