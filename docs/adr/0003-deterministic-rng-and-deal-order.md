# ADR 0003: Deterministic RNG and deal order

- **Status:** Accepted
- **Date:** 2026-08-12
- **Decision:** Freeze v1 case bytes, hash/key derivation, Philox4x32-10, bounded selection, and deal order so every conforming backend exactly replays a case, seed, and simulation-ID range.

## Context

ADR 0001 requires a fixed caller-requested trial count and forbids implicit reseeding, adaptation, and fallback. ADR 0002 fixes card IDs and known-card validation. The legacy launch-geometry-dependent stream and unsupported antithetic claim are not a replayable CPU/CUDA contract. RFC-style `MUST`, `MUST NOT`, and `SHOULD` are normative.

## Decision

### 1. Identity, scope, and simulation IDs

```text
algorithm_id      = "poker-knight-ng/philox4x32-10"
algorithm_version = 1
```

A v1 request/result/trace MUST report both and reject an unsupported identity/version before execution. Sampling is IID: no antithetic pairing, complementing, mirrored deal, common-random-number scheme, stratification, or variance reduction is part of v1.

A successful request for `N` trials executes logical simulation IDs `0..N-1`, in that logical order. Scheduling may differ but MUST preserve each ID's deal/outcome. ADR 0002 bounds `N <= floor((2^64-1)/420)`, so every ID is `uint64`.

### 2. Canonical v1 case bytes and hash

Before this encoding, validate ADR 0002 public cards and v1 ranges: exactly two distinct hero cards; board length `B` in `{0,3,4,5}`; opponent count `O` in `1..6`; and all known cards distinct. Card IDs are ADR 0002's `suit_index * 13 + rank_index`. Hero and board are semantically unordered sets, so each MUST be sorted by ascending `card_id` before encoding.

`canonical_case_bytes` is exactly this byte concatenation; all numeric fields are one unsigned octet (`uint8`), with no padding, delimiter, NUL, JSON, Unicode normalization, or omitted field:

```text
uint8(23) || ASCII("poker-knight-ng/case/v1") ||
uint8(2)  || hero_card_id[0] || hero_card_id[1] ||
uint8(B)  || board_card_id[0] || ... || board_card_id[B-1] ||
uint8(O)
```

The length byte is part of the bytes and denotes exactly the following ASCII-domain length. The hero-count byte is fixed at `2` and is retained to make the field boundary explicit. `canonical_case_hash = SHA-256(canonical_case_bytes)`, exactly 32 digest bytes in standard SHA-256 output byte order (the order emitted by FIPS 180-4 / ordinary SHA-256 APIs). This ADR does not use digest words `H`; implementations MUST NOT reinterpret, fold, truncate, or caller-supply a substitute hash.

Normal requests MUST derive these bytes and hash internally, and MUST verify any supplied replay hash before execution; mismatch is a pre-execution error. A fixture replay may supply a recorded hash only together with matching recorded canonical bytes. Later request schemas/canonicalizers MUST produce precisely this encoding. The hash excludes seed and requested trials: a smaller solve remains the simulation-ID prefix of a larger solve for the same case/seed.

### 3. Remaining deck and logical deal slots

For every simulation, create a fresh mutable `deck` by iterating IDs `0..51` ascending and retaining IDs absent from known hero/board cards. It is the canonical-ID ordered remaining deck, not a physical shuffle.

```text
D  = (5 - B) + 2*O
R0 = 52 - (2 + B)
```

Supported ranges give `2 <= D <= 17`, `R0 = 50-B`, and `R0-D >= 33`; every selection range is nonempty. Invalid ranges or deck capacity fail before execution.

Slots `0..D-1` are fixed before evaluation: slots `0..(5-B-1)` fill missing board positions in ascending board-slot order (`B..4`); then opponents `0..O-1`, each hole card 0 then 1. There are no burn cards, dealer button, or casino-dealing convention.

### 4. Key derivation and Philox input

All arithmetic below is unsigned fixed-width. `uint32` addition, multiplication results assigned to `uint32`, and XOR wrap modulo `2^32`; `uint64` is exact modulo `2^64` where stated. Encode `seed` as eight little-endian bytes `LE64(seed)`.

Derive a per-case/seed key digest using this exact, domain-separated preimage:

```text
key_digest = SHA-256(uint8(26) || ASCII("poker-knight-ng/rng-key/v1") ||
                      LE64(seed) || canonical_case_hash)
key = (LE32(key_digest[0..3]), LE32(key_digest[4..7]))
```

Byte ranges are inclusive. The complete 32-byte case digest is an input to this SHA-256 preimage, so this is not a weak XOR fold. This is deterministic domain separation, **not** a claim of cryptographic unpredictability or secrecy.

For `simulation_id s`, `draw_slot d`, and `rejection_attempt a`, all representable without truncation, the initial Philox lanes are:

```text
counter = (uint32(s), uint32(s >> 32), uint32(d), uint32(a))
key     = derived key above
```

`d <= 16`; `a` is `uint32`; implementations MUST fail rather than truncate. A candidate is a pure function of algorithm identity/version, canonical bytes/hash, explicit seed, `s`, `d`, and `a`, with no thread, block, grid, batch, stream, retry placement, device, or GPU-count dependency.

### 5. Normative Philox4x32-10 pseudocode

`mullo(x,y)` and `mulhi(x,y)` are respectively the low/high `uint32` halves of the exact unsigned `uint64(x)*uint64(y)` product. The input/output tuple order is lane order shown below.

```text
M0 = 0xD2511F53; M1 = 0xCD9E8D57
W0 = 0x9E3779B9; W1 = 0xBB67AE85

philox4x32_10(c0,c1,c2,c3, k0,k1):
  for round = 0 to 9 inclusive:
    hi0 = mulhi(M0, c0); lo0 = mullo(M0, c0)
    hi1 = mulhi(M1, c2); lo1 = mullo(M1, c2)
    (c0,c1,c2,c3) = (hi1 XOR c1 XOR k0, lo1,
                      hi0 XOR c3 XOR k1, lo0)
    if round != 9:
      (k0,k1) = (k0 + W0, k1 + W1)       // uint32 wrap
  return (c0,c1,c2,c3)
```

Thus there are exactly ten rounds and key bumping occurs after rounds 0..8 only. Use output lane 0 only; lanes 1..3 are discarded and MUST NOT be consumed by another slot/attempt.

The following official DEShaw Random123 `tests/kat_vectors` values are normative cross-checks (source: <https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors>):

| case | counter `(c0,c1,c2,c3)` | key `(k0,k1)` | output `(c0,c1,c2,c3)` |
| --- | --- | --- | --- |
| zero | `(00000000,00000000,00000000,00000000)` | `(00000000,00000000)` | `(6627e8d5,e169c58d,bc57ac4c,9b00dbd8)` |
| all ones | `(ffffffff,ffffffff,ffffffff,ffffffff)` | `(ffffffff,ffffffff)` | `(408f276d,41c83b0e,a20bc7c6,6d5451fd)` |
| digits of pi | `(243f6a88,85a308d3,13198a2e,03707344)` | `(a4093822,299f31d0)` | `(d16cfe09,94fdcceb,5001e420,24126ea1)` |

The URL is verification provenance only; this ADR's pseudocode and tables are the semantics.

### 6. Unbiased partial Fisher-Yates swap-with-tail selection

Immediately before slot `d`, `R = R0-d`. For attempts beginning at `a=0`, let `w` be Philox output lane 0 and compute with exact `uint64` (or wider):

```text
q = floor(2^32 / R)
L = q * R
```

Require `0 < L <= 2^32` without narrowing/wrapping `L`. If `L == 2^32`, accept every `w`; otherwise accept iff `w < L`. On acceptance, select `i = w mod R`, emit `deck[i]`, then exactly:

```text
deck[i] = deck[R-1]
R = R-1
```

This is partial Fisher-Yates **swap-with-tail selection**, removes from active prefix `[0,R)`, and prevents recurrence. On rejection do not modulo-map: retry only the same `(s,d)` at `a+1`.

Attempts are exactly `0..UINT32_MAX` inclusive. If the candidate at `UINT32_MAX` rejects, MUST NOT increment/form another counter; fail with `RNG_REJECTION_EXHAUSTED`, `final_attempt=UINT32_MAX`, and discard all partial solve work. No reseed, algorithm switch, lane reuse, biased map, trial-count change, fallback, or successful partial result is permitted. Concurrent terminal failures report the lexicographically smallest `(simulation_id, draw_slot)`. The public error need only expose the code; case hash, seed, location, attempt, and replay tuple are trace/debug metadata and MUST be logged/sealed only under secrets-safe policy. Seed is provenance metadata, not asserted cryptographic secret material; raw seeds MUST NOT be required in user-visible errors or routine logs.

### 7. End-to-end verified replay vector

This vector was generated by a standalone Python `uint32`/`uint64` implementation after it passed all three Random123 KAT rows above; it is independently reproducible from sections 2--6.

Known case: hero `As Ah` IDs `{12,25}`, board `2s 3h Td` IDs `{0,14,34}`, `O=2`; sorted encoding is:

```text
canonical_case_bytes hex =
17706f6b65722d6b6e696768742d6e672f636173652f7631020c1903000e2202
canonical_case_hash      =
fb3c0fa3e41cdd7f89e45b458f17f14174d51f285723c5178c68bd2756fec3eb
seed = 0x0123456789abcdef
key_digest =
88e7ce310c8febab77f7362ee4951b9b74aa958f8005c084bad9f433ce6162ef
key = (0x31cee788, 0xabeb8f0c)
```

Here `B=3`, `D=6`, `R0=47`; destinations are turn, river, opponent 0 hole 0/1, opponent 1 hole 0/1. For `s=7`, these candidate vectors give counter, all output lanes, and selection result:

| `(s,d,a)` | Philox output `(lane0,lane1,lane2,lane3)` | `R` | `L` | accepted `i` | emitted card ID |
| --- | --- | ---: | ---: | ---: | ---: |
| `(7,0,0)` | `(794b88d3,48f5e880,4001a663,df59ffb3)` | 47 | 4294967254 | 6 | 7 |
| `(7,1,0)` | `(ac0109cb,43e5dfae,7e258ccc,d133cf5a)` | 46 | 4294967284 | 7 | 8 |
| `(7,2,0)` | `(7774fe53,a961a432,1baa8f19,7174c3d8)` | 45 | 4294967265 | 32 | 37 |
| `(7,3,0)` | `(cb6667ed,1f95dd3c,f304b96f,a118317d)` | 44 | 4294967292 | 37 | 42 |
| `(7,4,0)` | `(ecdfc51a,56b21c03,28c238ae,c3eea09e)` | 43 | 4294967280 | 12 | 15 |
| `(7,5,0)` | `(d61a77cd,40da40dd,4d54fd0c,002b0f79)` | 42 | 4294967292 | 9 | 10 |
| `(7,1,1)` | `(eb06ccc9,afa346cc,4a12a579,2095d67c)` | 46 | 4294967284 | 19 | would emit 20; nonzero-attempt counter check |

Each row's counter is explicitly `(00000007,00000000,d,a)`, with `d,a` rendered as zero-padded `uint32` hex. The first six dealt IDs are therefore `[7,8,37,42,15,10]`; the nonzero-attempt row is a counter-function vector, not an assertion that attempt 0 rejected.

For mapping boundary validation, implementations MUST test all active `R=34..50` and `R=32`: for every such `R`, `0 < L <= 2^32`, `L mod R = 0`, and for `R=32`, `L=2^32` (full-domain acceptance).

### 8. Replay and execution invariants

Equal identity/version, canonical bytes/hash, seed, and `N` MUST reproduce every deal and integer result count on CPU and qualified CUDA. Batching, launches, block/grid size, streams, retry placement external to this operation, and GPU count may alter location/time only, never identity. Invalid data/capability is pre-execution; requested CUDA unavailable MUST explicitly error rather than fall back. Runtime resource failure MUST NOT masquerade as completion.

## Consequences and deferred implementation work

This specifies exact replay now, while deliberately spending three Philox lanes per candidate for clarity. Deferred work is implementation/performance only: CPU/CUDA code, trace/result schemas, seed-bank fixtures, batching, memory layout, reductions, and statistical intervals. Such work MUST preserve every byte, hash, key, counter, mapping, and failure behavior above; it cannot change replay semantics.
