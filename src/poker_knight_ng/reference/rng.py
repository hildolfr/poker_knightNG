"""Exact pure-Python ADR 0003 Philox4x32-10 candidate primitives.

These deliberately expose only deterministic key/counter/candidate construction;
dealing and bounded selection are a later checkpoint.
"""

from hashlib import sha256
from typing import Final

UINT32_MAX: Final = (1 << 32) - 1
UINT64_MAX: Final = (1 << 64) - 1
_M0: Final = 0xD2511F53
_M1: Final = 0xCD9E8D57
_W0: Final = 0x9E3779B9
_W1: Final = 0xBB67AE85
_KEY_LABEL: Final = b"poker-knight-ng/rng-key/v1"


def _uint(value: object, bits: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << bits):
        raise ValueError(f"{name} must be an unsigned {bits}-bit integer")
    return value


def _lanes(value: object, count: int, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or len(value) != count:
        raise ValueError(f"{name} must be a {count}-lane tuple")
    return tuple(_uint(lane, 32, f"{name}[{index}]") for index, lane in enumerate(value))


def philox4x32_10(counter: object, key: object) -> tuple[int, int, int, int]:
    """Return the exact ten-round Philox4x32 output for four lanes and two key words."""
    c0, c1, c2, c3 = _lanes(counter, 4, "counter")
    k0, k1 = _lanes(key, 2, "key")
    for round_number in range(10):
        product0 = _M0 * c0
        product1 = _M1 * c2
        hi0, lo0 = product0 >> 32, product0 & UINT32_MAX
        hi1, lo1 = product1 >> 32, product1 & UINT32_MAX
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0)
        if round_number != 9:
            k0 = (k0 + _W0) & UINT32_MAX
            k1 = (k1 + _W1) & UINT32_MAX
    return c0, c1, c2, c3


def derive_philox_key(seed: object, canonical_case_hash: object) -> tuple[bytes, tuple[int, int]]:
    """Derive ADR 0003's digest and little-endian Philox key from seed and case hash."""
    seed_int = _uint(seed, 64, "seed")
    if not isinstance(canonical_case_hash, bytes) or len(canonical_case_hash) != 32:
        raise ValueError("canonical_case_hash must be exactly 32 bytes")
    digest = sha256(bytes([len(_KEY_LABEL)]) + _KEY_LABEL + seed_int.to_bytes(8, "little") + canonical_case_hash).digest()
    return digest, (int.from_bytes(digest[:4], "little"), int.from_bytes(digest[4:8], "little"))


def adr0003_counter(simulation_id: object, draw_slot: object, rejection_attempt: object) -> tuple[int, int, int, int]:
    """Construct ADR 0003's Philox lanes without narrowing caller-provided values."""
    simulation = _uint(simulation_id, 64, "simulation_id")
    slot = _uint(draw_slot, 32, "draw_slot")
    if slot > 16:
        raise ValueError("draw_slot must be at most 16")
    attempt = _uint(rejection_attempt, 32, "rejection_attempt")
    return simulation & UINT32_MAX, simulation >> 32, slot, attempt


def adr0003_candidate(
    seed: object,
    canonical_case_hash: object,
    simulation_id: object,
    draw_slot: object,
    rejection_attempt: object,
) -> tuple[int, int, int, int]:
    """Return the pure ADR 0003 Philox candidate block for one slot attempt."""
    _, key = derive_philox_key(seed, canonical_case_hash)
    return philox4x32_10(adr0003_counter(simulation_id, draw_slot, rejection_attempt), key)
