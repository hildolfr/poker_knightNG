"""Normative ADR 0003 deterministic-RNG primitive tests."""

from hashlib import sha256

import pytest

from poker_knight_ng.reference.rng import (
    UINT32_MAX,
    adr0003_candidate,
    adr0003_counter,
    derive_philox_key,
    philox4x32_10,
)


@pytest.mark.parametrize(
    ("counter", "key", "expected"),
    [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        ((UINT32_MAX,) * 4, (UINT32_MAX,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
        ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0), (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
    ],
)
def test_philox4x32_10_matches_random123_authoritative_kats(counter, key, expected):
    assert philox4x32_10(counter, key) == expected


def test_key_derivation_matches_adr_0003_known_case():
    case_hash = bytes.fromhex("fb3c0fa3e41cdd7f89e45b458f17f14174d51f285723c5178c68bd2756fec3eb")

    digest, key = derive_philox_key(0x0123456789ABCDEF, case_hash)

    assert digest.hex() == "88e7ce310c8febab77f7362ee4951b9b74aa958f8005c084bad9f433ce6162ef"
    assert key == (0x31CEE788, 0xABEB8F0C)


def test_candidate_uses_adr_counter_lanes_and_matches_known_nonzero_attempt_vector():
    case_hash = bytes.fromhex("fb3c0fa3e41cdd7f89e45b458f17f14174d51f285723c5178c68bd2756fec3eb")

    assert adr0003_counter(7, 1, 1) == (7, 0, 1, 1)
    assert adr0003_candidate(0x0123456789ABCDEF, case_hash, 7, 1, 1) == (
        0xEB06CCC9, 0xAFA346CC, 0x4A12A579, 0x2095D67C,
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: philox4x32_10((0, 0, 0), (0, 0)),
        lambda: philox4x32_10((0, 0, 0, 2**32), (0, 0)),
        lambda: philox4x32_10((0, 0, 0, 0), (0,)),
        lambda: derive_philox_key(-1, bytes(32)),
        lambda: derive_philox_key(0, bytes(31)),
        lambda: adr0003_counter(2**64, 0, 0),
        lambda: adr0003_counter(0, 17, 0),
        lambda: adr0003_counter(0, 0, 2**32),
    ],
)
def test_rng_primitives_reject_out_of_range_inputs_without_truncation(call):
    with pytest.raises(ValueError):
        call()


def test_key_derivation_preimage_is_exact_domain_label_seed_little_endian_and_hash():
    case_hash = bytes(range(32))
    digest, key = derive_philox_key(0x0123456789ABCDEF, case_hash)

    expected = sha256(
        bytes([26]) + b"poker-knight-ng/rng-key/v1" + bytes.fromhex("efcdab8967452301") + case_hash
    ).digest()
    assert digest == expected
    assert key == (int.from_bytes(expected[:4], "little"), int.from_bytes(expected[4:8], "little"))
