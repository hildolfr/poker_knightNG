"""ADR 0003 unbiased selection and deterministic CPU dealing tests."""

import pytest

import poker_knight_ng.reference.dealer as dealer
from poker_knight_ng.reference.dealer import (
    CANONICAL_CASE_HASH_MISMATCH,
    CanonicalCaseHashMismatch,
    RNG_REJECTION_EXHAUSTED,
    RngRejectionExhausted,
    canonical_case_bytes,
    canonical_case_hash,
    deal_cpu,
    select_unbiased,
)


CASE_HASH = bytes.fromhex("fb3c0fa3e41cdd7f89e45b458f17f14174d51f285723c5178c68bd2756fec3eb")
SEED = 0x0123456789ABCDEF


def test_cpu_deal_replays_adr_known_vector_no_replacement_and_slot_order():
    deal = deal_cpu(
        seed=SEED,
        canonical_case_hash=CASE_HASH,
        hero_card_ids=(12, 25),
        board_card_ids=(0, 14, 34),
        opponent_count=2,
        simulation_id=7,
    )

    assert deal.dealt_card_ids == (7, 8, 37, 42, 15, 10)
    assert deal.board_card_ids == (0, 14, 34, 7, 8)
    assert deal.opponent_hole_card_ids == ((37, 42), (15, 10))
    assert len(set(deal.known_card_ids + deal.dealt_card_ids)) == 11
    assert tuple(slot.rejection_count for slot in deal.trace.slots) == (0, 0, 0, 0, 0, 0)
    assert tuple(slot.final_attempt for slot in deal.trace.slots) == (0, 0, 0, 0, 0, 0)


def test_cpu_deal_rejects_valid_length_mismatched_replay_hash_before_philox(monkeypatch):
    def candidate_must_not_run(*_args):
        raise AssertionError("Philox candidate execution must not begin")

    monkeypatch.setattr(dealer, "adr0003_candidate", candidate_must_not_run)

    with pytest.raises(CanonicalCaseHashMismatch) as raised:
        deal_cpu(
            seed=SEED,
            canonical_case_hash=bytes(32),
            hero_card_ids=(12, 25),
            board_card_ids=(0, 14, 34),
            opponent_count=2,
            simulation_id=7,
        )

    assert raised.value.code == CANONICAL_CASE_HASH_MISMATCH
    assert str(raised.value) == CANONICAL_CASE_HASH_MISMATCH


def test_cpu_deal_semantic_card_order_preserves_canonical_bytes_hash_and_deal():
    reordered_bytes = canonical_case_bytes((25, 12), (34, 0, 14), 2)
    ordered_bytes = canonical_case_bytes((12, 25), (0, 14, 34), 2)
    assert ordered_bytes.hex() == "17706f6b65722d6b6e696768742d6e672f636173652f7631020c1903000e2202"
    assert reordered_bytes == ordered_bytes
    assert canonical_case_hash((25, 12), (34, 0, 14), 2) == CASE_HASH

    reordered = deal_cpu(
        seed=SEED,
        canonical_case_hash=CASE_HASH,
        hero_card_ids=(25, 12),
        board_card_ids=(34, 0, 14),
        opponent_count=2,
        simulation_id=7,
    )
    ordered = deal_cpu(
        seed=SEED,
        canonical_case_hash=CASE_HASH,
        hero_card_ids=(12, 25),
        board_card_ids=(0, 14, 34),
        opponent_count=2,
        simulation_id=7,
    )
    assert reordered == ordered


@pytest.mark.parametrize(
    ("hero_card_ids", "board_card_ids"),
    (
        ((True, 25), (0, 14, 34)),
        ((12, 25), (0, 14, 14)),
        ((12, 25), (0, 14, 52)),
        ((12, 25), (12, 14, 34)),
    ),
)
def test_cpu_deal_rejects_malformed_or_duplicate_ids_before_hashing_or_dealing(
    monkeypatch, hero_card_ids, board_card_ids
):
    def hashing_must_not_run(_payload):
        raise AssertionError("canonical hashing must not begin")

    def candidate_must_not_run(*_args):
        raise AssertionError("Philox candidate execution must not begin")

    monkeypatch.setattr(dealer, "sha256", hashing_must_not_run, raising=False)
    monkeypatch.setattr(dealer, "adr0003_candidate", candidate_must_not_run)

    with pytest.raises(ValueError):
        deal_cpu(
            seed=SEED,
            canonical_case_hash=CASE_HASH,
            hero_card_ids=hero_card_ids,
            board_card_ids=board_card_ids,
            opponent_count=2,
            simulation_id=7,
        )


def test_unbiased_selection_retries_same_slot_and_accounts_rejection_without_mapping_it():
    # For R=3, L=4294967295. UINT32_MAX rejects; the next word selects index 1.
    selection = select_unbiased(
        remaining_card_ids=(10, 11, 12),
        simulation_id=4,
        draw_slot=2,
        candidate_lane0=lambda attempt: (0xFFFFFFFF, 4)[attempt],
    )

    assert selection.card_id == 11
    assert selection.deck_after == (10, 12)
    assert selection.trace.rejection_count == 1
    assert selection.trace.final_attempt == 1
    assert selection.trace.accepted_word == 4
    assert selection.trace.active_range == 3
    assert selection.trace.acceptance_limit == 4294967295


def test_rejected_last_uint32_attempt_raises_stable_error_with_trace_metadata():
    with pytest.raises(RngRejectionExhausted) as raised:
        select_unbiased(
            remaining_card_ids=(10, 11, 12),
            simulation_id=4,
            draw_slot=2,
            candidate_lane0=lambda attempt: 0xFFFFFFFF,
            initial_attempt=0xFFFFFFFF,
        )

    error = raised.value
    assert error.code == RNG_REJECTION_EXHAUSTED
    assert error.simulation_id == 4
    assert error.draw_slot == 2
    assert error.final_attempt == 0xFFFFFFFF


@pytest.mark.parametrize("active_range", (32, *range(34, 51)))
def test_selection_limit_is_exact_multiple_including_full_domain_boundary(active_range):
    selection = select_unbiased(
        remaining_card_ids=tuple(range(active_range)),
        simulation_id=0,
        draw_slot=0,
        candidate_lane0=lambda attempt: 0,
    )
    assert 0 < selection.trace.acceptance_limit <= 2**32
    assert selection.trace.acceptance_limit % active_range == 0
    if active_range == 32:
        assert selection.trace.acceptance_limit == 2**32
