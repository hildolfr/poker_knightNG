"""Preregistered RNG seed-bank exact and statistical verification."""

import hashlib
import json
from decimal import Decimal, getcontext
from pathlib import Path

import pytest

import poker_knight_ng.reference.monte_carlo as monte_carlo
from poker_knight_ng.reference.dealer import canonical_case_bytes, canonical_case_hash


ROOT = Path(__file__).parents[2]
SEED_BANK = ROOT / "validation/holdem/v1/rng_seed_bank.json"
SEED_BANK_MANIFEST = ROOT / "validation/holdem/v1/manifests/rng_seed_bank.sha256"


def _load_bank():
    entries = {
        name: digest
        for digest, name in (
            line.rstrip("\n").split("  ", 1)
            for line in SEED_BANK_MANIFEST.read_text(encoding="ascii").splitlines()
        )
    }
    assert entries["validation/holdem/v1/rng_seed_bank.json"] == hashlib.sha256(SEED_BANK.read_bytes()).hexdigest()
    return json.loads(SEED_BANK.read_text(encoding="utf-8"))


def _case(vector):
    return dict(
        seed=int(vector["seed"], 16),
        hero_card_ids=tuple(vector["hero_card_ids"]),
        board_card_ids=tuple(vector["board_card_ids"]),
        opponent_count=vector["opponent_count"],
        requested_trials=int(vector["requested_trials"]),
    )


def _result_fields(result):
    return {"completed_trials": str(result.completed_trials), "unique_wins": str(result.unique_wins), "tie_by_other_winners": {str(i+1): str(v) for i,v in enumerate(result.tie_by_other_winners)}, "losses": str(result.losses), "equity_share_units": str(result.equity_share_units), "hero_category_counts": {name: str(result.hero_category_counts[i]) for i,name in enumerate(("high_card","one_pair","two_pair","three_of_a_kind","straight","flush","full_house","four_of_a_kind","straight_flush"))}, "rejection_count": str(result.rejection_count)}

def _wilson_interval(successes: int, total: int, z: Decimal) -> tuple[Decimal, Decimal]:
    getcontext().prec = 50
    n = Decimal(total)
    p = Decimal(successes) / n
    denominator = Decimal(1) + z * z / n
    center = (p + z * z / (Decimal(2) * n)) / denominator
    radius = z * ((p * (Decimal(1) - p) / n + z * z / (Decimal(4) * n * n)).sqrt()) / denominator
    return center - radius, center + radius


@pytest.mark.parametrize("vector", _load_bank()["exact_vectors"], ids=lambda vector: vector["id"])
def test_seed_bank_exact_hash_bound_replay(vector):
    case = _case(vector)
    expected_bytes = bytes.fromhex(vector["canonical_case_bytes_hex"])
    expected_hash = bytes.fromhex(vector["canonical_case_hash_hex"])

    assert canonical_case_bytes(case["hero_card_ids"], case["board_card_ids"], case["opponent_count"]) == expected_bytes
    assert canonical_case_hash(case["hero_card_ids"], case["board_card_ids"], case["opponent_count"]) == expected_hash
    result = monte_carlo.run_cpu_monte_carlo(**case, replay_case_hash=expected_hash)
    assert _result_fields(result) == vector["expected"]
    result.validate(case["opponent_count"], case["requested_trials"], len(case["board_card_ids"]))


@pytest.mark.parametrize("vector", _load_bank()["exact_vectors"], ids=lambda vector: vector["id"])
def test_seed_bank_hash_mutation_is_rejected_before_a_single_deal(monkeypatch, vector):
    def deal_must_not_run(**_kwargs):
        raise AssertionError("hash mismatch must fail before dealing")

    monkeypatch.setattr(monte_carlo, "deal_cpu", deal_must_not_run)
    bad_hash = bytearray.fromhex(vector["canonical_case_hash_hex"])
    bad_hash[-1] ^= 1
    with pytest.raises(ValueError, match="replay hash"):
        monte_carlo.run_cpu_monte_carlo(**_case(vector), replay_case_hash=bytes(bad_hash))


@pytest.mark.parametrize("vector", _load_bank()["statistical_vectors"], ids=lambda vector: vector["id"])
def test_seed_bank_preregistered_wilson_checks(vector):
    case = _case(vector)
    assert canonical_case_hash(case["hero_card_ids"], case["board_card_ids"], case["opponent_count"]).hex() == vector["canonical_case_hash_hex"]
    result = monte_carlo.run_cpu_monte_carlo(**case)
    assert _result_fields(result) == vector["expected"]

    z = Decimal(vector["confidence"]["z"])
    observed = {
        "unique_win": result.unique_wins,
        "tie": sum(result.tie_by_other_winners),
        "loss": result.losses,
    }
    for name, estimand in vector["estimands"].items():
        lower, upper = _wilson_interval(estimand["numerator"], estimand["denominator"], z)
        proportion = Decimal(observed[name]) / Decimal(result.completed_trials)
        assert lower <= proportion <= upper, (name, lower, proportion, upper)
