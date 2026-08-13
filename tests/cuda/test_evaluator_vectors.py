"""Host-compiled conformance checks for the standalone CUDA evaluator headers.

These tests deliberately compile no CUDA code and never initialize a GPU.  The headers
are device-callable C++17 but are also valid ordinary C++17, so canonical evaluator
semantics can be checked without disturbing an active GPU workload.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import pytest

from poker_knight_ng.reference.evaluator import best_five

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ROOT / "src" / "poker_knight_ng" / "cuda-sources"
VECTORS = ROOT / "validation" / "holdem" / "v1" / "canonical_rank_vectors.jsonl"

HARNESS = r'''
#include <array>
#include <cstdint>
#include <iostream>
#include "cards.cuh"
#include "evaluator.cuh"
int main() {
  std::array<std::uint8_t, 7> cards{};
  int count;
  while (std::cin >> count) {
    for (int i = 0; i < count; ++i) { unsigned int card; std::cin >> card; cards[i] = static_cast<std::uint8_t>(card); }
    const auto score = poker_knight_ng::cuda::best_five_score(cards.data(), count);
    std::cout << static_cast<int>(score.category) << ' ' << static_cast<int>(score.k1) << ' ' << static_cast<int>(score.k2) << ' ' << static_cast<int>(score.k3) << ' ' << static_cast<int>(score.k4) << ' ' << static_cast<int>(score.k5) << '\n';
  }
}
'''


def _card_id(token: str) -> int:
    ranks = "23456789TJQKA"
    suits = "shdc"
    return suits.index(token[1]) * 13 + ranks.index(token[0])


@pytest.fixture(scope="module")
def evaluator_harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("cuda-evaluator-host")
    source = directory / "evaluator_harness.cpp"
    binary = directory / "evaluator_harness"
    source.write_text(HARNESS)
    subprocess.run(
        ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(INCLUDE), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def _evaluate(binary: Path, cards: list[str]) -> tuple[int, int, int, int, int, int]:
    stdin = f"{len(cards)} {' '.join(str(_card_id(card)) for card in cards)}\n"
    completed = subprocess.run([str(binary)], input=stdin, check=True, capture_output=True, text=True)
    return tuple(map(int, completed.stdout.split()))  # type: ignore[return-value]


def _evaluate_many(binary: Path, hands: list[tuple[str, ...]]) -> list[tuple[int, int, int, int, int, int]]:
    stdin = "".join(f"{len(cards)} {' '.join(str(_card_id(card)) for card in cards)}\n" for cards in hands)
    completed = subprocess.run([str(binary)], input=stdin, check=True, capture_output=True, text=True)
    return [tuple(map(int, line.split())) for line in completed.stdout.splitlines()]  # type: ignore[return-value]


def test_standalone_cuda_evaluator_matches_reference_for_balanced_five_six_seven_card_corpus(
    evaluator_harness: Path,
) -> None:
    deck = tuple(f"{rank}{suit}" for suit in "shdc" for rank in "23456789TJQKA")
    rng = random.Random(0xC0DA15)
    hands = [
        tuple(rng.sample(deck, card_count))
        for card_count in (5, 6, 7)
        for _ in range(5_000)
    ]

    assert len(hands) == 15_000
    assert {card_count: sum(len(hand) == card_count for hand in hands) for card_count in (5, 6, 7)} == {
        5: 5_000,
        6: 5_000,
        7: 5_000,
    }
    actual = _evaluate_many(evaluator_harness, hands)
    expected = [best_five(hand).score.key for hand in hands]

    assert actual == expected


def test_canonical_rank_vectors_match_standalone_cuda_evaluator(evaluator_harness: Path) -> None:
    for line in VECTORS.read_text().splitlines():
        vector = json.loads(line)
        assert _evaluate(evaluator_harness, vector["cards"]) == tuple(vector["expected_key"]), vector["id"]


def test_quads_kicker_comes_from_a_card_present_in_the_hand(evaluator_harness: Path) -> None:
    assert _evaluate(evaluator_harness, ["Ks", "Kh", "Kd", "Kc", "Js"]) == (7, 13, 11, 0, 0, 0)


def test_wheel_is_five_high_and_loses_to_six_high(evaluator_harness: Path) -> None:
    wheel = _evaluate(evaluator_harness, ["As", "2h", "3d", "4c", "5s"])
    six_high = _evaluate(evaluator_harness, ["2s", "3h", "4d", "5c", "6s"])
    wheel_flush = _evaluate(evaluator_harness, ["As", "2s", "3s", "4s", "5s", "Kd", "Qc"])
    assert wheel == (4, 5, 0, 0, 0, 0)
    assert six_high == (4, 6, 0, 0, 0, 0)
    assert wheel_flush == (8, 5, 0, 0, 0, 0)
    assert wheel < six_high


def test_internal_card_ids_follow_adr_0002() -> None:
    assert _card_id("2s") == 0
    assert _card_id("As") == 12
    assert _card_id("2h") == 13
    assert _card_id("Th") == 21
    assert _card_id("2c") == 39
