"""Host-only conformance tests for the deterministic standalone one-trial simulator."""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path

import pytest

from poker_knight_ng.reference.dealer import canonical_case_hash
from poker_knight_ng.reference.monte_carlo import _run_trial

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ROOT / "src" / "poker_knight_ng" / "cuda-sources"
SEED_BANK = ROOT / "validation" / "holdem" / "v1" / "rng_seed_bank.json"

HARNESS = r'''
#include <cstdint>
#include <iostream>
#include <type_traits>
#include "simulate.cuh"
using namespace poker_knight_ng::cuda;
int main() {
  static_assert(std::is_standard_layout<TrialResult>::value, "stable result layout");
  static_assert(std::is_trivially_copyable<TrialResult>::value, "reduction-safe result");
  unsigned mode;
  while (std::cin >> mode) {
    if (mode == 0) {
      std::uint32_t k0, k1, opponents, known_count; std::uint64_t sim;
      std::uint8_t hero[2]{}, board[5]{};
      std::cin >> k0 >> k1 >> sim >> opponents >> known_count;
      for (unsigned i=0; i<2; ++i) { unsigned x; std::cin >> x; hero[i]=static_cast<std::uint8_t>(x); }
      for (unsigned i=0; i+2<known_count; ++i) { unsigned x; std::cin >> x; board[i]=static_cast<std::uint8_t>(x); }
      TrialResult r{}; const TrialStatus s=simulate_one_trial(hero, board, known_count-2, opponents, k0, k1, sim, &r);
      std::cout << static_cast<unsigned>(s) << ' ' << static_cast<unsigned>(r.status) << ' '
                << static_cast<unsigned>(r.outcome) << ' ' << static_cast<unsigned>(r.tie_other_winners) << ' '
                << r.equity_share_units << ' ' << static_cast<unsigned>(r.hero_category) << ' ' << r.rejection_count << '\n';
    } else if (mode == 1) {
      std::cout << sizeof(TrialResult) << ' ' << sizeof(decltype(TrialResult::equity_share_units)) << ' '
                << sizeof(decltype(TrialResult::rejection_count)) << '\n';
    } else {
      unsigned dealer_status; std::cin >> dealer_status;
      std::cout << static_cast<unsigned>(trial_status_from_dealer(static_cast<DealerStatus>(dealer_status))) << '\n';
    }
  }
}
'''


@pytest.fixture(scope="module")
def simulator_harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("cuda-simulator-host")
    source, binary = directory / "simulator_harness.cpp", directory / "simulator_harness"
    source.write_text(HARNESS)
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(INCLUDE), str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


def _key(seed: int, hero: tuple[int, int], board: tuple[int, ...], opponents: int) -> tuple[int, int]:
    digest = hashlib.sha256(bytes([26]) + b"poker-knight-ng/rng-key/v1" + seed.to_bytes(8, "little") + canonical_case_hash(hero, board, opponents)).digest()
    return int.from_bytes(digest[:4], "little"), int.from_bytes(digest[4:8], "little")


def _line(seed: int, hero: tuple[int, int], board: tuple[int, ...], opponents: int, simulation_id: int) -> str:
    k0, k1 = _key(seed, hero, board, opponents)
    return "0 %d %d %d %d %d %s" % (k0, k1, simulation_id, opponents, 2 + len(board), " ".join(map(str, hero + board)))


def _run(binary: Path, lines: list[str]) -> list[tuple[int, ...]]:
    result = subprocess.run([str(binary)], input="\n".join(lines) + "\n", check=True, capture_output=True, text=True)
    return [tuple(map(int, row.split())) for row in result.stdout.splitlines()]


def _expected(seed: int, hero: tuple[int, int], board: tuple[int, ...], opponents: int, simulation_id: int) -> tuple[int, ...]:
    result = _run_trial(seed=seed, hero_card_ids=hero, board_card_ids=board, opponent_count=opponents, simulation_id=simulation_id)
    ties = next((index + 1 for index, count in enumerate(result.tie_by_other_winners) if count), 0)
    outcome = 0 if result.unique_wins else (1 if ties else 2)
    category = next(index for index, count in enumerate(result.hero_category_counts) if count)
    return (0, 0, outcome, ties, result.equity_share_units, category, result.rejection_count)


def test_trial_direct_known_vectors_and_result_abi(simulator_harness: Path) -> None:
    cases = [
        (0x0123456789ABCDEF, (12, 25), (0, 14, 34), 2, 7),
        (1, (12, 37), (0, 18, 33, 47, 48), 1, (1 << 32) + 9),
        (0x9988776655443322, (3, 41), (), 6, 0),
    ]
    assert _run(simulator_harness, [_line(*case) for case in cases]) == [_expected(*case) for case in cases]
    size, units_width, rejections_width = _run(simulator_harness, ["1"])[0]
    assert size >= 16
    assert units_width == 2
    assert rejections_width == 8


def test_trial_differential_5000_full_topology_and_scheduling(simulator_harness: Path) -> None:
    rng = random.Random(0x51A0)
    pairs = [(b, o) for b in (0, 3, 4, 5) for o in range(1, 7)]
    counts = {pair: 0 for pair in pairs}
    cases = []
    for number in range(5_000):
        board_count, opponents = pairs[number % len(pairs)]
        counts[(board_count, opponents)] += 1
        cards = rng.sample(range(52), 2 + board_count)
        hero, board = tuple(sorted(cards[:2])), tuple(sorted(cards[2:]))
        seed = rng.getrandbits(64)
        simulation_id = (1 << 32) + number if number % 3 == 0 else rng.getrandbits(64)
        cases.append((seed, hero, board, opponents, simulation_id))
    assert min(counts.values()) == 208 and max(counts.values()) == 209
    rows = _run(simulator_harness, [_line(*case) for case in cases])
    assert rows == [_expected(*case) for case in cases]
    selected = [7, 2, 4999, 17, 700]
    assert _run(simulator_harness, [_line(*cases[index]) for index in reversed(selected)]) == [rows[index] for index in reversed(selected)]
    assert _run(simulator_harness, [_line(*case) for case in cases[:23]]) == rows[:23]


def test_trial_seed_bank_aggregates_match_published_vectors(simulator_harness: Path) -> None:
    vectors = json.loads(SEED_BANK.read_text())["exact_vectors"]
    assert len(vectors) == 3
    for vector in vectors:
        seed = int(vector["seed"], 16)
        hero, board = tuple(vector["hero_card_ids"]), tuple(vector["board_card_ids"])
        opponents, requested = vector["opponent_count"], int(vector["requested_trials"])
        rows = _run(simulator_harness, [_line(seed, hero, board, opponents, sim) for sim in range(requested)])
        expected = vector["expected"]
        assert sum(row[2] == 0 for row in rows) == int(expected["unique_wins"])
        assert sum(row[2] == 2 for row in rows) == int(expected["losses"])
        assert sum(row[4] for row in rows) == int(expected["equity_share_units"])
        assert sum(row[6] for row in rows) == int(expected["rejection_count"])
        for other in range(1, 7):
            assert sum(row[2] == 1 and row[3] == other for row in rows) == int(expected["tie_by_other_winners"][str(other)])
        names = ("high_card", "one_pair", "two_pair", "three_of_a_kind", "straight", "flush", "full_house", "four_of_a_kind", "straight_flush")
        for category, name in enumerate(names):
            assert sum(row[5] == category for row in rows) == int(expected["hero_category_counts"][name])


def test_named_outcome_branches_max_tie_and_rare_category_boundaries(simulator_harness: Path) -> None:
    # Search is only to select stable ordinary unique-win/loss/tie-one vectors;
    # the checked C++ records are then compared to the authoritative CPU leaf.
    ordinary = (0x0123456789ABCDEF, (12, 25), (0, 14, 34), 2)
    found: dict[int, tuple[int, tuple[int, int], tuple[int, ...], int, int]] = {}
    for simulation_id in range(20_000):
        expected = _expected(*ordinary, simulation_id)
        if expected[2] not in found:
            found[expected[2]] = (*ordinary, simulation_id)
        if expected[2] == 1 and expected[3] == 1:
            found[3] = (*ordinary, simulation_id)
        if {0, 2, 3} <= set(found):
            break
    assert {0, 2, 3} <= set(found)
    rows = _run(simulator_harness, [_line(*found[key]) for key in (0, 2, 3)])
    assert rows == [_expected(*found[key]) for key in (0, 2, 3)]
    assert rows[0][4] == 420 and rows[1][4] == 0 and rows[2][3:5] == (1, 210)

    # A five-card royal-flush board is necessarily shared by all six opponents.
    shared = (7, (13, 14), (8, 9, 10, 11, 12), 6, (1 << 32) + 11)
    max_tie = _run(simulator_harness, [_line(*shared)])[0]
    assert max_tie == _expected(*shared)
    assert max_tie[2:6] == (1, 6, 60, 8)

    # Complete-board vectors make rare hero categories deterministic, independently
    # of opponent holes: quads and straight flush are both exercised through the leaf.
    rare_cases = [
        (11, (12, 25), (0, 1, 13, 26, 39), 1, 4, 7),
        (11, (13, 14), (8, 9, 10, 11, 12), 1, 5, 8),
    ]
    rare_rows = _run(simulator_harness, [_line(*case[:-1]) for case in rare_cases])
    assert rare_rows == [
        _expected(*case[:-1]) for case in rare_cases
    ]
    assert [row[5] for row in rare_rows] == [7, 8]


def test_dealer_status_mapping_is_total_for_declared_statuses(simulator_harness: Path) -> None:
    assert _run(simulator_harness, ["2 0", "2 1", "2 2"]) == [(0,), (1,), (2,)]


def test_invalid_topology_has_invalid_status(simulator_harness: Path) -> None:
    # Invalid board count is rejected at the simulator boundary, without relying on host validation.
    row = _run(simulator_harness, ["0 1 2 3 1 4 2 3 4"])[0]
    assert row[:2] == (2, 2)
