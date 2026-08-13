#!/usr/bin/env python3
"""Build or strictly verify the hash-bound Phase 2 Hold'em corpus.

``--verify`` is fast: it authenticates source bytes and validates the complete
canonical and semantic shape of the checked-in corpus. ``--release`` is the
expensive gate: the transparent reference and pinned Treys 0.1.8 must both
match the frozen checkpoint values before any files are published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from poker_knight_ng.reference.cards import CARD_DECK, parse_cards
from poker_knight_ng.reference.enumerate import (
    enumerate_fixed_holes,
    enumerate_unknown_opponent,
    evaluate_terminal,
)
from poker_knight_ng.reference.evaluator import CATEGORY_NAMES, best_five, score_five

ROOT = Path(__file__).parents[1]
UINT64_MAX = 18_446_744_073_709_551_615
COUNTER_RE = re.compile(r"0|[1-9][0-9]*", re.ASCII)

CORPUS_NAMES = (
    "canonical_rank_vectors.jsonl",
    "exact_holdem_cases.jsonl",
    "tie_and_split_cases.jsonl",
)
AUTHORITY_NAMES = (
    "tools/generate_oracle_fixtures.py",
    "docs/adr/0001-equity-v1-scope.md",
    "docs/adr/0002-card-rank-and-tie-semantics.md",
    "docs/adr/0003-deterministic-rng-and-deal-order.md",
    "contracts/v1/equity-request.schema.json",
    "contracts/v1/equity-result.schema.json",
    "contracts/v1/problem.schema.json",
    "src/poker_knight_ng/reference/cards.py",
    "src/poker_knight_ng/reference/evaluator.py",
    "src/poker_knight_ng/reference/enumerate.py",
)
MANIFEST_NAMES = CORPUS_NAMES + AUTHORITY_NAMES

RANK_DERIVATION = "transparent-reference+treys-0.1.8"
DIRECT_DERIVATION = "direct-arithmetic+transparent-reference+treys-0.1.8"
RANK_KEYS = (
    "format_version",
    "contract_version",
    "derivation",
    "id",
    "cards",
    "expected_key",
)
EQUITY_KEYS = (
    "format_version",
    "contract_version",
    "derivation",
    "id",
    "operation",
    "hero_cards",
    "board_cards",
    "opponent_holes",
    "expected",
)
EXPECTED_KEYS = (
    "completed_trials",
    "unique_wins",
    "tie_by_other_winners",
    "losses",
    "equity_share_units",
    "hero_category_counts",
)
TIE_KEYS = tuple(str(value) for value in range(1, 7))

# (id, cards, exact transparent score key)
RANK_ROWS = (
    ("high-card", ("As", "Kd", "Qh", "Jc", "9s"), (0, 14, 13, 12, 11, 9)),
    ("one-pair", ("As", "Ah", "Kc", "Qd", "Jh"), (1, 14, 13, 12, 11, 0)),
    ("two-pair-kicker", ("As", "Ad", "Kc", "Kd", "Qh"), (2, 14, 13, 12, 0, 0)),
    ("trips", ("As", "Ah", "Ad", "Kc", "Qd"), (3, 14, 13, 12, 0, 0)),
    ("wheel", ("As", "2h", "3d", "4c", "5s"), (4, 5, 0, 0, 0, 0)),
    ("six-high", ("2s", "3h", "4d", "5c", "6s"), (4, 6, 0, 0, 0, 0)),
    ("flush-kicker", ("Ah", "Kh", "9h", "5h", "2h"), (5, 14, 13, 9, 5, 2)),
    ("double-trips", ("As", "Ah", "Ad", "Ks", "Kh", "Kd", "Qc"), (6, 14, 13, 0, 0, 0)),
    ("quads-kicker", ("As", "Ah", "Ad", "Ac", "Kd"), (7, 14, 13, 0, 0, 0)),
    (
        "royal-display-straight-flush",
        ("As", "Ks", "Qs", "Js", "Ts", "2c", "3d"),
        (8, 14, 0, 0, 0, 0),
    ),
)

# (id, operation, hero, board, opponent holes, derivation)
EXACT_CASES = (
    ("unknown-river", "unknown_opponent", ("As", "Kd"), ("2s", "7h", "9d", "Tc", "Jc"), None, RANK_DERIVATION),
    ("unknown-turn", "unknown_opponent", ("As", "Kd"), ("2s", "7h", "9d", "Tc"), None, RANK_DERIVATION),
    ("unknown-flop", "unknown_opponent", ("As", "Kd"), ("2s", "7h", "9d"), None, RANK_DERIVATION),
    ("fixed-holes-turn", "fixed_holes", ("As", "Kd"), ("2s", "7h", "9d", "Tc"), (("Qh", "Jd"),), RANK_DERIVATION),
    ("fixed-holes-preflop-aa-vs-kk", "fixed_holes", ("As", "Ah"), (), (("Ks", "Kh"),), RANK_DERIVATION),
    ("terminal-loss", "terminal", ("Ks", "Kd"), ("As", "Ah", "Ad", "Kc", "Qd"), (("Ac", "Qc"),), DIRECT_DERIVATION),
)

TIE_POOL = (
    ("2c", "3d"),
    ("4h", "5c"),
    ("6d", "7h"),
    ("8c", "9d"),
    ("Th", "Jc"),
    ("Qd", "Kh"),
)

# Values are (trials, wins, six tie bins, losses, units, nine categories).
FROZEN_RESULTS = {
    "unknown-river": (990, 268, (9, 0, 0, 0, 0, 0), 713, 114450, (990, 0, 0, 0, 0, 0, 0, 0, 0)),
    "unknown-turn": (45540, 19317, (396, 0, 0, 0, 0, 0), 25827, 8196300, (27720, 17820, 0, 0, 0, 0, 0, 0, 0)),
    "unknown-flop": (1070190, 554910, (11522, 0, 0, 0, 0, 0), 503758, 235481820, (443520, 522720, 89100, 14850, 0, 0, 0, 0, 0)),
    "fixed-holes-turn": (44, 31, (0, 0, 0, 0, 0, 0), 13, 13020, (26, 18, 0, 0, 0, 0, 0, 0, 0)),
    "fixed-holes-preflop-aa-vs-kk": (1712304, 1410336, (9308, 0, 0, 0, 0, 0), 292660, 594295800, (0, 606936, 680288, 208130, 20336, 27798, 153032, 15664, 120)),
    "terminal-loss": (1, 0, (0, 0, 0, 0, 0, 0), 1, 0, (0, 0, 0, 0, 0, 0, 1, 0, 0)),
    "shared-board-1-other-winners": (1, 0, (1, 0, 0, 0, 0, 0), 0, 210, (0, 0, 0, 0, 0, 0, 0, 0, 1)),
    "shared-board-2-other-winners": (1, 0, (0, 1, 0, 0, 0, 0), 0, 140, (0, 0, 0, 0, 0, 0, 0, 0, 1)),
    "shared-board-3-other-winners": (1, 0, (0, 0, 1, 0, 0, 0), 0, 105, (0, 0, 0, 0, 0, 0, 0, 0, 1)),
    "shared-board-4-other-winners": (1, 0, (0, 0, 0, 1, 0, 0), 0, 84, (0, 0, 0, 0, 0, 0, 0, 0, 1)),
    "shared-board-5-other-winners": (1, 0, (0, 0, 0, 0, 1, 0), 0, 70, (0, 0, 0, 0, 0, 0, 0, 0, 1)),
    "shared-board-6-other-winners": (1, 0, (0, 0, 0, 0, 0, 1), 0, 60, (0, 0, 0, 0, 0, 0, 0, 0, 1)),
}


class QualificationError(RuntimeError):
    """Stable release/verification failure."""


def tie_cases() -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            f"shared-board-{count}-other-winners",
            "terminal",
            ("Ac", "Ad"),
            ("As", "Ks", "Qs", "Js", "Ts"),
            TIE_POOL[:count],
            DIRECT_DERIVATION,
        )
        for count in range(1, 7)
    )


def compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True) + "\n"


def _expected_from_tuple(values: tuple[Any, ...]) -> dict[str, Any]:
    trials, wins, ties, losses, units, categories = values
    return {
        "completed_trials": str(trials),
        "unique_wins": str(wins),
        "tie_by_other_winners": {
            str(index): str(value) for index, value in enumerate(ties, 1)
        },
        "losses": str(losses),
        "equity_share_units": str(units),
        "hero_category_counts": {
            name: str(value) for name, value in zip(CATEGORY_NAMES, categories)
        },
    }


def named(result: object) -> dict[str, Any]:
    return {
        "completed_trials": str(result.completed_trials),
        "unique_wins": str(result.unique_wins),
        "tie_by_other_winners": {
            str(index): str(value)
            for index, value in enumerate(result.tie_by_other_winners, 1)
        },
        "losses": str(result.losses),
        "equity_share_units": str(result.equity_share_units),
        "hero_category_counts": {
            name: str(value)
            for name, value in zip(CATEGORY_NAMES, result.hero_category_counts)
        },
    }


def transparent(
    operation: str,
    hero: Iterable[str],
    board: Iterable[str],
    opponents: Iterable[Iterable[str]] | None,
):
    if operation == "unknown_opponent":
        return enumerate_unknown_opponent(hero, board)
    if operation == "fixed_holes":
        return enumerate_fixed_holes(hero, board, opponents)
    if operation == "terminal":
        return evaluate_terminal(hero, board, opponents)
    raise QualificationError("unsupported fixture operation")


def treys_equity(
    operation: str,
    hero_tokens: Iterable[str],
    board_tokens: Iterable[str],
    opponent_tokens: Iterable[Iterable[str]] | None,
) -> dict[str, Any]:
    import treys

    evaluator = treys.Evaluator()
    convert = treys.Card.new
    hero = list(map(convert, hero_tokens))
    board = list(map(convert, board_tokens))
    opponents = (
        []
        if opponent_tokens is None
        else [list(map(convert, hand)) for hand in opponent_tokens]
    )
    used = {*hero, *board, *(card for hand in opponents for card in hand)}
    remaining = [
        convert(card.token) for card in CARD_DECK if convert(card.token) not in used
    ]

    if operation == "unknown_opponent":
        rows = (
            (hero, board + list(runout), [list(hand)])
            for runout in combinations(remaining, 5 - len(board))
            for hand in combinations(
                [card for card in remaining if card not in runout], 2
            )
        )
    elif operation == "fixed_holes":
        rows = (
            (hero, board + list(runout), opponents)
            for runout in combinations(remaining, 5 - len(board))
        )
    elif operation == "terminal":
        rows = ((hero, board, opponents),)
    else:
        raise QualificationError("unsupported Treys operation")

    wins = losses = units = total = 0
    ties = [0] * 6
    categories = [0] * 9
    for player, final_board, other_players in rows:
        hero_score = evaluator.evaluate(player, final_board)
        rank_class = evaluator.get_rank_class(hero_score)
        categories[8 if rank_class in (0, 1) else 9 - rank_class] += 1
        other_scores = [evaluator.evaluate(hand, final_board) for hand in other_players]
        best_other = min(other_scores)
        total += 1
        if hero_score < best_other:
            wins += 1
            units += 420
        elif hero_score == best_other:
            other_winners = sum(score == hero_score for score in other_scores)
            if not 1 <= other_winners <= 6:
                raise QualificationError("invalid Treys tie multiplicity")
            ties[other_winners - 1] += 1
            units += 420 // (other_winners + 1)
        else:
            losses += 1

    return _expected_from_tuple(
        (total, wins, tuple(ties), losses, units, tuple(categories))
    )


def _rank_row(identifier: str, cards: tuple[str, ...], key: tuple[int, ...]):
    return {
        "format_version": "1",
        "contract_version": "v1",
        "derivation": RANK_DERIVATION,
        "id": identifier,
        "cards": list(cards),
        "expected_key": list(key),
    }


def _equity_row(case: tuple[Any, ...]) -> dict[str, Any]:
    identifier, operation, hero, board, opponents, derivation = case
    return {
        "format_version": "1",
        "contract_version": "v1",
        "derivation": derivation,
        "id": identifier,
        "operation": operation,
        "hero_cards": list(hero),
        "board_cards": list(board),
        "opponent_holes": (
            None if opponents is None else [list(hand) for hand in opponents]
        ),
        "expected": _expected_from_tuple(FROZEN_RESULTS[identifier]),
    }


def _treys_rank_category(cards: tuple[str, ...]) -> int:
    import treys

    evaluator = treys.Evaluator()
    converted = [treys.Card.new(card) for card in cards]
    score = (
        evaluator.evaluate([], converted)
        if len(converted) == 5
        else evaluator.evaluate(converted[:2], converted[2:])
    )
    rank_class = evaluator.get_rank_class(score)
    return 8 if rank_class in (0, 1) else 9 - rank_class


def build(root: Path, case_filter: set[str] | None = None) -> dict[str, bytes]:
    rank_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []

    for identifier, cards, frozen_key in RANK_ROWS:
        if case_filter is not None and identifier not in case_filter:
            continue
        transparent_key = (
            score_five(cards).key if len(cards) == 5 else best_five(cards).score.key
        )
        if transparent_key != frozen_key:
            raise QualificationError("transparent rank disagreement")
        if _treys_rank_category(cards) != frozen_key[0]:
            raise QualificationError("Treys rank disagreement")
        rank_rows.append(_rank_row(identifier, cards, frozen_key))

    for cases, destination in ((EXACT_CASES, exact_rows), (tie_cases(), tie_rows)):
        for case in cases:
            identifier, operation, hero, board, opponents, _derivation = case
            if case_filter is not None and identifier not in case_filter:
                continue
            frozen = _expected_from_tuple(FROZEN_RESULTS[identifier])
            if named(transparent(operation, hero, board, opponents)) != frozen:
                raise QualificationError("transparent equity disagreement")
            if treys_equity(operation, hero, board, opponents) != frozen:
                raise QualificationError("Treys equity disagreement")
            destination.append(_equity_row(case))

    return {
        "canonical_rank_vectors.jsonl": "".join(map(compact, rank_rows)).encode(),
        "exact_holdem_cases.jsonl": "".join(map(compact, exact_rows)).encode(),
        "tie_and_split_cases.jsonl": "".join(map(compact, tie_rows)).encode(),
    }


def manifest(root: Path, outputs: dict[str, bytes]) -> bytes:
    lines = [
        f"{hashlib.sha256(outputs[name]).hexdigest()}  {name}\n"
        for name in CORPUS_NAMES
    ]
    lines.extend(
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
        for name in AUTHORITY_NAMES
    )
    return "".join(lines).encode("ascii")


def _stage_bytes(directory: Path, data: bytes) -> Path:
    fd, raw_path = tempfile.mkstemp(
        prefix=".oracle-fixture-", suffix=".tmp", dir=directory
    )
    path = Path(raw_path)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def atomic_write(destination: Path, outputs: dict[str, bytes]) -> None:
    """Publish with rollback for caught failures; not cross-file crash atomic."""
    staged: dict[Path, Path] = {}
    originals: dict[Path, bytes | None] = {}
    replaced: list[Path] = []
    try:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "manifests").mkdir(exist_ok=True)
        paths = {
            (
                destination / name
                if name in CORPUS_NAMES
                else destination / "manifests" / name
            ): data
            for name, data in outputs.items()
        }
        for path, data in paths.items():
            originals[path] = path.read_bytes() if path.exists() else None
            staged[path] = _stage_bytes(path.parent, data)
        for path in paths:
            os.replace(staged[path], path)
            replaced.append(path)
            del staged[path]
    except Exception:
        for path in reversed(replaced):
            original = originals[path]
            try:
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    os.replace(_stage_bytes(path.parent, original), path)
            except Exception:
                pass
        raise QualificationError("fixture publication failed") from None
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def release(
    root: Path = ROOT,
    destination: Path | None = None,
    case_filter: set[str] | None = None,
) -> dict[str, bytes]:
    output = Path(destination or root / "validation/holdem/v1")
    corpus = build(root, case_filter)
    corpus["sha256sums.txt"] = manifest(root, corpus)
    atomic_write(output, corpus)
    return corpus


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(_value: str):
    raise QualificationError("invalid JSON constant")


def _parse_jsonl(data: bytes, expected_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    if not data or not data.endswith(b"\n") or b"\r" in data or data.startswith(b"\xef\xbb\xbf"):
        raise QualificationError("noncanonical JSONL framing")
    rows: list[dict[str, Any]] = []
    for encoded_line in data.splitlines(keepends=True):
        if encoded_line == b"\n" or not encoded_line.endswith(b"\n"):
            raise QualificationError("noncanonical JSONL framing")
        try:
            text = encoded_line[:-1].decode("utf-8", errors="strict")
            row = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except QualificationError:
            raise
        except Exception:
            raise QualificationError("invalid JSONL") from None
        if type(row) is not dict:
            raise QualificationError("invalid JSONL row")
        if compact(row).encode("ascii") != encoded_line:
            raise QualificationError("noncanonical JSONL serialization")
        rows.append(row)
    identifiers = tuple(
        row.get("id") if type(row.get("id")) is str else None for row in rows
    )
    if identifiers != expected_ids or len(set(identifiers)) != len(identifiers):
        raise QualificationError("invalid JSONL row identities")
    return rows


def _parse_counter(value: object) -> int:
    if type(value) is not str or COUNTER_RE.fullmatch(value) is None:
        raise QualificationError("invalid counter")
    parsed = int(value)
    if parsed > UINT64_MAX:
        raise QualificationError("counter out of range")
    return parsed


def _validate_cards(tokens: object, expected: tuple[str, ...]) -> tuple[str, ...]:
    if type(tokens) is not list or any(type(token) is not str for token in tokens):
        raise QualificationError("invalid card list")
    actual = tuple(tokens)
    if actual != expected:
        raise QualificationError("fixture cards changed")
    try:
        parsed = parse_cards(actual)
    except Exception:
        raise QualificationError("invalid fixture cards") from None
    if len(parsed) != len(actual):
        raise QualificationError("invalid fixture cards")
    return actual


def _validate_expected(
    expected: object,
    frozen: dict[str, Any],
    opponent_count: int,
) -> None:
    if type(expected) is not dict or tuple(expected) != EXPECTED_KEYS:
        raise QualificationError("invalid expected object")
    if expected != frozen:
        raise QualificationError("qualified counters changed")
    ties_object = expected["tie_by_other_winners"]
    categories_object = expected["hero_category_counts"]
    if type(ties_object) is not dict or tuple(ties_object) != TIE_KEYS:
        raise QualificationError("invalid tie counters")
    if type(categories_object) is not dict or tuple(categories_object) != tuple(CATEGORY_NAMES):
        raise QualificationError("invalid category counters")

    trials = _parse_counter(expected["completed_trials"])
    wins = _parse_counter(expected["unique_wins"])
    losses = _parse_counter(expected["losses"])
    units = _parse_counter(expected["equity_share_units"])
    ties = {int(key): _parse_counter(value) for key, value in ties_object.items()}
    categories = [_parse_counter(value) for value in categories_object.values()]
    if trials == 0 or wins + losses + sum(ties.values()) != trials:
        raise QualificationError("counter conservation failure")
    if sum(categories) != trials:
        raise QualificationError("category conservation failure")
    expected_units = 420 * wins + sum(
        420 // (other_winners + 1) * count
        for other_winners, count in ties.items()
    )
    if units != expected_units or units > UINT64_MAX:
        raise QualificationError("equity unit conservation failure")
    if any(count for other_winners, count in ties.items() if other_winners > opponent_count):
        raise QualificationError("impossible tie counter")


def _validate_rank_rows(rows: list[dict[str, Any]]) -> None:
    for row, (identifier, cards, frozen_key) in zip(rows, RANK_ROWS):
        if tuple(row) != RANK_KEYS:
            raise QualificationError("invalid rank row shape")
        if (
            row["format_version"] != "1"
            or row["contract_version"] != "v1"
            or row["derivation"] != RANK_DERIVATION
            or row["id"] != identifier
        ):
            raise QualificationError("invalid rank metadata")
        actual_cards = _validate_cards(row["cards"], cards)
        key = row["expected_key"]
        if (
            type(key) is not list
            or len(key) != 6
            or any(type(value) is not int or not 0 <= value <= 14 for value in key)
            or tuple(key) != frozen_key
        ):
            raise QualificationError("invalid rank key")
        transparent_key = (
            score_five(actual_cards).key
            if len(actual_cards) == 5
            else best_five(actual_cards).score.key
        )
        if transparent_key != frozen_key:
            raise QualificationError("rank semantics changed")


def _validate_equity_rows(
    rows: list[dict[str, Any]],
    cases: tuple[tuple[Any, ...], ...],
) -> None:
    for row, case in zip(rows, cases):
        identifier, operation, hero, board, opponents, derivation = case
        if tuple(row) != EQUITY_KEYS:
            raise QualificationError("invalid equity row shape")
        if (
            row["format_version"] != "1"
            or row["contract_version"] != "v1"
            or row["derivation"] != derivation
            or row["id"] != identifier
            or row["operation"] != operation
        ):
            raise QualificationError("invalid equity metadata")
        _validate_cards(row["hero_cards"], hero)
        _validate_cards(row["board_cards"], board)
        if opponents is None:
            if row["opponent_holes"] is not None:
                raise QualificationError("invalid opponent topology")
            opponent_count = 1
            flat_opponents: tuple[str, ...] = ()
        else:
            expected_opponents = [list(hand) for hand in opponents]
            if row["opponent_holes"] != expected_opponents:
                raise QualificationError("invalid opponent topology")
            if type(row["opponent_holes"]) is not list:
                raise QualificationError("invalid opponent topology")
            for hand, expected_hand in zip(row["opponent_holes"], opponents):
                _validate_cards(hand, expected_hand)
            opponent_count = len(opponents)
            flat_opponents = tuple(card for hand in opponents for card in hand)
        try:
            parse_cards(hero + board + flat_opponents)
        except Exception:
            raise QualificationError("duplicate fixture cards") from None
        _validate_expected(
            row["expected"],
            _expected_from_tuple(FROZEN_RESULTS[identifier]),
            opponent_count,
        )


def _parse_manifest(data: bytes) -> tuple[tuple[str, str], ...]:
    if not data or not data.endswith(b"\n") or b"\r" in data:
        raise QualificationError("invalid manifest")
    lines = data.splitlines(keepends=True)
    if len(lines) != len(MANIFEST_NAMES):
        raise QualificationError("invalid manifest")
    entries: list[tuple[str, str]] = []
    for line, expected_name in zip(lines, MANIFEST_NAMES):
        expected_suffix = b"  " + expected_name.encode("ascii") + b"\n"
        if len(line) != 64 + len(expected_suffix) or line[64:] != expected_suffix:
            raise QualificationError("invalid manifest")
        try:
            digest = line[:64].decode("ascii")
        except UnicodeDecodeError:
            raise QualificationError("invalid manifest") from None
        if re.fullmatch(r"[0-9a-f]{64}", digest, re.ASCII) is None:
            raise QualificationError("invalid manifest")
        entries.append((digest, expected_name))
    return tuple(entries)


def verify(root: Path = ROOT) -> None:
    corpus = root / "validation/holdem/v1"
    try:
        entries = _parse_manifest(
            (corpus / "manifests/sha256sums.txt").read_bytes()
        )
        corpus_bytes = {name: (corpus / name).read_bytes() for name in CORPUS_NAMES}
        rank_rows = _parse_jsonl(
            corpus_bytes[CORPUS_NAMES[0]],
            tuple(row[0] for row in RANK_ROWS),
        )
        exact_rows = _parse_jsonl(
            corpus_bytes[CORPUS_NAMES[1]],
            tuple(case[0] for case in EXACT_CASES),
        )
        tie_rows = _parse_jsonl(
            corpus_bytes[CORPUS_NAMES[2]],
            tuple(case[0] for case in tie_cases()),
        )
        _validate_rank_rows(rank_rows)
        _validate_equity_rows(exact_rows, EXACT_CASES)
        _validate_equity_rows(tie_rows, tie_cases())
        for digest, name in entries:
            path = corpus / name if name in CORPUS_NAMES else root / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise QualificationError("manifest hash mismatch")
    except QualificationError:
        raise
    except Exception:
        raise QualificationError("fixture verification failed") from None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.release == arguments.verify:
        parser.error("select exactly one of --release or --verify")
    if arguments.release:
        release(ROOT, arguments.output)
    else:
        verify(ROOT)


if __name__ == "__main__":
    main()
