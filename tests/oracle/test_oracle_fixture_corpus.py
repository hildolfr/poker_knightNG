"""Fast corpus invariants and opt-in exhaustive release qualification."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import shutil
from pathlib import Path

import pytest

from poker_knight_ng.reference.cards import CARD_DECK, _card_id
from poker_knight_ng.reference.enumerate import _score_seven_ids
from poker_knight_ng.reference.evaluator import CATEGORY_NAMES, best_five, score_five

ROOT = Path(__file__).parents[2]
CORPUS = ROOT / "validation" / "holdem" / "v1"
MANIFEST = CORPUS / "manifests" / "sha256sums.txt"
CORPUS_NAMES = ("canonical_rank_vectors.jsonl", "exact_holdem_cases.jsonl", "tie_and_split_cases.jsonl")
AUTHORITY_NAMES = (
    "tools/generate_oracle_fixtures.py", "docs/adr/0001-equity-v1-scope.md",
    "docs/adr/0002-card-rank-and-tie-semantics.md", "docs/adr/0003-deterministic-rng-and-deal-order.md",
    "contracts/v1/equity-request.schema.json", "contracts/v1/equity-result.schema.json",
    "contracts/v1/problem.schema.json", "src/poker_knight_ng/reference/cards.py",
    "src/poker_knight_ng/reference/evaluator.py", "src/poker_knight_ng/reference/enumerate.py",
)
EXPECTED_NAMES = set(CORPUS_NAMES + AUTHORITY_NAMES)
CATEGORY_KEYS = set(CATEGORY_NAMES)
EXACT_KEYS = {"format_version", "contract_version", "derivation", "id", "operation", "hero_cards", "board_cards", "opponent_holes", "expected"}
EXPECTED_KEYS = {"completed_trials", "unique_wins", "tie_by_other_winners", "losses", "equity_share_units", "hero_category_counts"}


def records(name: str):
    return [json.loads(line) for line in (CORPUS / name).read_text().splitlines() if line]


def as_int(value: str) -> int:
    assert isinstance(value, str) and (value == "0" or (value[0] != "0" and value.isdecimal()))
    return int(value)


def _generator():
    import importlib.util
    spec = importlib.util.spec_from_file_location("oracle_fixture_generator", ROOT / "tools" / "generate_oracle_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifest_has_exact_hash_bound_corpus_and_authority_set():
    entries = {name: digest for digest, name in (line.split("  ", 1) for line in MANIFEST.read_text().splitlines() if line and not line.startswith("#"))}
    assert set(entries) == EXPECTED_NAMES
    for path, digest in entries.items():
        assert len(digest) == 64
        assert hashlib.sha256((ROOT / path if path in AUTHORITY_NAMES else CORPUS / path).read_bytes()).hexdigest() == digest


def test_rank_rows_are_closed_and_reference_evaluated():
    rows = records("canonical_rank_vectors.jsonl")
    assert len(rows) == 10
    for row in rows:
        assert set(row) == {"format_version", "contract_version", "derivation", "id", "cards", "expected_key"}
        assert row["format_version"] == "1" and row["contract_version"] == "v1"
        actual = best_five(row["cards"]).score.key if len(row["cards"]) > 5 else score_five(row["cards"]).key
        assert list(actual) == row["expected_key"]


def test_score_seven_shortcut_matches_transparent_best_five_for_ten_thousand_unique_hands():
    explicit = (
        "As Kd Qh Jc 9s 8d 7c",  # high card
        "As Ah Kc Qd Jh 9s 8d",  # one pair
        "As Ad Kc Kd Qh Js 9c",  # two pair
        "As Ah Ad Kc Qd Jh 9s",  # trips
        "As 2h 3d 4c 5s Kd Qh",  # wheel
        "Ah Kh 9h 5h 2h Qd Js",  # flush
        "As Ah Ad Ks Kh Kd Qc",  # full house
        "As Ah Ad Ac Kd Qh Js",  # quads
        "As 2s 3s 4s 5s Kd Qh",  # wheel straight flush
    )
    deck_by_token = {card.token: card for card in CARD_DECK}
    hands = {tuple(sorted(_card_id(deck_by_token[token]) for token in cards.split())) for cards in explicit}
    rng = random.Random(0xC0FFEE)
    while len(hands) < 10_000:
        hands.add(tuple(sorted(_card_id(card) for card in rng.sample(CARD_DECK, 7))))
    assert len(hands) == 10_000
    for ids in hands:
        cards = tuple(CARD_DECK[card_id].token for card_id in ids)
        assert _score_seven_ids(ids) == best_five(cards).score.key


def test_exact_and_tie_rows_are_closed_canonical_and_conserved():
    exact = records("exact_holdem_cases.jsonl")
    assert [row["id"] for row in exact] == ["unknown-river", "unknown-turn", "unknown-flop", "fixed-holes-turn", "fixed-holes-preflop-aa-vs-kk", "terminal-loss"]
    assert [as_int(row["expected"]["completed_trials"]) for row in exact] == [990, 45540, 1070190, 44, 1712304, 1]
    ties = records("tie_and_split_cases.jsonl")
    assert len(ties) == 6
    for row in exact + ties:
        assert set(row) == EXACT_KEYS
        assert row["format_version"] == "1" and row["contract_version"] == "v1"
        assert row["derivation"]
        expected = row["expected"]
        assert set(expected) == EXPECTED_KEYS
        assert set(expected["tie_by_other_winners"]) == {str(i) for i in range(1, 7)}
        assert set(expected["hero_category_counts"]) == CATEGORY_KEYS
        n = as_int(expected["completed_trials"]); wins = as_int(expected["unique_wins"]); losses = as_int(expected["losses"]); units = as_int(expected["equity_share_units"])
        bins = {int(k): as_int(v) for k, v in expected["tie_by_other_winners"].items()}
        cats = [as_int(expected["hero_category_counts"][key]) for key in CATEGORY_NAMES]
        assert wins + sum(bins.values()) + losses == n
        assert sum(cats) == n
        assert units == 420 * wins + sum(420 // (k + 1) * v for k, v in bins.items())
    for k, row in enumerate(ties, 1):
        assert as_int(row["expected"]["tie_by_other_winners"][str(k)]) == 1
        assert as_int(row["expected"]["equity_share_units"]) == 420 // (k + 1)


def test_readme_copies_and_qualification_truth_are_consistent():
    assert (ROOT / "README.md").read_bytes() == (ROOT / "src/poker_knight_ng/README.md").read_bytes()
    text = (CORPUS / "QUALIFICATION.md").read_text()
    assert "RUN_ORACLE_RELEASE_QUALIFICATION=1" in text
    assert "133,784,560" in text and "outstanding" in text.lower()


def test_generator_verify_is_fast_and_nonmutating(tmp_path):
    generator = _generator()
    before = {name: (CORPUS / name).read_bytes() for name in CORPUS_NAMES + ("manifests/sha256sums.txt",)}
    generator.verify(ROOT)
    assert before == {name: (CORPUS / name).read_bytes() for name in before}


def _copied_root(tmp_path):
    """Copy the verifier's fixed authority and corpus inputs, never real corpus."""
    root = tmp_path / "root"
    shutil.copytree(ROOT / "validation", root / "validation")
    for name in AUTHORITY_NAMES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    return root


def _resign(generator, root):
    corpus = root / "validation/holdem/v1"
    outputs = {name: (corpus / name).read_bytes() for name in CORPUS_NAMES}
    (corpus / "manifests/sha256sums.txt").write_bytes(generator.manifest(root, outputs))


@pytest.mark.parametrize("name,payload", [
    ("canonical_rank_vectors.jsonl", b"{not-json}\n"),
    ("canonical_rank_vectors.jsonl", b'{"format_version":"1","format_version":"1"}\n'),
    ("exact_holdem_cases.jsonl", b'{"id":"x","expected":{"losses":"0","losses":"1"}}\n'),
    ("canonical_rank_vectors.jsonl", b'{"format_version":"1","x":NaN}\n'),
    ("canonical_rank_vectors.jsonl", b'{ "format_version":"1"}\n'),
    ("canonical_rank_vectors.jsonl", b'{"format_version":"1"}\r\n'),
    ("canonical_rank_vectors.jsonl", b'{"format_version":"1"}'),
    ("canonical_rank_vectors.jsonl", b'\xef\xbb\xbf{"format_version":"1"}\n'),
    ("canonical_rank_vectors.jsonl", b'{"format_version":"\xff"}\n'),
])
def test_verify_rejects_signed_noncanonical_jsonl(tmp_path, name, payload):
    generator = _generator(); root = _copied_root(tmp_path)
    (root / "validation/holdem/v1" / name).write_bytes(payload)
    _resign(generator, root)
    with pytest.raises(generator.QualificationError):
        generator.verify(root)


def test_verify_rejects_signed_duplicate_row_and_counter_forgery(tmp_path):
    generator = _generator(); root = _copied_root(tmp_path); corpus = root / "validation/holdem/v1"
    rows = corpus.joinpath("exact_holdem_cases.jsonl").read_bytes().splitlines()
    corpus.joinpath("exact_holdem_cases.jsonl").write_bytes(
        rows[0] + b"\n" + rows[0] + b"\n" + b"\n".join(rows[1:]) + b"\n"
    )
    _resign(generator, root)
    with pytest.raises(generator.QualificationError): generator.verify(root)
    root = _copied_root(tmp_path / "again"); corpus = root / "validation/holdem/v1"
    data = corpus.joinpath("exact_holdem_cases.jsonl").read_bytes().replace(b'"unique_wins":"268"', b'"unique_wins":"true"', 1)
    corpus.joinpath("exact_holdem_cases.jsonl").write_bytes(data); _resign(generator, root)
    with pytest.raises(generator.QualificationError): generator.verify(root)


def _mutate_first_exact(root, mutation):
    path = root / "validation/holdem/v1/exact_holdem_cases.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows[0])
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(format_version="2"),
    lambda row: row.update(contract_version="v2"),
    lambda row: row.update(derivation="unqualified"),
    lambda row: row.update(operation="terminal"),
    lambda row: row.update(hero_cards=["As", "Qd"]),
    lambda row: row.update(extra="field"),
    lambda row: row.pop("operation"),
    lambda row: row["expected"].update(extra="field"),
    lambda row: row["expected"].pop("losses"),
    lambda row: row["expected"].update(unique_wins=268),
    lambda row: row["expected"].update(unique_wins=True),
    lambda row: row["expected"].update(unique_wins="0268"),
    lambda row: row["expected"].update(unique_wins="18446744073709551616"),
    lambda row: row["expected"].update(unique_wins="269"),
    lambda row: row["expected"].update(equity_share_units="114451"),
    lambda row: row["expected"]["hero_category_counts"].update(high_card="989"),
    lambda row: row["expected"]["tie_by_other_winners"].update({"2": "1"}),
])
def test_verify_rejects_signed_semantic_forgery(tmp_path, mutation):
    generator = _generator()
    root = _copied_root(tmp_path)
    _mutate_first_exact(root, mutation)
    _resign(generator, root)
    with pytest.raises(generator.QualificationError):
        generator.verify(root)


def test_verify_rejects_signed_missing_or_reordered_rows(tmp_path):
    generator = _generator()
    for label, mutation in (
        ("missing", lambda rows: rows[1:]),
        ("reordered", lambda rows: [rows[1], rows[0], *rows[2:]]),
    ):
        root = _copied_root(tmp_path / label)
        path = root / "validation/holdem/v1/exact_holdem_cases.jsonl"
        rows = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(mutation(rows)))
        _resign(generator, root)
        with pytest.raises(generator.QualificationError):
            generator.verify(root)


@pytest.mark.parametrize("mutate", [
    lambda data: data + data.splitlines()[0] + b"\n",
    lambda data: b"\n".join(data.splitlines()[:-1]) + b"\n",
    lambda data: b"\n".join([data.splitlines()[1], data.splitlines()[0], *data.splitlines()[2:]]) + b"\n",
    lambda data: data.replace(b"  canonical_rank_vectors.jsonl", b"\tcanonical_rank_vectors.jsonl", 1),
    lambda data: data.replace(b"  canonical_rank_vectors.jsonl", b" canonical_rank_vectors.jsonl", 1),
    lambda data: data.replace(b"  canonical_rank_vectors.jsonl", b"   canonical_rank_vectors.jsonl", 1),
    lambda data: data.replace(b"canonical_rank_vectors.jsonl", b"../bad", 1),
    lambda data: data.replace(b"canonical_rank_vectors.jsonl", b"/absolute", 1),
    lambda data: data.replace(b"canonical_rank_vectors.jsonl", b"bad\\path", 1),
    lambda data: data[:1].upper() + data[1:],
    lambda data: data[1:],
    lambda data: data.replace(b"\n", b"\r\n"),
    lambda data: data.rstrip(b"\n"),
])
def test_verify_rejects_ambiguous_or_noncanonical_manifest(tmp_path, mutate):
    generator = _generator()
    root = _copied_root(tmp_path)
    path = root / "validation/holdem/v1/manifests/sha256sums.txt"
    path.write_bytes(mutate(path.read_bytes()))
    with pytest.raises(generator.QualificationError):
        generator.verify(root)


def test_release_failure_is_atomic(tmp_path, monkeypatch):
    generator = _generator()
    out = tmp_path / "v1"; out.mkdir()
    sentinel = out / "exact_holdem_cases.jsonl"; sentinel.write_bytes(b"unchanged")
    monkeypatch.setattr(generator, "treys_equity", lambda *args, **kwargs: (_ for _ in ()).throw(generator.QualificationError("deliberate mismatch")))
    with pytest.raises(generator.QualificationError):
        generator.release(ROOT, out, case_filter={"terminal-loss"})
    assert sentinel.read_bytes() == b"unchanged"


def test_atomic_write_rolls_back_all_destinations_after_publication_failure(tmp_path, monkeypatch):
    generator = _generator()
    out = tmp_path / "v1"
    original = {
        "canonical_rank_vectors.jsonl": b"rank original",
        "exact_holdem_cases.jsonl": b"exact original",
        "tie_and_split_cases.jsonl": b"tie original",
        "sha256sums.txt": b"manifest original",
    }
    out.mkdir(); (out / "manifests").mkdir()
    for name, data in original.items():
        (out / name if name in CORPUS_NAMES else out / "manifests" / name).write_bytes(data)
    outputs = {name: b"new " + name.encode() for name in original}
    real_replace = generator.os.replace
    publications = 0

    def fail_third_publication(source, destination):
        nonlocal publications
        if Path(source).name.startswith(".oracle-fixture-"):
            publications += 1
            if publications == 3:
                raise OSError("injected publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(generator.os, "replace", fail_third_publication)
    with pytest.raises(generator.QualificationError, match="fixture publication failed"):
        generator.atomic_write(out, outputs)
    assert {name: (out / name if name in CORPUS_NAMES else out / "manifests" / name).read_bytes() for name in original} == original


def test_atomic_write_removes_initially_absent_destinations_after_failure(tmp_path, monkeypatch):
    generator = _generator()
    out = tmp_path / "v1"; out.mkdir(); (out / "manifests").mkdir()
    outputs = {name: b"new " + name.encode() for name in (*CORPUS_NAMES, "sha256sums.txt")}
    real_replace = generator.os.replace
    publications = 0

    def fail_second_publication(source, destination):
        nonlocal publications
        if Path(source).name.startswith(".oracle-fixture-"):
            publications += 1
            if publications == 2:
                raise OSError("injected publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(generator.os, "replace", fail_second_publication)
    with pytest.raises(generator.QualificationError, match="fixture publication failed"):
        generator.atomic_write(out, outputs)
    assert not any((out / name).exists() for name in CORPUS_NAMES)
    assert not (out / "manifests" / "sha256sums.txt").exists()

@pytest.mark.skipif(os.environ.get("RUN_ORACLE_RELEASE_QUALIFICATION") != "1", reason="set RUN_ORACLE_RELEASE_QUALIFICATION=1 for exhaustive transparent+Treys gate")
def test_release_qualification_regenerates_byte_identical_corpus(tmp_path):
    generator = _generator()
    out = tmp_path / "v1"
    generator.release(ROOT, out)
    for name in CORPUS_NAMES:
        assert (out / name).read_bytes() == (CORPUS / name).read_bytes()
    assert (out / "manifests/sha256sums.txt").read_bytes() == MANIFEST.read_bytes()
