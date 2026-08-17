"""Fast corpus invariants and opt-in exhaustive release qualification."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import subprocess
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
CORPUS_PREFIX = "validation/holdem/v1/"
AUTHORITY_NAMES = (
    "tools/generate_oracle_fixtures.py", "docs/adr/0001-equity-v1-scope.md",
    "docs/adr/0002-card-rank-and-tie-semantics.md", "docs/adr/0003-deterministic-rng-and-deal-order.md",
    "contracts/v1/equity-request.schema.json", "contracts/v1/equity-result.schema.json",
    "contracts/v1/problem.schema.json", "src/poker_knight_ng/reference/cards.py",
    "src/poker_knight_ng/reference/evaluator.py", "src/poker_knight_ng/reference/enumerate.py",
 "tools/qualify_seven_card_corpus.py", "tools/seven_card_category_counter.c",
 "validation/holdem/v1/seven_card_release_qualification.json",
 )
EXPECTED_NAMES = set(tuple(CORPUS_PREFIX + name for name in CORPUS_NAMES) + AUTHORITY_NAMES)
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
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


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


def _qualifier():
    spec = importlib.util.spec_from_file_location("seven_card_qualifier", ROOT / "tools" / "qualify_seven_card_corpus.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _copied_qualifier(tmp_path):
    root = _copied_root(tmp_path)
    qualifier = _qualifier()
    qualifier.ROOT = root
    qualifier.CORPUS = root / "validation/holdem/v1"
    qualifier.EVIDENCE = qualifier.CORPUS / "seven_card_release_qualification.json"
    return root, qualifier


def test_seven_card_sample_is_canonical_unique_and_broadly_spans_deck():
    qualifier = _qualifier()
    hands = qualifier._sample_ids()
    assert len(hands) == 10_000 and hands == tuple(sorted(hands)) and len(set(hands)) == len(hands)
    explicit_categories = {best_five(tuple(CARD_DECK[i].token for i in hand)).score.key[0] for hand in hands}
    assert explicit_categories == set(range(9))
    wheel = tuple(sorted(_card_id({card.token: card for card in CARD_DECK}[token]) for token in "As 2s 3s 4s 5s Kd Qh".split()))
    assert wheel in hands
    ids = [card_id for hand in hands for card_id in hand]
    assert min(ids) == 0 and max(ids) == 51
    assert all(sum(card_id // 13 == suit for card_id in ids) > 10_000 for suit in range(4))
    assert all(sum(card_id % 13 == rank for card_id in ids) > 4_000 for rank in range(13))


def test_c_category_accelerator_sample_protocol_matches_canonical_and_wheel_hands(tmp_path):
    qualifier = _qualifier()
    executable = qualifier._compile_counter(tmp_path / "counter")
    hands = (
        (0, 1, 2, 3, 4, 18, 31),  # wheel straight flush
        tuple(sorted(_card_id({card.token: card for card in CARD_DECK}[token]) for token in "As 2h 3d 4c 5s Kd Qh".split())),
        tuple(sorted(_card_id({card.token: card for card in CARD_DECK}[token]) for token in "As Ah Ad Ac Kd Qh Js".split())),
    )
    actual = qualifier._accelerated_categories(hands, executable)
    expected = tuple(best_five(tuple(CARD_DECK[card_id].token for card_id in hand)).score.key[0] for hand in hands)
    assert actual == expected == (8, 4, 7)
    for payload in ("0 1 2 3 4 5\n", "0 1 2 3 4 5 52\n", "0 1 2 3 4 5 5\n", "0 1 2 3 4 5 nope\n", "0 1 2 3 4 5 6 extra\n", "999999999999999999999999999999999999999999999999 1 2 3 4 5 6\n", "-999999999999999999999999999999999999999999999999 1 2 3 4 5 6\n"):
        result = subprocess.run([str(executable), "--sample"], input=payload, text=True, capture_output=True)
        assert result.returncode != 0


def test_c_category_accelerator_rejects_invalid_python_hands_and_short_output(tmp_path, monkeypatch):
    qualifier = _qualifier()
    executable = qualifier._compile_counter(tmp_path / "counter")
    with pytest.raises(ValueError, match="seven distinct canonical"):
        qualifier._accelerated_categories(((0, 1, 2, 3, 4, 5, 5),), executable)

    class ShortResult:
        returncode = 0
        stdout = "4\n"
        stderr = ""
    monkeypatch.setattr(qualifier.subprocess, "run", lambda *args, **kwargs: ShortResult())
    with pytest.raises(RuntimeError, match="output cardinality"):
        qualifier._accelerated_categories(((0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12, 13)), executable)


def test_c_differential_uses_all_ten_thousand_hands_and_rejects_category_mismatch(monkeypatch):
    qualifier = _qualifier()
    seen = []
    def categories(hands, executable):
        seen.append(len(hands))
        return tuple(best_five(tuple(CARD_DECK[card_id].token for card_id in hand)).score.key[0] for hand in hands)
    monkeypatch.setattr(qualifier, "_accelerated_categories", categories)
    qualifier.differential(Path("counter"))
    assert seen == [10_000]
    monkeypatch.setattr(qualifier, "_accelerated_categories", lambda hands, executable: (9,) * len(hands))
    with pytest.raises(RuntimeError, match="C category accelerator differs"):
        qualifier.differential(Path("counter"))


def test_seven_card_evidence_verify_rejects_resigned_malformed_review_record(tmp_path):
    generator = _generator(); root, qualifier = _copied_qualifier(tmp_path)
    document = json.loads(qualifier.EVIDENCE.read_text())
    document.update(format_version="2", qualification={}, independent_engine={}, total_matches_target=1, counts_match_canonical="true", implementation_sha256={"tools/qualify_seven_card_corpus.py": ""}, extra="forged")
    qualifier.EVIDENCE.write_text(json.dumps(document, separators=(",", ":")) + "\n")
    _resign(generator, root)
    with pytest.raises(RuntimeError, match="invalid seven-card qualification evidence"):
        qualifier.verify()


@pytest.mark.parametrize("mutate", [
    lambda raw: raw.replace(b'"format_version":"1"', b'"format_version":"2"', 1),
    lambda raw: raw.replace(b'"format_version":"1"', b'"format_version":"1","format_version":"1"', 1),
    lambda raw: raw.replace(b'"total_matches_target":true', b'"total_matches_target":1'),
    lambda raw: raw.replace(b'"counts_match_canonical":true', b'"counts_match_canonical":"true"'),
    lambda raw: raw.replace(b'"optimized_scorer_differential":{"authority":"transparent-five-card-subset-evaluator","scope":', b'"optimized_scorer_differential":{"authority":"transparent-five-card-subset-evaluator","equal":1,"scope":', 1).replace(b',"equal":true}', b'}', 1),
    lambda raw: raw.replace(b'"independent_engine":{"name":"treys","version":"0.1.8","scope":', b'"independent_engine":{"name":"treys","version":"0.1.8","equal":1,"scope":', 1).replace(b',"equal":true}', b'}', 1),
    lambda raw: raw.replace(b'"optimized_scorer_differential":{"authority":"transparent-five-card-subset-evaluator","scope":', b'"optimized_scorer_differential":{"authority":"transparent-five-card-subset-evaluator","equal":"true","scope":', 1).replace(b',"equal":true}', b'}', 1),
    lambda raw: raw.replace(b'"independent_engine":{"name":"treys","version":"0.1.8","scope":', b'"independent_engine":{"name":"treys","version":"0.1.8","equal":null,"scope":', 1).replace(b',"equal":true}', b'}', 1),
    lambda raw: raw.replace(b'"optimized_scorer_differential":{"authority":"transparent-five-card-subset-evaluator","scope":', b'"optimized_scorer_differential":{"authority":"transparent-five-card-subset-evaluator","equal":false,"scope":', 1).replace(b',"equal":true}', b'}', 1),
    lambda raw: raw.replace(b'"independent_engine":{"name":"treys","version":"0.1.8","scope":', b'"independent_engine":{"name":"treys","version":"0.1.8","equal":false,"scope":', 1).replace(b',"equal":true}', b'}', 1),
    lambda raw: raw.replace(b'"target_cardinality":"133784560"', b'"target_cardinality":133784560'),
    lambda raw: raw.replace(b'"high_card":"23294460"', b'"high_card":"023294460"'),
    lambda raw: raw.replace(b'"elapsed_seconds":"', b'"elapsed_seconds":"-'),
    lambda raw: raw.replace(b'"elapsed_seconds":"', b'"unexpected":true,"elapsed_seconds":"'),
    lambda raw: raw.replace(b'"tools/qualify_seven_card_corpus.py":"', b'"unknown/path.py":"'),
    lambda raw: b'\xef\xbb\xbf' + raw,
    lambda raw: raw.replace(b'\n', b'\r\n'),
    lambda raw: raw.rstrip(b'\n'),
    lambda raw: raw + b'{}\n',
    lambda raw: raw.replace(b'"elapsed_seconds":"', b'"elapsed_seconds":NaN'),
])
def test_seven_card_evidence_verify_rejects_resigned_noncanonical_mutations(tmp_path, mutate):
    generator = _generator(); root, qualifier = _copied_qualifier(tmp_path)
    qualifier.EVIDENCE.write_bytes(mutate(qualifier.EVIDENCE.read_bytes()))
    _resign(generator, root)
    with pytest.raises(RuntimeError):
        qualifier.verify()


def _seam_release(qualifier, monkeypatch):
    monkeypatch.setenv("RUN_SEVEN_CARD_RELEASE_QUALIFICATION", "1")
    monkeypatch.setattr(qualifier, "_compile_counter", lambda executable: executable)
    monkeypatch.setattr(qualifier, "differential", lambda executable: None)
    class Result:
        stdout = "133784560 23294460 58627800 31433400 6461620 6180020 4047644 3473184 224848 41584\n"
    monkeypatch.setattr(qualifier.subprocess, "run", lambda *args, **kwargs: Result())


def test_seam_release_publishes_evidence_and_manifest_together(tmp_path, monkeypatch):
    generator = _generator(); root, qualifier = _copied_qualifier(tmp_path)
    _seam_release(qualifier, monkeypatch)
    qualifier.release()
    qualifier.verify()
    generator.verify(root)
    manifest = (qualifier.CORPUS / "manifests/sha256sums.txt").read_bytes()
    assert hashlib.sha256(qualifier.EVIDENCE.read_bytes()).hexdigest().encode() in manifest
    assert not list(qualifier.CORPUS.rglob(".oracle-fixture-*"))


def test_seam_release_reuses_valid_committed_elapsed_without_rewriting_evidence_or_manifest(tmp_path, monkeypatch):
    root, qualifier = _copied_qualifier(tmp_path)
    _seam_release(qualifier, monkeypatch)
    qualifier.EVIDENCE.unlink()
    monkeypatch.setattr(qualifier.time, "monotonic", iter((100.0, 101.25, 200.0, 203.5)).__next__)
    first = qualifier.release()
    evidence = qualifier.EVIDENCE.read_bytes(); manifest = (qualifier.CORPUS / "manifests/sha256sums.txt").read_bytes()
    second = qualifier.release()
    assert first["elapsed_seconds"] == "1.250000"
    assert second["elapsed_seconds"] == "1.250000"
    assert qualifier.EVIDENCE.read_bytes() == evidence
    assert (qualifier.CORPUS / "manifests/sha256sums.txt").read_bytes() == manifest
    assert second["measured_elapsed_seconds"] == "3.500000"


def test_seam_release_records_new_elapsed_when_qualification_state_changes(tmp_path, monkeypatch):
    root, qualifier = _copied_qualifier(tmp_path)
    _seam_release(qualifier, monkeypatch)
    monkeypatch.setattr(qualifier.time, "monotonic", iter((10.0, 11.0, 20.0, 22.0)).__next__)
    qualifier.release()
    before = qualifier.EVIDENCE.read_bytes()
    (root / "tools/seven_card_category_counter.c").write_text("changed governing implementation\n")
    second = qualifier.release()
    assert second["elapsed_seconds"] == "2.000000"
    assert qualifier.EVIDENCE.read_bytes() != before


def test_seam_release_records_new_elapsed_when_manifest_bound_authority_changes(tmp_path, monkeypatch):
    root, qualifier = _copied_qualifier(tmp_path)
    _seam_release(qualifier, monkeypatch)
    qualifier.EVIDENCE.unlink()
    monkeypatch.setattr(qualifier.time, "monotonic", iter((10.0, 11.0, 20.0, 22.0)).__next__)
    qualifier.release()
    before = qualifier.EVIDENCE.read_bytes()
    generator_path = root / "tools/generate_oracle_fixtures.py"
    generator_path.write_text(generator_path.read_text() + "\n# changed manifest-bound authority\n")
    second = qualifier.release()
    assert second["elapsed_seconds"] == "2.000000"
    assert qualifier.EVIDENCE.read_bytes() != before


@pytest.mark.parametrize("boundary", (1, 2))
def test_seam_release_rolls_back_evidence_and_manifest_at_each_publication_boundary(tmp_path, monkeypatch, boundary):
    root, qualifier = _copied_qualifier(tmp_path)
    _seam_release(qualifier, monkeypatch)
    destinations = (qualifier.EVIDENCE, qualifier.CORPUS / "manifests/sha256sums.txt")
    before = {path: path.read_bytes() for path in destinations}
    real_replace = qualifier._publication_module().os.replace
    publications = 0
    def fail_boundary(source, destination):
        nonlocal publications
        if Path(source).name.startswith(".oracle-fixture-"):
            publications += 1
            if publications == boundary:
                raise OSError("injected publication failure")
        return real_replace(source, destination)
    monkeypatch.setattr(qualifier._publication_module().os, "replace", fail_boundary)
    with pytest.raises(RuntimeError, match="fixture publication failed"):
        qualifier.release()
    assert {path: path.read_bytes() for path in destinations} == before
    assert not list(qualifier.CORPUS.rglob(".oracle-fixture-*"))


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
    assert "RUN_SEVEN_CARD_RELEASE_QUALIFICATION=1" in text
    assert "133,784,560" in text and "outstanding" not in text.lower()


def test_committed_seven_card_release_evidence_is_hash_bound_and_exact():
    generator = _generator()
    evidence = CORPUS / "seven_card_release_qualification.json"
    assert "validation/holdem/v1/seven_card_release_qualification.json" in generator.MANIFEST_NAMES
    document = json.loads(evidence.read_text())
    assert document["format_version"] == "1"
    assert document["qualification"] == "phase-2-seven-card-release-candidate"
    assert document["command"] == "RUN_SEVEN_CARD_RELEASE_QUALIFICATION=1 uv run python tools/qualify_seven_card_corpus.py --release"
    assert document["target_cardinality"] == "133784560"
    assert document["category_counts"] == {
        "high_card": "23294460", "one_pair": "58627800", "two_pair": "31433400",
        "three_of_a_kind": "6461620", "straight": "6180020", "flush": "4047644",
        "full_house": "3473184", "four_of_a_kind": "224848", "straight_flush": "41584",
    }
    assert document["total_matches_target"] is True and document["counts_match_canonical"] is True
    assert document["optimized_scorer_differential"]["equal"] is True
    assert document["independent_engine"] == {
        "name": "treys", "version": "0.1.8",
        "scope": "10000 deterministic unique seven-card hands including all categories and wheel; SHA-256 counter-derived candidates with duplicate rejection; direct C category-counter differential; C category counter and transparent evaluator agree",
        "equal": True,
    }
    assert set(document["implementation_sha256"]) == {
        "tools/qualify_seven_card_corpus.py", "tools/seven_card_category_counter.c",
        "src/poker_knight_ng/reference/evaluator.py",
    }
    assert all(len(value) == 64 for value in document["implementation_sha256"].values())
    assert isinstance(document["elapsed_seconds"], str) and document["elapsed_seconds"].count(".") == 1


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
    lambda data: data.replace(b"  validation/holdem/v1/canonical_rank_vectors.jsonl", b"\tvalidation/holdem/v1/canonical_rank_vectors.jsonl", 1),
    lambda data: data.replace(b"  validation/holdem/v1/canonical_rank_vectors.jsonl", b" validation/holdem/v1/canonical_rank_vectors.jsonl", 1),
    lambda data: data.replace(b"  validation/holdem/v1/canonical_rank_vectors.jsonl", b"   validation/holdem/v1/canonical_rank_vectors.jsonl", 1),
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


@pytest.mark.parametrize("mode", (0o644, 0o600, 0o755))
def test_atomic_write_preserves_existing_destination_mode_on_success(tmp_path, mode):
    generator = _generator()
    destination = tmp_path / "artifact"
    destination.write_bytes(b"old"); os.chmod(destination, mode)
    generator.atomic_write_paths({destination: b"new"})
    assert destination.read_bytes() == b"new"
    assert os.stat(destination).st_mode & 0o777 == mode


def test_stage_bytes_closes_descriptor_and_removes_stage_when_fchmod_fails(tmp_path, monkeypatch):
    generator = _generator()
    real_close = generator.os.close
    closed: list[int] = []

    def fail_fchmod(_fd, _mode):
        raise OSError("injected fchmod failure")

    def track_close(fd):
        closed.append(fd)
        return real_close(fd)

    monkeypatch.setattr(generator.os, "fchmod", fail_fchmod)
    monkeypatch.setattr(generator.os, "close", track_close)
    with pytest.raises(OSError, match="injected fchmod failure"):
        generator._stage_bytes(tmp_path, b"payload")
    assert len(closed) == 1
    assert not list(tmp_path.glob(".oracle-fixture-*"))


def test_atomic_write_preserves_modes_during_fallback_rollback(tmp_path, monkeypatch):
    generator = _generator()
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"old first"); second.write_bytes(b"old second")
    os.chmod(first, 0o644); os.chmod(second, 0o600)
    real_replace = generator.os.replace; publications = 0
    def fail_publish_and_atomic_restore(source, destination):
        nonlocal publications
        if Path(source).name.endswith(".tmp"):
            publications += 1
            if publications == 2: raise OSError("publish")
        elif publications >= 2: raise OSError("restore")
        return real_replace(source, destination)
    monkeypatch.setattr(generator.os, "replace", fail_publish_and_atomic_restore)
    with pytest.raises(generator.QualificationError):
        generator.atomic_write_paths({first: b"new first", second: b"new second"})
    assert first.read_bytes() == b"old first" and second.read_bytes() == b"old second"
    assert os.stat(first).st_mode & 0o777 == 0o644
    assert os.stat(second).st_mode & 0o777 == 0o600


def test_seam_release_preserves_evidence_and_manifest_modes(tmp_path, monkeypatch):
    root, qualifier = _copied_qualifier(tmp_path)
    _seam_release(qualifier, monkeypatch)
    manifest = qualifier.CORPUS / "manifests/sha256sums.txt"
    os.chmod(qualifier.EVIDENCE, 0o600); os.chmod(manifest, 0o644)
    qualifier.release()
    assert os.stat(qualifier.EVIDENCE).st_mode & 0o777 == 0o600
    assert os.stat(manifest).st_mode & 0o777 == 0o644


def test_atomic_write_falls_back_to_durable_backup_when_atomic_restore_fails(tmp_path, monkeypatch):
    generator = _generator()
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first.write_bytes(b"first original"); second.write_bytes(b"second original")
    real_replace = generator.os.replace
    publications = 0

    def fail_second_publish_and_all_restores(source, destination):
        nonlocal publications
        if Path(source).name.endswith(".tmp"):
            publications += 1
            if publications >= 2:
                raise OSError("injected publication failure")
        elif publications >= 2:
            raise OSError("injected atomic restore failure")
        return real_replace(source, destination)

    monkeypatch.setattr(generator.os, "replace", fail_second_publish_and_all_restores)
    with pytest.raises(generator.QualificationError, match=r"^fixture publication failed$"):
        generator.atomic_write_paths({first: b"first new", second: b"second new"})
    assert first.read_bytes() == b"first original"
    assert second.read_bytes() == b"second original"
    assert not list(tmp_path.glob(".oracle-fixture-*"))


def test_atomic_write_reports_incomplete_rollback_and_retains_backup_when_fallback_fails(tmp_path, monkeypatch):
    generator = _generator()
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first.write_bytes(b"first original"); second.write_bytes(b"second original")
    real_replace = generator.os.replace
    publications = 0

    def fail_second_publish_and_all_restores(source, destination):
        nonlocal publications
        if Path(source).name.endswith(".tmp"):
            publications += 1
            if publications >= 2:
                raise OSError("injected publication failure")
        elif publications >= 2:
            raise OSError("injected atomic restore failure")
        return real_replace(source, destination)

    monkeypatch.setattr(generator.os, "replace", fail_second_publish_and_all_restores)
    real_stage = generator._stage_bytes

    def fail_fallback_write(directory, data):
        if data == b"first original" and publications >= 2:
            raise OSError("injected fallback write failure")
        return real_stage(directory, data)

    # The implementation's direct fallback is deliberately a narrow seam.
    monkeypatch.setattr(generator, "_write_recovery_bytes", fail_fallback_write, raising=False)
    with pytest.raises(generator.QualificationError, match="rollback incomplete") as caught:
        generator.atomic_write_paths({first: b"first new", second: b"second new"})
    assert str(first) in str(caught.value)
    assert first.read_bytes() == b"first new"
    backups = list(tmp_path.glob(".oracle-fixture-*.bak"))
    assert len(backups) == 1 and backups[0].read_bytes() == b"first original"
    assert not list(tmp_path.glob(".oracle-fixture-*.tmp"))


def test_atomic_write_surfaces_backup_cleanup_failure_without_destroying_recovery_data(tmp_path, monkeypatch):
    generator = _generator()
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first.write_bytes(b"first original"); second.write_bytes(b"second original")
    real_replace = generator.os.replace
    publications = 0

    def fail_second_publish_and_restore(source, destination):
        nonlocal publications
        if Path(source).name.endswith(".tmp"):
            publications += 1
            if publications >= 2:
                raise OSError("injected publication failure")
        elif publications >= 2:
            raise OSError("injected atomic restore failure")
        return real_replace(source, destination)

    monkeypatch.setattr(generator.os, "replace", fail_second_publish_and_restore)
    real_cleanup = generator._cleanup

    retained_backup = None

    def fail_backup_cleanup(paths):
        nonlocal retained_backup
        paths = list(paths)
        if retained_backup is None:
            retained_backup = next(path for path in paths if path.name.endswith(".bak"))
            return [(retained_backup, "OSError")]
        return real_cleanup(paths)

    monkeypatch.setattr(generator, "_cleanup", fail_backup_cleanup)
    with pytest.raises(generator.QualificationError, match="rollback incomplete"):
        generator.atomic_write_paths({first: b"first new", second: b"second new"})
    backups = list(tmp_path.glob(".oracle-fixture-*.bak"))
    assert any(backup.read_bytes() == b"first original" for backup in backups)
    assert not list(tmp_path.glob(".oracle-fixture-*.tmp"))


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
