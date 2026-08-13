#!/usr/bin/env python3
"""Run or verify the Phase 2 all-seven-card release qualification."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from poker_knight_ng.reference.cards import CARD_DECK, _card_id
from poker_knight_ng.reference.enumerate import _score_seven_ids
from poker_knight_ng.reference.evaluator import CATEGORY_NAMES, best_five

ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "validation" / "holdem" / "v1"
EVIDENCE = CORPUS / "seven_card_release_qualification.json"
EXPECTED = (23294460, 58627800, 31433400, 6461620, 6180020, 4047644, 3473184, 224848, 41584)
TARGET = 133784560
COMMAND = "RUN_SEVEN_CARD_RELEASE_QUALIFICATION=1 uv run python tools/qualify_seven_card_corpus.py --release"
IMPLEMENTATIONS = ("tools/qualify_seven_card_corpus.py", "tools/seven_card_category_counter.c", "src/poker_knight_ng/reference/evaluator.py")
SAMPLE_SCOPE = "10000 deterministic unique seven-card hands including all categories and wheel; SHA-256 counter-derived candidates with duplicate rejection"
C_SCOPE = SAMPLE_SCOPE + "; direct C category-counter differential"
DECIMAL = re.compile(r"0|[1-9][0-9]*\Z")
ELAPSED = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{6}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")


def _sample_ids() -> tuple[tuple[int, ...], ...]:
    """Stable SHA-256 counter sample; no Python RNG/hash ordering dependency."""
    explicit = (
        "As Kd Qh Jc 9s 8d 7c", "As Ah Kc Qd Jh 9s 8d", "As Ad Kc Kd Qh Js 9c",
        "As Ah Ad Kc Qd Jh 9s", "As 2h 3d 4c 5s Kd Qh", "Ah Kh 9h 5h 2h Qd Js",
        "As Ah Ad Ks Kh Kd Qc", "As Ah Ad Ac Kd Qh Js", "As 2s 3s 4s 5s Kd Qh",
    )
    by_token = {card.token: card for card in CARD_DECK}
    values = {tuple(sorted(_card_id(by_token[token]) for token in hand.split())) for hand in explicit}
    counter = 0
    while len(values) < 10_000:
        # Rejection-sample seven distinct IDs from a SHA-256 byte stream.
        stream = b""; block = 0
        while len(stream) < 56:
            stream += hashlib.sha256(f"poker-knight-ng/phase-2-seven-card-sample/{counter}/{block}".encode()).digest(); block += 1
        hand = []
        for value in stream:
            if value < 208:
                card = value % 52
                if card not in hand:
                    hand.append(card)
                    if len(hand) == 7:
                        values.add(tuple(sorted(hand))); break
        counter += 1
    return tuple(sorted(values))


def _compile_counter(executable: Path) -> Path:
    subprocess.run(["cc", "-O3", "-std=c11", "-Wall", "-Wextra", "-Werror", "-o", str(executable), str(ROOT / "tools/seven_card_category_counter.c")], check=True, capture_output=True, text=True)
    return executable


def _accelerated_categories(hands: tuple[tuple[int, ...], ...], executable: Path) -> tuple[int, ...]:
    for hand in hands:
        if len(hand) != 7 or any(type(card) is not int or not 0 <= card <= 51 for card in hand) or len(set(hand)) != 7:
            raise ValueError("sample hand must contain seven distinct canonical IDs 0..51")
    payload = "".join(" ".join(str(card) for card in hand) + "\n" for hand in hands)
    result = subprocess.run([str(executable), "--sample"], input=payload, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("C category accelerator sample protocol failed")
    lines = result.stdout.splitlines()
    if len(lines) != len(hands):
        raise RuntimeError("C category accelerator output cardinality mismatch")
    if any(not re.fullmatch(r"[0-8]", line) for line in lines):
        raise RuntimeError("C category accelerator output category out of range")
    return tuple(int(line) for line in lines)


def differential(executable: Path) -> None:
    try:
        import treys
    except ImportError as error:
        raise RuntimeError("pinned treys==0.1.8 is required") from error
    if getattr(treys, "__version__", "0.1.8") != "0.1.8": raise RuntimeError("unexpected Treys version")
    evaluator = treys.Evaluator()
    hands = _sample_ids()
    accelerated = _accelerated_categories(hands, executable)
    for ids, c_category in zip(hands, accelerated):
        cards = tuple(CARD_DECK[card_id].token for card_id in ids)
        shortcut = _score_seven_ids(ids)
        if shortcut != best_five(cards).score.key: raise RuntimeError("optimized scorer differs from transparent evaluator")
        transparent_category = shortcut[0]
        if c_category != transparent_category: raise RuntimeError("C category accelerator differs from transparent evaluator")
        treys_cards = [treys.Card.new(card) for card in cards]
        treys_category = 9 - evaluator.get_rank_class(evaluator.evaluate([], treys_cards))
        if treys_category != transparent_category: raise RuntimeError("Treys category differs from transparent evaluator")
        if c_category != treys_category: raise RuntimeError("C category accelerator differs from Treys")


def _digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def _publication_module():
    """Load the shared staged-publication implementation from this checkout."""
    spec = importlib.util.spec_from_file_location(
        "seven_card_publication", ROOT / "tools/generate_oracle_fixtures.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load fixture publication helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def release() -> dict[str, object]:
    if os.environ.get("RUN_SEVEN_CARD_RELEASE_QUALIFICATION") != "1": raise RuntimeError("set RUN_SEVEN_CARD_RELEASE_QUALIFICATION=1")
    with tempfile.TemporaryDirectory(prefix="poker-knight-ng-seven-card-") as directory:
        executable = _compile_counter(Path(directory) / "category-counter")
        differential(executable)
        started = time.monotonic()
        output = subprocess.run([str(executable)], check=True, capture_output=True, text=True).stdout.strip().split()
    elapsed = time.monotonic() - started
    values = tuple(int(value) for value in output)
    if len(values) != 10 or values[0] != TARGET or values[1:] != EXPECTED:
        raise RuntimeError("full seven-card category corpus mismatch")
    document = {
        "format_version": "1", "qualification": "phase-2-seven-card-release-candidate",
        "command": COMMAND, "target_cardinality": str(TARGET),
        "category_counts": {name: str(value) for name, value in zip(CATEGORY_NAMES, values[1:])},
        "total_matches_target": True, "counts_match_canonical": True,
        "optimized_scorer_differential": {"authority": "transparent-five-card-subset-evaluator", "scope": SAMPLE_SCOPE, "equal": True},
        "independent_engine": {"name": "treys", "version": "0.1.8", "scope": C_SCOPE + "; C category counter and transparent evaluator agree", "equal": True},
        "implementation_sha256": {name: _digest(ROOT / name) for name in IMPLEMENTATIONS},
        "elapsed_seconds": f"{elapsed:.6f}",
    }
    publication = _publication_module()
    # A valid existing record can donate elapsed only when both its canonical
    # semantics and the complete manifest-bound authority state still verify.
    try:
        verify()
        publication.verify(ROOT)
        existing = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        if {key: value for key, value in existing.items() if key != "elapsed_seconds"} == {key: value for key, value in document.items() if key != "elapsed_seconds"}:
            document["elapsed_seconds"] = existing["elapsed_seconds"]
    except (RuntimeError, publication.QualificationError):
        pass
    evidence = json.dumps(document, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"
    corpus_outputs = {name: (CORPUS / name).read_bytes() for name in publication.CORPUS_NAMES}
    relative_evidence = EVIDENCE.relative_to(ROOT).as_posix()
    manifest = publication.manifest(ROOT, corpus_outputs, {relative_evidence: evidence})
    publication.atomic_write_paths({EVIDENCE: evidence, CORPUS / "manifests/sha256sums.txt": manifest})
    return {**document, "measured_elapsed_seconds": f"{elapsed:.6f}"}


def _reject_constant(value: str): raise ValueError(value)
def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError("duplicate key")
        result[key] = value
    return result

def _invalid(): raise RuntimeError("invalid seven-card qualification evidence")
def _canonical_decimal(value): return isinstance(value, str) and DECIMAL.fullmatch(value) is not None

def verify() -> None:
    try:
        raw = EVIDENCE.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or not raw.endswith(b"\n") or raw.count(b"\n") != 1: _invalid()
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant)
        top = {"format_version","qualification","command","target_cardinality","category_counts","total_matches_target","counts_match_canonical","optimized_scorer_differential","independent_engine","implementation_sha256","elapsed_seconds"}
        if not isinstance(document, dict) or set(document) != top: _invalid()
        if document["format_version"] != "1" or document["qualification"] != "phase-2-seven-card-release-candidate" or document["command"] != COMMAND or document["target_cardinality"] != str(TARGET): _invalid()
        counts = document["category_counts"]
        if not isinstance(counts, dict) or tuple(counts) != CATEGORY_NAMES or any(not _canonical_decimal(counts[name]) for name in CATEGORY_NAMES) or tuple(int(counts[name]) for name in CATEGORY_NAMES) != EXPECTED: _invalid()
        if type(document["total_matches_target"]) is not bool or document["total_matches_target"] is not True or type(document["counts_match_canonical"]) is not bool or document["counts_match_canonical"] is not True: _invalid()
        optimized = document["optimized_scorer_differential"]
        if not isinstance(optimized, dict) or set(optimized) != {"authority", "scope", "equal"} or type(optimized["authority"]) is not str or optimized["authority"] != "transparent-five-card-subset-evaluator" or type(optimized["scope"]) is not str or optimized["scope"] != SAMPLE_SCOPE or type(optimized["equal"]) is not bool or optimized["equal"] is not True: _invalid()
        independent = document["independent_engine"]
        if not isinstance(independent, dict) or set(independent) != {"name", "version", "scope", "equal"} or type(independent["name"]) is not str or independent["name"] != "treys" or type(independent["version"]) is not str or independent["version"] != "0.1.8" or type(independent["scope"]) is not str or independent["scope"] != C_SCOPE + "; C category counter and transparent evaluator agree" or type(independent["equal"]) is not bool or independent["equal"] is not True: _invalid()
        hashes = document["implementation_sha256"]
        if not isinstance(hashes, dict) or tuple(hashes) != IMPLEMENTATIONS or any(not isinstance(hashes[name],str) or HEX.fullmatch(hashes[name]) is None for name in IMPLEMENTATIONS): _invalid()
        if not isinstance(document["elapsed_seconds"], str) or ELAPSED.fullmatch(document["elapsed_seconds"]) is None: _invalid()
        for name in IMPLEMENTATIONS:
            if _digest(ROOT / name) != hashes[name]: raise RuntimeError("seven-card implementation hash mismatch")
    except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid seven-card qualification evidence") from error


def main() -> None:
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True); group.add_argument("--release",action="store_true"); group.add_argument("--verify",action="store_true"); arguments=parser.parse_args()
    if arguments.release: print(json.dumps(release(),separators=(",",":")))
    else: verify()
if __name__ == "__main__": main()
