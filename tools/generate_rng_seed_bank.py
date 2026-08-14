#!/usr/bin/env python3
"""Generate, independently verify, and atomically publish the RNG seed bank.

The checked-in bank is never hand-edited: this program derives every replay
counter, independently replays each generated row with a separately implemented
Philox/deal path, binds source and artifact bytes in a dedicated manifest, then
publishes the bank and manifest as a recoverable atomic bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from decimal import Decimal, getcontext
import re

from poker_knight_ng.reference.cards import CARD_DECK
from poker_knight_ng.reference.evaluator import best_five, CATEGORY_NAMES
from poker_knight_ng.reference.enumerate import enumerate_unknown_opponent
from poker_knight_ng.reference.monte_carlo import run_cpu_monte_carlo

ROOT = Path(__file__).parents[1]
BANK_NAME = "rng_seed_bank.json"
MANIFEST_NAME = "rng_seed_bank.sha256"
BANK_PATH = Path("validation/holdem/v1") / BANK_NAME
MANIFEST_PATH = Path("validation/holdem/v1/manifests") / MANIFEST_NAME
AUTHORITY_NAMES = (
    "tools/generate_rng_seed_bank.py",
    "tools/generate_oracle_fixtures.py",
    "docs/adr/0002-card-rank-and-tie-semantics.md",
    "docs/adr/0003-deterministic-rng-and-deal-order.md",
    "validation/holdem/v1/exact_holdem_cases.jsonl",
    "validation/holdem/v1/SPEC.md",
    "src/poker_knight_ng/reference/cards.py",
    "src/poker_knight_ng/reference/evaluator.py",
    "src/poker_knight_ng/reference/enumerate.py",
    "src/poker_knight_ng/reference/monte_carlo.py",
    "src/poker_knight_ng/reference/dealer.py",
    "src/poker_knight_ng/reference/rng.py",
)
UINT32 = 1 << 32
M0, M1, W0, W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
CASE_LABEL, KEY_LABEL = b"poker-knight-ng/case/v1", b"poker-knight-ng/rng-key/v1"

EXACT_SPECS = (
    ("adr0003-known-flop-prefix-64", 0x0123456789ABCDEF, (12, 25), (0, 14, 34), 2, 64),
    ("river-high-card-prefix-64", 1, (12, 37), (0, 18, 33, 47, 48), 1, 64),
    ("preflop-extreme-seed-prefix-64", 0xFFFFFFFFFFFFFFFF, (0, 51), (), 1, 64),
)
STAT_SPECS = (("river-high-card-wtl-2000", 1, (12, 37), (0, 18, 33, 47, 48), 1, 2000),)
CONFIDENCE = {"method": "Wilson score interval", "two_sided_alpha": "0.000001", "z": "4.891638475698591"}


class SeedBankError(RuntimeError):
    """Stable generation, verification, or publication failure."""


def _canonical(hero: tuple[int, int], board: tuple[int, ...], opponents: int) -> bytes:
    hero, board = tuple(sorted(hero)), tuple(sorted(board))
    if len(hero) != 2 or len(board) not in (0, 3, 4, 5) or not 1 <= opponents <= 6 or len(set(hero + board)) != 2 + len(board):
        raise SeedBankError("invalid fixed seed-bank topology")
    return bytes([len(CASE_LABEL)]) + CASE_LABEL + bytes([2, *hero, len(board), *board, opponents])


def _philox(counter: tuple[int, int, int, int], key: tuple[int, int]) -> tuple[int, int, int, int]:
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for round_number in range(10):
        p0, p1 = M0 * c0, M1 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0, p1 & 0xffffffff, (p0 >> 32) ^ c3 ^ k1, p0 & 0xffffffff)
        if round_number != 9:
            k0, k1 = (k0 + W0) & 0xffffffff, (k1 + W1) & 0xffffffff
    return c0, c1, c2, c3


def _independent_result(seed: int, hero: tuple[int, int], board: tuple[int, ...], opponents: int, trials: int) -> dict[str, Any]:
    canonical = _canonical(hero, board, opponents)
    digest = hashlib.sha256(canonical).digest()
    key_digest = hashlib.sha256(bytes([len(KEY_LABEL)]) + KEY_LABEL + seed.to_bytes(8, "little") + digest).digest()
    key = (int.from_bytes(key_digest[:4], "little"), int.from_bytes(key_digest[4:8], "little"))
    totals: dict[str, Any] = {"completed_trials": 0, "unique_wins": 0, "tie_by_other_winners": [0] * 6, "losses": 0, "equity_share_units": 0, "hero_category_counts": [0] * 9, "rejection_count": 0}
    known = set(hero + board)
    for simulation_id in range(trials):
        deck = [card for card in range(52) if card not in known]
        drawn: list[int] = []
        for slot in range(5 - len(board) + 2 * opponents):
            active = len(deck)
            limit = (UINT32 // active) * active
            attempt = 0
            while True:
                word = _philox((simulation_id & 0xffffffff, simulation_id >> 32, slot, attempt), key)[0]
                if word < limit:
                    index = word % active
                    drawn.append(deck[index])
                    deck[index] = deck[-1]
                    deck.pop()
                    totals["rejection_count"] += attempt
                    break
                attempt += 1
                if attempt >= UINT32:
                    raise SeedBankError("independent rejection exhaustion")
        completed_board = board + tuple(drawn[: 5 - len(board)])
        hero_score = best_five(tuple(CARD_DECK[card] for card in hero + completed_board)).score
        totals["hero_category_counts"][hero_score.category] += 1
        scores = [best_five(tuple(CARD_DECK[card] for card in drawn[5 - len(board) + 2 * index: 5 - len(board) + 2 * index + 2] + list(completed_board))).score for index in range(opponents)]
        maximum = max(scores)
        totals["completed_trials"] += 1
        if maximum < hero_score:
            totals["unique_wins"] += 1; totals["equity_share_units"] += 420
        elif maximum > hero_score:
            totals["losses"] += 1
        else:
            equals = sum(score == hero_score for score in scores)
            totals["tie_by_other_winners"][equals - 1] += 1
            totals["equity_share_units"] += 420 // (equals + 1)
    return totals


def _wire_result(result):
    return {"completed_trials":str(result["completed_trials"]),"unique_wins":str(result["unique_wins"]),"tie_by_other_winners":{str(i+1):str(v) for i,v in enumerate(result["tie_by_other_winners"])},"losses":str(result["losses"]),"equity_share_units":str(result["equity_share_units"]),"hero_category_counts":{n:str(result["hero_category_counts"][i]) for i,n in enumerate(CATEGORY_NAMES)},"rejection_count":str(result["rejection_count"])}

def _case_dict(spec: tuple[Any, ...], expected: dict[str, Any], include_bytes: bool) -> dict[str, Any]:
    identifier, seed, hero, board, opponents, trials = spec
    canonical = _canonical(hero, board, opponents)
    row: dict[str, Any] = {"id": identifier, "seed": f"0x{seed:016x}", "hero_card_ids": list(hero), "board_card_ids": list(board), "opponent_count": opponents, "requested_trials": str(trials)}
    if include_bytes:
        row["canonical_case_bytes_hex"] = canonical.hex()
    row["canonical_case_hash_hex"] = hashlib.sha256(canonical).hexdigest()
    row["expected"] = _wire_result(expected)
    return row


def build(root: Path = ROOT) -> dict[str, bytes]:
    # Specification declaration order is part of the frozen artifact contract.
    exact = [_case_dict(spec, _independent_result(*spec[1:]), True) for spec in EXACT_SPECS]
    statistical = [_case_dict(spec, _independent_result(*spec[1:]), False) for spec in STAT_SPECS]
    # Independently enumerated exact river population is the preregistered W/T/L estimand.
    population = enumerate_unknown_opponent(("As", "Kd"), ("2s", "7h", "9d", "Tc", "Jc"))
    statistical[0]["confidence"] = CONFIDENCE
    statistical[0]["estimands"] = {"unique_win": {"numerator": str(population.unique_wins), "denominator": str(population.completed_trials)}, "tie": {"numerator": str(sum(population.tie_by_other_winners)), "denominator": str(population.completed_trials)}, "loss": {"numerator": str(population.losses), "denominator": str(population.completed_trials)}}
    statistical[0]["bounded_mean_equity"]={"method":"Hoeffding bounded mean","two_sided_alpha":"0.000001","range":["0","1"],"formula":"sqrt(log(2/alpha)/(2*N))","population_exact_units":str(population.equity_share_units),"population_N":str(population.completed_trials)}
    bank = {"format_version": "1", "purpose": "Phase 3 checkpoint D generated, independently verified, hash-bound deterministic CPU seed-bank verification", "rng": {"algorithm_id": "poker-knight-ng/philox4x32-10", "algorithm_version": "1"}, "exact_vectors": exact, "statistical_vectors": statistical}
    bank_bytes = (json.dumps(bank, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    manifest_entries = [(hashlib.sha256(bank_bytes).hexdigest(), BANK_PATH.as_posix())]
    manifest_entries.extend((hashlib.sha256((root / name).read_bytes()).hexdigest(), name) for name in AUTHORITY_NAMES)
    manifest_bytes = "".join(f"{digest}  {name}\n" for digest, name in manifest_entries).encode("ascii")
    return {BANK_NAME: bank_bytes, MANIFEST_NAME: manifest_bytes}


def _result_fields(result: Any) -> dict[str, Any]:
    return {"completed_trials": result.completed_trials, "unique_wins": result.unique_wins, "tie_by_other_winners": list(result.tie_by_other_winners), "losses": result.losses, "equity_share_units": result.equity_share_units, "hero_category_counts": list(result.hero_category_counts), "rejection_count": result.rejection_count}


COUNTER_RE = re.compile(r"0|[1-9][0-9]*", re.ASCII)
UINT64_MAX = (1 << 64) - 1
def _reject_duplicate_keys(pairs):
    output = {}
    for key, value in pairs:
        if key in output: raise SeedBankError("duplicate JSON key")
        output[key] = value
    return output
def _reject_constant(_): raise SeedBankError("non-finite JSON constant")
def _uint(value, name, *, positive=False):
    if type(value) is not str or COUNTER_RE.fullmatch(value) is None or int(value) > UINT64_MAX or (positive and value == "0"):
        raise SeedBankError(f"invalid {name}")
    return int(value)

def _parse_bank(data):
    if not data or not data.endswith(b"\n") or b"\r" in data or data.startswith(b"\xef\xbb\xbf"): raise SeedBankError("noncanonical bank framing")
    try: bank=json.loads(data.decode("ascii"), object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except SeedBankError: raise
    except Exception: raise SeedBankError("invalid bank JSON") from None
    if type(bank) is not dict or set(bank) != {"exact_vectors","format_version","purpose","rng","statistical_vectors"} or (json.dumps(bank,sort_keys=True,separators=(",",":"),ensure_ascii=True)+"\n").encode("ascii") != data: raise SeedBankError("invalid canonical bank")
    if bank["format_version"] != "1" or bank["purpose"] != "Phase 3 checkpoint D generated, independently verified, hash-bound deterministic CPU seed-bank verification" or bank["rng"] != {"algorithm_id":"poker-knight-ng/philox4x32-10","algorithm_version":"1"}: raise SeedBankError("invalid metadata")
    expected_ids = (tuple(x[0] for x in EXACT_SPECS), tuple(x[0] for x in STAT_SPECS))
    if (type(bank["exact_vectors"]) is not list or type(bank["statistical_vectors"]) is not list or tuple(r.get("id") if type(r) is dict else None for r in bank["exact_vectors"]) != expected_ids[0] or tuple(r.get("id") if type(r) is dict else None for r in bank["statistical_vectors"]) != expected_ids[1]): raise SeedBankError("invalid vector IDs/order")
    rows=bank["exact_vectors"]+bank["statistical_vectors"]
    for index, row in enumerate(rows):
        stat = index >= len(bank["exact_vectors"])
        required={"id","seed","hero_card_ids","board_card_ids","opponent_count","requested_trials","canonical_case_hash_hex","expected"} | ({"canonical_case_bytes_hex"} if not stat else {"confidence","estimands","bounded_mean_equity"})
        if type(row) is not dict or set(row) != required or type(row["id"]) is not str or type(row["seed"]) is not str or re.fullmatch(r"0x[0-9a-f]{16}",row["seed"]) is None: raise SeedBankError("invalid vector schema")
        if type(row["hero_card_ids"]) is not list or type(row["board_card_ids"]) is not list or any(type(x) is not int or not 0 <= x < 52 for x in row["hero_card_ids"]+row["board_card_ids"]) or len(row["hero_card_ids"]) != 2 or len(row["board_card_ids"]) not in (0,3,4,5) or row["hero_card_ids"] != sorted(row["hero_card_ids"]) or row["board_card_ids"] != sorted(row["board_card_ids"]) or len(set(row["hero_card_ids"]+row["board_card_ids"])) != 2+len(row["board_card_ids"]) or type(row["opponent_count"]) is not int or not 1 <= row["opponent_count"] <= 6: raise SeedBankError("invalid vector topology")
        _uint(row["requested_trials"], "requested trials", positive=True)
        canonical=_canonical(tuple(row["hero_card_ids"]),tuple(row["board_card_ids"]),row["opponent_count"])
        if re.fullmatch(r"[0-9a-f]{64}",row["canonical_case_hash_hex"]) is None or row["canonical_case_hash_hex"] != hashlib.sha256(canonical).hexdigest() or (not stat and (re.fullmatch(r"[0-9a-f]+",row["canonical_case_bytes_hex"]) is None or row["canonical_case_bytes_hex"] != canonical.hex())): raise SeedBankError("invalid canonical binding")
        expected=row.get("expected")
        if type(expected) is not dict or set(expected) != {"completed_trials","unique_wins","tie_by_other_winners","losses","equity_share_units","hero_category_counts","rejection_count"} or type(expected["tie_by_other_winners"]) is not dict or set(expected["tie_by_other_winners"]) != set(str(i) for i in range(1,7)) or type(expected["hero_category_counts"]) is not dict or set(expected["hero_category_counts"]) != set(CATEGORY_NAMES): raise SeedBankError("invalid counter map")
        for x in [expected[k] for k in ("completed_trials","unique_wins","losses","equity_share_units","rejection_count")]+list(expected["tie_by_other_winners"].values())+list(expected["hero_category_counts"].values()):
            _uint(x, "counter")
        n=_uint(expected["completed_trials"], "completed trials"); requested=_uint(row["requested_trials"],"requested trials",positive=True)
        if n != requested or _uint(expected["unique_wins"],"wins") + sum(_uint(x,"ties") for x in expected["tie_by_other_winners"].values()) + _uint(expected["losses"],"losses") != n or sum(_uint(x,"category") for x in expected["hero_category_counts"].values()) != n or any(_uint(expected["tie_by_other_winners"][str(i)],"ties") for i in range(row["opponent_count"] + 1,7)): raise SeedBankError("invalid counter conservation")
        equity=420*_uint(expected["unique_wins"],"wins")+sum((420//(i+1))*_uint(expected["tie_by_other_winners"][str(i)],"ties") for i in range(1,7))
        if _uint(expected["equity_share_units"],"equity") != equity: raise SeedBankError("invalid equity")
        if stat:
            if row["confidence"] != CONFIDENCE or type(row["estimands"]) is not dict or set(row["estimands"]) != {"unique_win","tie","loss"}: raise SeedBankError("invalid statistical metadata")
            denoms=[]; nums=[]
            for name in ("unique_win","tie","loss"):
                item=row["estimands"][name]
                if type(item) is not dict or set(item)!={"numerator","denominator"}: raise SeedBankError("invalid estimand")
                num=_uint(item["numerator"],"estimand numerator"); den=_uint(item["denominator"],"estimand denominator",positive=True)
                if num>den: raise SeedBankError("invalid estimand bounds")
                nums.append(num); denoms.append(den)
            b=row["bounded_mean_equity"]
            if type(b) is not dict or set(b)!={"method","two_sided_alpha","range","formula","population_exact_units","population_N"} or b["method"]!="Hoeffding bounded mean" or b["two_sided_alpha"]!="0.000001" or b["range"] != ["0","1"] or b["formula"]!="sqrt(log(2/alpha)/(2*N))" or len(set(denoms)) != 1 or sum(nums)!=denoms[0] or _uint(b["population_exact_units"],"population units") > 420*_uint(b["population_N"],"population N",positive=True): raise SeedBankError("invalid bounded mean")
    return bank
def _parse_manifest(data):
    names=(BANK_PATH.as_posix(),)+AUTHORITY_NAMES
    if not data or not data.endswith(b"\n") or b"\r" in data: raise SeedBankError("invalid manifest")
    try: lines=data.decode("ascii").splitlines()
    except UnicodeDecodeError: raise SeedBankError("invalid manifest ASCII") from None
    if len(lines)!=len(names): raise SeedBankError("invalid manifest entries")
    out=[]
    for line,name in zip(lines,names):
        if re.fullmatch(r"[0-9a-f]{64}  "+re.escape(name),line,re.ASCII) is None: raise SeedBankError("invalid manifest ordering")
        out.append((line[:64],name))
    return out
def _verify_statistics(row: dict[str, Any], result: dict[str, Any], population: Any) -> None:
    """Bind preregistered estimands to exact population before interval use."""
    expected_estimands = {
        "unique_win": population.unique_wins,
        "tie": sum(population.tie_by_other_winners),
        "loss": population.losses,
    }
    bounded = row["bounded_mean_equity"]
    if (bounded["population_N"] != str(population.completed_trials)
            or bounded["population_exact_units"] != str(population.equity_share_units)
            or any(row["estimands"][name]["numerator"] != str(value)
                   or row["estimands"][name]["denominator"] != str(population.completed_trials)
                   for name, value in expected_estimands.items())):
        raise SeedBankError("statistical population metadata mismatch")
    observed={"unique_win":int(result["unique_wins"]),"tie":sum(map(int,result["tie_by_other_winners"].values())),"loss":int(result["losses"])}
    observed_total=int(result["completed_trials"])
    for name, estimand in row["estimands"].items():
        lower, upper = _wilson_interval(observed[name], observed_total, Decimal(row["confidence"]["z"]))
        population_proportion=Decimal(estimand["numerator"])/Decimal(estimand["denominator"])
        if not lower <= population_proportion <= upper: raise SeedBankError(f"Wilson {name} interval failed")
    alpha=Decimal(row["bounded_mean_equity"]["two_sided_alpha"]); n=Decimal(result["completed_trials"]); radius=((Decimal(2)/alpha).ln()/(2*n)).sqrt(); observed_equity=Decimal(result["equity_share_units"])/(420*n); exact=Decimal(population.equity_share_units)/(420*Decimal(population.completed_trials))
    if abs(observed_equity-exact)>radius: raise SeedBankError("bounded mean equity interval failed")


def verify_bundle(root: Path, data: bytes, manifest: bytes) -> None:
    """Strictly verify supplied artifact bytes against authorities rooted at root."""
    bank=_parse_bank(data)
    for digest,name in _parse_manifest(manifest):
        contents = data if name == BANK_PATH.as_posix() else (root/name).read_bytes()
        if hashlib.sha256(contents).hexdigest()!=digest: raise SeedBankError("manifest hash mismatch")
    generated=build(root)
    if data != generated[BANK_NAME] or manifest != generated[MANIFEST_NAME]: raise SeedBankError("bank differs from frozen generated authority")
    replayed = []
    for row in bank["exact_vectors"]+bank["statistical_vectors"]:
        result=run_cpu_monte_carlo(seed=int(row["seed"],16),hero_card_ids=tuple(row["hero_card_ids"]),board_card_ids=tuple(row["board_card_ids"]),opponent_count=row["opponent_count"],requested_trials=int(row["requested_trials"]),replay_case_hash=bytes.fromhex(row["canonical_case_hash_hex"]))
        if _wire_result(_result_fields(result))!=row["expected"]: raise SeedBankError("production replay mismatch")
        replayed.append(_wire_result(_result_fields(result)))
    pop=enumerate_unknown_opponent(("As","Kd"),("2s","7h","9d","Tc","Jc"))
    for row, result in zip(bank["statistical_vectors"], replayed[len(bank["exact_vectors"]):]): _verify_statistics(row, result, pop)


def verify(root: Path = ROOT) -> None:
    verify_bundle(root, (root/BANK_PATH).read_bytes(), (root/MANIFEST_PATH).read_bytes())

def _wilson_interval(successes: int, total: int, z: Decimal) -> tuple[Decimal, Decimal]:
    getcontext().prec = 60
    n=Decimal(total); p=Decimal(successes)/n; denominator=Decimal(1)+z*z/n
    center=(p+z*z/(2*n))/denominator
    radius=z*(p*(Decimal(1)-p)/n+z*z/(4*n*n)).sqrt()/denominator
    return center-radius, center+radius

def _stage_bytes(directory: Path, data: bytes, suffix: str = ".tmp", mode: int = 0o644) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=".rng-seed-bank-", suffix=suffix, dir=directory)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), mode); stream.write(data); stream.flush(); os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True); raise
    try: _fsync_parents([path])
    except Exception as error:
        cleanup=_cleanup([path])
        detail="; ".join(f"{p}: {kind}" for p,kind in cleanup[:4])
        raise SeedBankError(f"stage directory sync failed for {path.parent}: {type(error).__name__}"+(f"; cleanup {detail}" if detail else "")) from error
    return path

def _write_recovery_bytes(destination: Path, recovery: Path, mode: int) -> None:
    # Read the durable authority before opening destination: a failed read must
    # never turn an otherwise recoverable published file into an empty file.
    data = recovery.read_bytes()
    with destination.open("wb") as stream:
        os.fchmod(stream.fileno(), mode); stream.write(data); stream.flush(); os.fsync(stream.fileno())

def _cleanup(paths: list[Path]) -> list[tuple[Path,str]]:
    failures=[]; removed=[]
    for path in sorted(set(paths), key=str):
        try:
            if path.exists(): path.unlink(); removed.append(path)
        except Exception as error: failures.append((path,type(error).__name__))
    try: _fsync_parents(removed)
    except Exception as error:
        for parent in sorted({path.parent for path in removed},key=str): failures.append((parent,type(error).__name__))
    return failures

def _fsync_parent(parent: Path) -> None:
    descriptor=os.open(parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)

def _fsync_parents(paths: list[Path]) -> None:
    for parent in sorted({path.parent for path in paths}, key=str): _fsync_parent(parent)


def _cleanup_recovery(recovery: Path, incomplete: list[str], retained: set[Path]) -> None:
    """Clean recovery only while another durable authority remains available."""
    try: cleanup_authority = _stage_bytes(recovery.parent, recovery.read_bytes(), ".recovery.bak", recovery.stat().st_mode & 0o777)
    except Exception as error:
        retained.add(recovery); incomplete.append(f"retain recovery {recovery}: {type(error).__name__}"); return
    failures = _cleanup([recovery])
    if failures:
        # A failed post-unlink directory sync means recovery's disappearance is
        # uncertain, so retain the separately staged authority for discovery.
        retained.add(cleanup_authority)
        incomplete.extend(f"cleanup recovery {path}: {kind}" for path,kind in failures)
        return
    failures = _cleanup([cleanup_authority])
    if failures:
        # Keep this independently durable, exact-byte authority out of final
        # cleanup: an unlink/fsync failure makes its removal uncertain.
        retained.add(cleanup_authority)
        incomplete.extend(f"cleanup recovery authority {path}: {kind}" for path,kind in failures)


def atomic_write_paths(paths: dict[Path, bytes]) -> None:
    staged: dict[Path, Path] = {}; backups: dict[Path, Path | None] = {}; modes={}; replaced: list[Path] = []; ordered=tuple(sorted(paths,key=str))
    try:
        for path in ordered:
            data=paths[path]
            path.parent.mkdir(parents=True, exist_ok=True); mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
            modes[path]=mode; backups[path] = _stage_bytes(path.parent, path.read_bytes(), ".bak", mode) if path.exists() else None
            staged[path] = _stage_bytes(path.parent, data, mode=mode)
        for path in ordered:
            os.replace(staged[path], path); replaced.append(path); del staged[path]; _fsync_parents([path])
    except Exception as publication_error:
        incomplete=[]; retained=set()
        for path in reversed(replaced):
            backup = backups[path]
            if backup is None:
                try:
                    if path.exists(): path.unlink(); _fsync_parents([path])
                except Exception as error: incomplete.append(f"remove {path}: {type(error).__name__}")
            else:
                recovery=None
                try: recovery=_stage_bytes(path.parent, backup.read_bytes(), ".recovery.bak", modes[path])
                except Exception as error:
                    retained.add(backup); incomplete.append(f"preserve recovery backup {backup}: {type(error).__name__}"); continue
                try:
                    # recovery is the durable authority before this replace
                    # consumes backup. Do not clear either bookkeeping entry
                    # until the restored name is directory-durable.
                    os.replace(backup,path); _fsync_parents([path])
                except Exception as restore_error:
                    # If replace succeeded but directory fsync failed, backup
                    # has been consumed: preserve recovery untouched and never
                    # try to read the moved pathname.
                    if not backup.exists():
                        retained.add(recovery); incomplete.append(f"restore durability {path}: {type(restore_error).__name__}"); continue
                    try: _write_recovery_bytes(path,recovery,modes[path])
                    except Exception as fallback_error: retained.add(backup); incomplete.append(f"restore {path} from {backup}: {type(restore_error).__name__}/{type(fallback_error).__name__}")
                    else:
                        try: _fsync_parents([path])
                        except Exception as error:
                            retained.update((backup,recovery)); incomplete.append(f"sync recovery {path}: {type(error).__name__}")
                        else:
                            backups[path]=None
                            cleanup=_cleanup([backup])
                            incomplete.extend(f"cleanup {p}: {k}" for p,k in cleanup)
                            if cleanup:
                                # The backup's unlink may not be durable; do
                                # not consume the only known-good authority.
                                retained.add(recovery)
                            else:
                                _cleanup_recovery(recovery, incomplete, retained)
                else:
                    backups[path]=None
                    _cleanup_recovery(recovery, incomplete, retained)
        incomplete.extend(f"cleanup {p}: {k}" for p,k in _cleanup(list(staged.values())+[b for b in backups.values() if b is not None and b.exists() and b not in retained]))
        if incomplete: raise SeedBankError("seed-bank publication rollback incomplete: "+"; ".join(incomplete[:8])) from publication_error
        raise SeedBankError("seed-bank publication failed") from None
    failures=_cleanup([backup for backup in backups.values() if backup is not None])
    if failures: raise SeedBankError("seed-bank publication cleanup failed: "+"; ".join(f"{p}: {k}" for p,k in failures[:8]))


def release(root: Path = ROOT) -> None:
    outputs = build(root)
    # Verify the complete prospective bundle before changing either live output.
    verify_bundle(root, outputs[BANK_NAME], outputs[MANIFEST_NAME])
    atomic_write_paths({root / BANK_PATH: outputs[BANK_NAME], root / MANIFEST_PATH: outputs[MANIFEST_NAME]})
    verify(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--verify", action="store_true"); parser.add_argument("--release", action="store_true")
    options = parser.parse_args()
    if options.release: release()
    elif options.verify: verify()
    else: parser.error("choose --verify or --release")
