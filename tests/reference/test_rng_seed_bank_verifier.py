"""Hostile strict-verifier and interval-gate proof matrix."""
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
TOOL = ROOT / "tools/generate_rng_seed_bank.py"


def _g():
    spec = importlib.util.spec_from_file_location("seed_bank_verifier", TOOL)
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _signed(g, bank, root=ROOT):
    data = (json.dumps(bank, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    entries = [(hashlib.sha256(data).hexdigest(), g.BANK_PATH.as_posix())]
    entries += [(hashlib.sha256((root / name).read_bytes()).hexdigest(), name) for name in g.AUTHORITY_NAMES]
    entries.sort(key=lambda item: item[1])
    return data, "".join(f"{d}  {n}\n" for d, n in entries).encode("ascii")


def _bank(g):
    return json.loads((ROOT / g.BANK_PATH).read_text("ascii"))


def test_committed_manifest_paths_are_lexically_sorted():
    g = _g()
    manifest = (ROOT / g.MANIFEST_PATH).read_text("ascii").splitlines()
    paths = [line[66:] for line in manifest]
    assert paths == sorted(paths)


def _semantic_failure(g, monkeypatch, mutate):
    bank = _bank(g); mutate(bank)
    data, manifest = _signed(g, bank)
    monkeypatch.setattr(g, "build", lambda root: {g.BANK_NAME: data, g.MANIFEST_NAME: manifest})
    with pytest.raises(g.SeedBankError): g.verify_bundle(ROOT, data, manifest)


@pytest.mark.parametrize("mutate", [
    lambda b: b["exact_vectors"][0].__setitem__("unexpected", 1),
    lambda b: b["exact_vectors"][0]["expected"].__setitem__("unexpected", "0"),
    lambda b: b["statistical_vectors"][0]["confidence"].__setitem__("unexpected", "x"),
    lambda b: b["statistical_vectors"][0]["estimands"].pop("tie"),
    lambda b: b["statistical_vectors"][0]["bounded_mean_equity"].pop("formula"),
    lambda b: b["exact_vectors"][0].__setitem__("hero_card_ids", [True, 25]),
    lambda b: b["exact_vectors"][0].__setitem__("opponent_count", True),
    lambda b: b["exact_vectors"][0].__setitem__("hero_card_ids", [25, 12]),
    lambda b: b["exact_vectors"][0].__setitem__("board_card_ids", [0, 0, 34]),
    lambda b: b["exact_vectors"][0].__setitem__("board_card_ids", [0, 14, 52]),
    lambda b: b["exact_vectors"][0].__setitem__("requested_trials", "064"),
    lambda b: b["exact_vectors"][0].__setitem__("requested_trials", str(1 << 64)),
    lambda b: b["exact_vectors"][0].__setitem__("seed", "0x000000000000000A"),
    lambda b: b["exact_vectors"].__setitem__(0, b["exact_vectors"][1]),
    lambda b: b["exact_vectors"][0].__setitem__("canonical_case_bytes_hex", "00"),
    lambda b: b["exact_vectors"][0].__setitem__("canonical_case_hash_hex", "0" * 64),
    lambda b: b["exact_vectors"][0]["expected"].__setitem__("completed_trials", "63"),
    lambda b: b["exact_vectors"][0]["expected"]["hero_category_counts"].__setitem__("high_card", "1"),
    lambda b: b["exact_vectors"][0]["expected"]["tie_by_other_winners"].__setitem__("6", "1"),
    lambda b: b["exact_vectors"][0]["expected"].__setitem__("equity_share_units", "1"),
    lambda b: b["statistical_vectors"][0].__setitem__("confidence", {"method":"Wilson score interval","two_sided_alpha":"0.1","z":"4.891638475698591"}),
    lambda b: b["statistical_vectors"][0]["estimands"]["loss"].__setitem__("numerator", "1"),
    lambda b: b["statistical_vectors"][0]["bounded_mean_equity"].__setitem__("method", "bad"),
    lambda b: b["statistical_vectors"][0]["bounded_mean_equity"].__setitem__("population_N", "0"),
], ids=["unknown-row", "unknown-expected", "unknown-confidence", "missing-estimand", "missing-bounded", "bool-card", "bool-opponents", "unsorted", "duplicate-cards", "out-of-range", "leading-zero", "uint64-overflow", "seed", "wrong-order", "canonical-bytes", "canonical-hash", "wtl", "category-conservation", "impossible-tie", "equity", "confidence", "estimand", "bounded-method", "bounded-population"])
def test_resigned_semantic_mutations_are_rejected(monkeypatch, mutate):
    _semantic_failure(_g(), monkeypatch, mutate)


def _population_mutation(field):
    def mutate(bank):
        row = bank["statistical_vectors"][0]
        estimands = row["estimands"]
        if field == "population_exact_units":
            row["bounded_mean_equity"][field] = "0"
        elif field == "population_N":
            row["bounded_mean_equity"][field] = str(int(row["bounded_mean_equity"][field]) - 1)
        elif field == "unique_win numerator":
            estimands["unique_win"]["numerator"] = str(int(estimands["unique_win"]["numerator"]) - 1)
            estimands["loss"]["numerator"] = str(int(estimands["loss"]["numerator"]) + 1)
        elif field == "tie numerator":
            estimands["tie"]["numerator"] = str(int(estimands["tie"]["numerator"]) - 1)
            estimands["loss"]["numerator"] = str(int(estimands["loss"]["numerator"]) + 1)
        elif field == "loss numerator":
            estimands["loss"]["numerator"] = str(int(estimands["loss"]["numerator"]) - 1)
            estimands["unique_win"]["numerator"] = str(int(estimands["unique_win"]["numerator"]) + 1)
        elif field == "estimand denominator":
            for value in estimands.values(): value["denominator"] = str(int(value["denominator"]) - 1)
            estimands["loss"]["numerator"] = str(int(estimands["loss"]["numerator"]) - 1)
        else: raise AssertionError(field)
    return mutate


@pytest.mark.parametrize("field", ["population_exact_units", "population_N", "unique_win numerator", "tie numerator", "loss numerator", "estimand denominator"])
def test_resigned_population_metadata_must_match_independent_enumeration(monkeypatch, field):
    _semantic_failure(_g(), monkeypatch, _population_mutation(field))


@pytest.mark.parametrize("data", [
    b'{"exact_vectors":[],"exact_vectors":[]}\n', b'\xef\xbb\xbf{}\n', b'{}\r\n', b'{}', b'{}\n\n', b'{"x":NaN}\n',
], ids=["duplicate-key", "bom", "crlf", "missing-lf", "trailing-content", "nonfinite"])
def test_hostile_bank_wire_is_rejected(data):
    g = _g()
    with pytest.raises(g.SeedBankError): g.verify_bundle(ROOT, data, b"")


@pytest.mark.parametrize("mutate", [
    lambda m: b"x" + m[1:], lambda m: m.replace(b"  ", b" ", 1), lambda m: m.upper(),
    lambda m: b"00" * 32 + b"  unknown\n" + m, lambda m: m.replace(b"\n", b"\n" + m.splitlines()[0] + b"\n"), lambda m: b"\n".join(reversed(m.splitlines())) + b"\n", lambda m: m.replace(b"\n", b"\r\n"), lambda m: m[:-1],
], ids=["non-ascii", "spacing", "uppercase", "unknown-extra", "duplicate", "out-of-order", "crlf", "missing-lf"])
def test_hostile_manifest_wire_is_rejected(monkeypatch, mutate):
    g = _g(); bank = _bank(g); data, manifest = _signed(g, bank)
    with pytest.raises(g.SeedBankError): g.verify_bundle(ROOT, data, mutate(manifest))


def test_bad_authority_digest_is_rejected():
    g = _g(); data, manifest = _signed(g, _bank(g)); lines = manifest.splitlines(); lines[1] = b"0" * 64 + lines[1][64:]
    with pytest.raises(g.SeedBankError, match="manifest hash mismatch"): g.verify_bundle(ROOT, data, b"\n".join(lines)+b"\n")


@pytest.mark.parametrize("event", ["unique_win", "tie", "loss"])
def test_verify_invokes_event_specific_wilson_gate(monkeypatch, event):
    g = _g(); real = g._verify_statistics
    def fail(row, result, population):
        broken = deepcopy(result)
        if event == "tie":
            broken["tie_by_other_winners"] = {"1": broken["completed_trials"], **{str(i): "0" for i in range(2, 7)}}
        else:
            broken[{"unique_win":"unique_wins", "loss":"losses"}[event]] = "0"
        return real(row, broken, population)
    monkeypatch.setattr(g, "_verify_statistics", fail)
    with pytest.raises(g.SeedBankError, match=f"Wilson {event} interval failed"): g.verify(ROOT)


def test_verify_invokes_bounded_mean_gate(monkeypatch):
    g = _g(); real = g._verify_statistics
    def fail(row, result, population):
        broken = deepcopy(result); broken["equity_share_units"] = "0"; return real(row, broken, population)
    monkeypatch.setattr(g, "_verify_statistics", fail)
    with pytest.raises(g.SeedBankError, match="bounded mean equity interval failed"): g.verify(ROOT)


def test_statistics_pass_and_reports_wilson_endpoints():
    g = _g(); bank = _bank(g); row = bank["statistical_vectors"][0]; g.verify(ROOT)
    expected = row["expected"]
    observed = {
        "unique_win": int(expected["unique_wins"]),
        "tie": sum(map(int, expected["tie_by_other_winners"].values())),
        "loss": int(expected["losses"]),
    }
    total = int(expected["completed_trials"])
    z = __import__("decimal").Decimal(row["confidence"]["z"])
    for name, estimand in row["estimands"].items():
        lo, hi = g._wilson_interval(observed[name], total, z)
        exact = __import__("decimal").Decimal(estimand["numerator"]) / __import__("decimal").Decimal(estimand["denominator"])
        assert lo < hi and lo <= exact <= hi


def test_wilson_interval_is_built_from_observation_and_must_cover_population() -> None:
    g = _g()
    row = {
        "confidence": {"z": "4.891638475698591"},
        "estimands": {
            "unique_win": {"numerator": "500", "denominator": "1000"},
            "tie": {"numerator": "0", "denominator": "1000"},
            "loss": {"numerator": "500", "denominator": "1000"},
        },
        "bounded_mean_equity": {
            "population_N": "1000",
            "population_exact_units": "210000",
            "two_sided_alpha": "0.000001",
        },
    }
    result = {
        "completed_trials": "2000",
        "unique_wins": "1140",
        "tie_by_other_winners": {str(index): "0" for index in range(1, 7)},
        "losses": "860",
        "equity_share_units": "420000",
    }
    population = SimpleNamespace(
        completed_trials=1000,
        unique_wins=500,
        tie_by_other_winners=(0, 0, 0, 0, 0, 0),
        losses=500,
        equity_share_units=210000,
    )
    with pytest.raises(g.SeedBankError, match="Wilson unique_win interval failed"):
        g._verify_statistics(row, result, population)
