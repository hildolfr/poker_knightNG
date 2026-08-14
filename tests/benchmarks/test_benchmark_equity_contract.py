from decimal import Decimal
import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOL = ROOT / "tools/benchmark_equity.py"
SCENARIOS = ROOT / "benchmarks/scenarios/v1"


def load_tool():
    spec = importlib.util.spec_from_file_location("benchmark_equity", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preregistration_freezes_matrix_repetitions_and_statistics():
    tool = load_tool()
    record = tool.preregistration_record()
    assert record == {
        "analytical_reference": "separate_process_isolated_cache_public_cuda_repeatability_reference_not_independent_correctness_proof",
        "benchmark_id": "holdem-v1-cuda-baseline-1",
        "cache_modes": ["cold", "warm", "steady"],
        "end_to_end_scope": "one_normal_explicit_solve_cuda_host_monotonic",
        "format_version": "phase6c-benchmark-preregistration-v1",
        "outlier_policy": "none_all_valid_samples_retained",
        "percentile_method": "nearest_rank_ceil_p_times_n_div_100_minus_1",
        "percentiles": ["5", "50", "95"],
        "steady_repetitions": "30",
        "steady_warmups": "1",
        "stage_batch_blocks": "256",
        "stage_threads_per_block": "128",
        "stage_batch_counts_by_requested_trials": {
            "10000": "1", "100000": "4", "500000": "16", "1000000": "31",
        },
        "streets": ["preflop", "flop", "turn", "river"],
        "opponent_counts": ["1", "3", "6"],
        "requested_trials": ["10000", "100000", "500000", "1000000"],
        "startup_canary_cell_id": "v1-preflop-o1-n10000",
        "throughput_scale": "0.001",
    }


def test_canonical_scenarios_expand_to_exact_48_cell_matrix():
    tool = load_tool()
    scenarios = tool.load_scenarios(SCENARIOS)
    assert [scenario["id"] for scenario in scenarios] == [
        f"v1-{street}-o{opponents}"
        for street in ("preflop", "flop", "turn", "river")
        for opponents in ("1", "3", "6")
    ]
    cells = tool.expand_matrix(scenarios)
    assert len(cells) == 48
    assert len({cell["cell_id"] for cell in cells}) == 48
    assert {cell["backend"] for cell in cells} == {"cuda"}
    assert {cell["requested_trials"] for cell in cells} == {
        "10000", "100000", "500000", "1000000"
    }
    assert all(cell["seed"] == "0x0123456789abcdef" for cell in cells)


def test_scenario_manifest_is_closed_relative_and_hash_current():
    tool = load_tool()
    manifest = tool.scenario_manifest(SCENARIOS)
    assert set(manifest) == {"format_version", "scenarios"}
    assert manifest["format_version"] == "phase6c-scenario-manifest-v1"
    assert len(manifest["scenarios"]) == 12
    paths = [row["path"] for row in manifest["scenarios"]]
    assert paths == sorted(paths)
    for row in manifest["scenarios"]:
        assert set(row) == {"id", "path", "sha256", "size_bytes"}
        assert "/" not in row["path"] and "\\" not in row["path"]
        source = SCENARIOS / row["path"]
        assert row["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert row["size_bytes"] == str(source.stat().st_size)
    assert tool.scenario_manifest_sha256(manifest) == hashlib.sha256(
        tool.canonical(manifest)
    ).hexdigest()


def test_nearest_rank_and_throughput_are_exact_decimal_operations():
    tool = load_tool()
    samples = tuple(range(101, 131))
    assert tool.nearest_rank(samples, 5) == 102
    assert tool.nearest_rank(samples, 50) == 115
    assert tool.nearest_rank(samples, 95) == 129
    assert tool.samples_per_second(100_000, 12_345_678) == "8100000.664"
    assert Decimal(tool.samples_per_second(10_000, 1_000_000_000)) == Decimal("10000.000")


@pytest.mark.parametrize("samples,percentile", [
    ((), 50),
    ((1,), 0),
    ((1,), 101),
    ((1, True), 50),
    ((0, 1), 50),
])
def test_nearest_rank_rejects_invalid_samples(samples, percentile):
    tool = load_tool()
    with pytest.raises(tool.BenchmarkError):
        tool.nearest_rank(samples, percentile)


def test_method_identifies_qualification_bound_todo_as_legacy_history():
    todo = (ROOT / "TODO.md").read_text(encoding="utf-8")
    method = (ROOT / "docs/performance-method.md").read_text(encoding="utf-8")
    assert "> **LEGACY — NON-AUTHORITATIVE.**" in todo
    assert "Current Config (256 threads)" in todo
    assert "qualification-bound legacy history" in method
    assert "not a current runtime or performance claim" in method
