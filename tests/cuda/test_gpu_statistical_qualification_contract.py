"""Preregistered Phase 5C evidence wire contract."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).parents[2]
SCHEMA = ROOT / "validation/holdem/v1/cuda_statistical_qualification.schema.json"
PREREGISTRATION = ROOT / "validation/holdem/v1/STATISTICAL_QUALIFICATION.md"
TOOL = ROOT / "tools/qualify_gpu_statistics.py"
HEX64 = "0" * 64


def _file(name: str) -> dict[str, str]:
    return {"basename": name, "sha256": HEX64, "size": "1"}


def _aggregate() -> dict[str, object]:
    return {
        "completed_trials": "2000", "unique_wins": "561",
        "tie_by_other_winners": {str(i): "16" if i == 1 else "0" for i in range(1, 7)},
        "losses": "1423", "equity_share_units": "238980",
        "hero_category_counts": {
            "high_card": "2000", "one_pair": "0", "two_pair": "0",
            "three_of_a_kind": "0", "straight": "0", "flush": "0",
            "full_house": "0", "four_of_a_kind": "0", "straight_flush": "0",
        },
        "rejection_count": "0",
    }


def _batch() -> dict[str, object]:
    return {
        "ordinal": "0", "first_simulation_id": "0", "trials": "2000",
        "partial_blocks": "16", "simulate_grid": ["16"],
        "simulate_block": ["128"], "reduce_grid": ["1"],
        "reduce_block": ["128"],
    }


def _statistics() -> dict[str, object]:
    event = {
        "successes": "561", "trials": "2000", "lower": "0.1", "upper": "0.4",
        "population_numerator": "268", "population_denominator": "990", "status": "passed",
    }
    return {
        "wilson": {"unique_win": event, "tie": {**event, "successes": "16"}, "loss": {**event, "successes": "1423"}},
        "hoeffding": {
            "observed_equity": "0.2845", "exact_equity": "0.2752", "radius": "0.1",
            "observed_units": "238980", "trials": "2000",
            "population_exact_units": "114450", "population_N": "990", "status": "passed",
        },
    }


def _geometry(batch_blocks: str, budget: str, capacity: str) -> dict[str, object]:
    aggregate = _aggregate()
    return {
        "config": {"batch_blocks": batch_blocks, "vram_budget_bytes": budget},
        "actual_capacity": capacity, "batches": [_batch()],
        "cpu_aggregate": aggregate, "cuda_aggregate": deepcopy(aggregate),
        "frozen_aggregate": deepcopy(aggregate), "statistics": _statistics(),
        "duration_ns": "1",
    }


def _passed() -> dict[str, object]:
    return {
        "format_version": "1", "run_id": "phase5c-example", "status": "passed", "error_codes": [],
        "source": {
            "git_sha": "0" * 40, "branch": "revival/phase-5c-statistical-validation", "clean": "true",
            "cuda_source_sha256": HEX64, "harness_source_sha256": HEX64,
            "phase5c_tool_sha256": HEX64, "seed_generator_sha256": HEX64,
        },
        "artifacts": {
            "seed_bank": _file("rng_seed_bank.json"),
            "seed_manifest": _file("rng_seed_bank.sha256"),
            "wheel": _file("poker_knight_ng-0.1.0-py3-none-any.whl"),
            "sdist": _file("poker_knight_ng-0.1.0.tar.gz"),
        },
        "gates": {
            "checkout": "passed", "seed_bank_verify": "passed", "admission": "passed",
            "worker": "passed", "geometry": "passed", "exact_aggregate": "passed", "statistics": "passed",
        },
        "statistical_case": {
            "id": "river-high-card-wtl-2000", "seed": "0x0000000000000001",
            "requested_trials": "2000", "hero_card_ids": ["12", "37"],
            "board_card_ids": ["0", "18", "33", "47", "48"], "opponent_count": "1",
            "canonical_case_hash_hex": HEX64,
        },
        "preregistration": {
            "interpretation": "fixed_stream_calibration_not_iid_coverage_guarantee", "per_gate_alpha": "0.000001",
            "wilson_z": "4.891638475698591", "hoeffding_formula": "sqrt(log(2/alpha)/(2*N))",
            "maximum_union_bound": "0.000016",
        },
        "geometries": {
            "capacity_1": _geometry("1", "default", "1"),
            "budget_capacity_3": _geometry("256", "4864", "3"),
            "capacity_7": _geometry("7", "default", "7"),
            "capacity_256": _geometry("256", "default", "256"),
        },
        "provenance": {
            "qualification": "cuda-statistical-v1", "device_id": "cuda-uuid:" + "0" * 32,
            "kernel_id": "cuda-source-sha256:" + HEX64,
        },
        "environment": {
            "python_version": "3.13.15", "cupy_version": "14.1.1",
            "cuda_driver_version": "13000", "cuda_runtime_version": "13020",
            "compute_capability": "12.0", "memory_free_before_bytes": "2147483648",
            "memory_free_after_bytes": "2147483648",
        },
    }


def test_phase5c_schema_and_preregistration_are_closed_and_explicit() -> None:
    schema = json.loads(SCHEMA.read_text("ascii"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(_passed())
    validator.validate({"format_version": "1", "run_id": "phase5c-example", "status": "failed", "error_codes": ["VERIFY"]})

    for mutation in (
        lambda value: value.__setitem__("unexpected", "x"),
        lambda value: value["geometries"].pop("capacity_7"),
        lambda value: value["geometries"]["capacity_1"]["cuda_aggregate"].pop("rejection_count"),
        lambda value: value["geometries"]["capacity_1"]["batches"][0].pop("reduce_block"),
        lambda value: value["preregistration"].__setitem__("interpretation", "frequentist guarantee"),
    ):
        candidate = _passed(); mutation(candidate)
        with pytest.raises(ValidationError):
            validator.validate(candidate)

    text = PREREGISTRATION.read_text("utf-8")
    for phrase in (
        "fixed-stream calibration", "per-gate", "maximum union-bound error `0.000016`",
        "CPU raw aggregate == frozen aggregate", "CUDA raw aggregate == frozen aggregate",
        "generate_rng_seed_bank.py --verify", "4864", "No retry",
    ):
        assert phrase in text


def _tool():
    spec = importlib.util.spec_from_file_location("qualify_gpu_statistics_contract", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_producer_verifier_and_schema_share_preregistration_authority() -> None:
    tool = _tool()
    record = _passed()
    record["preregistration"] = tool.preregistration_record()
    statistical_case = record["statistical_case"]
    assert isinstance(statistical_case, dict)
    statistical_case["canonical_case_hash_hex"] = tool.STATISTICAL_HASH
    assert tool._passed_shape(record) == record

    schema = json.loads(SCHEMA.read_text("ascii"))
    Draft202012Validator(schema).validate(record)
