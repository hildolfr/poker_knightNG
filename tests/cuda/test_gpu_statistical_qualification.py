"""CPU-only core tests for the Phase 5C CUDA statistical qualifier."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
TOOL = ROOT / "tools/qualify_gpu_statistics.py"
BANK = ROOT / "validation/holdem/v1/rng_seed_bank.json"


def _tool():
    spec = importlib.util.spec_from_file_location("qualify_gpu_statistics", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_is_cupy_inert_and_geometries_are_frozen() -> None:
    before = set(sys.modules)
    tool = _tool()
    assert "cupy" not in set(sys.modules) - before
    assert tool.GEOMETRIES == {
        "capacity_1": {"batch_blocks": 1, "vram_budget_bytes": None, "capacity": 1},
        "budget_capacity_3": {"batch_blocks": 256, "vram_budget_bytes": 4864, "capacity": 3},
        "capacity_7": {"batch_blocks": 7, "vram_budget_bytes": None, "capacity": 7},
        "capacity_256": {"batch_blocks": 256, "vram_budget_bytes": None, "capacity": 256},
    }
    assert tool.expected_batch_plan(2000, 1) == [(i, i * 128, 128, 1) for i in range(15)] + [(15, 1920, 80, 1)]
    assert tool.expected_batch_plan(2000, 3) == [
        (0, 0, 384, 3), (1, 384, 384, 3), (2, 768, 384, 3),
        (3, 1152, 384, 3), (4, 1536, 384, 3), (5, 1920, 80, 1),
    ]
    assert tool.expected_batch_plan(2000, 7) == [(0, 0, 896, 7), (1, 896, 896, 7), (2, 1792, 208, 2)]
    assert tool.expected_batch_plan(2000, 256) == [(0, 0, 2000, 16)]


def test_exactly_one_frozen_statistical_case_is_selected() -> None:
    tool = _tool(); bank = json.loads(BANK.read_text("ascii"))
    row = tool.select_statistical_case(bank)
    assert row["id"] == "river-high-card-wtl-2000"
    mutations = []
    extra = deepcopy(bank); extra["statistical_vectors"].append(deepcopy(row)); mutations.append(extra)
    alternate = deepcopy(bank); alternate["statistical_vectors"][0]["seed"] = "0x0000000000000002"; mutations.append(alternate)
    trials = deepcopy(bank); trials["statistical_vectors"][0]["requested_trials"] = "1999"; mutations.append(trials)
    expected = deepcopy(bank); expected["statistical_vectors"][0]["expected"]["rejection_count"] = "1"; mutations.append(expected)
    population = deepcopy(bank); population["statistical_vectors"][0]["estimands"]["loss"]["numerator"] = "712"; mutations.append(population)
    for mutated in mutations:
        with pytest.raises(tool.QualificationError):
            tool.select_statistical_case(mutated)


def _trace(plan: list[tuple[int, int, int, int]]) -> list[dict[str, object]]:
    events = []
    for _ordinal, offset, trials, blocks in plan:
        events.extend([
            {"kernel": "simulate", "grid": (blocks,), "block": (128,), "first_simulation_id": offset, "trials": trials},
            {"kernel": "reduce", "grid": (1,), "block": (128,), "partial_count": blocks},
        ])
    return events


def test_actual_kernel_trace_must_equal_each_fixed_plan() -> None:
    tool = _tool()
    for name, config in tool.GEOMETRIES.items():
        plan = tool.expected_batch_plan(2000, config["capacity"])
        batches = tool.validate_launch_trace(name, _trace(plan))
        assert len(batches) == len(plan)
        assert batches[-1]["first_simulation_id"] == str(plan[-1][1])
        assert batches[-1]["partial_blocks"] == str(plan[-1][3])
    bad = _trace(tool.expected_batch_plan(2000, 3)); bad[-1]["partial_count"] = 2
    with pytest.raises(tool.QualificationError, match="GEOMETRY"):
        tool.validate_launch_trace("budget_capacity_3", bad)


def test_statistical_report_uses_observation_and_rejects_each_gate() -> None:
    tool = _tool(); row = tool.select_statistical_case(json.loads(BANK.read_text("ascii")))
    report = tool.statistical_report(row, row["expected"])
    assert set(report["wilson"]) == {"unique_win", "tie", "loss"}
    assert all(gate["status"] == "passed" for gate in report["wilson"].values())
    assert report["hoeffding"]["status"] == "passed"
    for field in ("unique_wins", "losses", "equity_share_units"):
        bad = deepcopy(row["expected"]); bad[field] = "0"
        with pytest.raises(tool.QualificationError, match="STATISTICS"):
            tool.statistical_report(row, bad)
    bad = deepcopy(row["expected"])
    bad["tie_by_other_winners"] = {str(i): "2000" if i == 1 else "0" for i in range(1, 7)}
    with pytest.raises(tool.QualificationError, match="STATISTICS"):
        tool.statistical_report(row, bad)


def test_raw_aggregate_category_order_is_semantic() -> None:
    tool = _tool()
    raw = SimpleNamespace(
        completed_trials=9, unique_wins=9, tie_by_other_winners=(0, 0, 0, 0, 0, 0),
        losses=0, equity_share_units=3780, hero_category_counts=tuple(range(9)),
        rejection_count=0,
    )
    mapped = tool.aggregate_wire(raw)["hero_category_counts"]
    assert mapped == {
        "high_card": "0", "one_pair": "1", "two_pair": "2",
        "three_of_a_kind": "3", "straight": "4", "flush": "5",
        "full_house": "6", "four_of_a_kind": "7", "straight_flush": "8",
    }


def test_parent_runs_full_seed_verifier_before_gpu_admission(tmp_path: Path) -> None:
    tool = _tool(); calls = []
    manifest = tmp_path / "rng_seed_bank.sha256"; manifest.write_text("manifest")
    bank = tmp_path / "rng_seed_bank.json"; bank.write_text("bank")

    class Base:
        def checkout_identity(self, root, target, branch):
            calls.append("checkout"); return target, branch
        def _run(self, argv, *, cwd, code):
            calls.append(("run", tuple(argv), code)); return SimpleNamespace(stdout="", stderr="")
        def verify_seed_manifest(self, root):
            calls.append("manifest"); return manifest, bank
        def gpu_admission(self, root):
            calls.append("admission"); return {"compute_capability": "12.0"}

    result = tool.run_pre_admission(tmp_path, "0" * 40, tool.EXPECTED_BRANCH, Base())
    assert result == (manifest, bank, {"compute_capability": "12.0"})
    assert calls[0] == "checkout"
    assert calls[1] == (
        "run", (sys.executable, str(tmp_path / "tools/generate_rng_seed_bank.py"), "--verify"), "VERIFY",
    )
    assert calls[2:] == ["manifest", "admission"]


def test_seed_verifier_failure_or_interruption_never_reaches_gpu(tmp_path: Path) -> None:
    tool = _tool()
    for failure in (tool.QualificationError("VERIFY"), KeyboardInterrupt()):
        calls = []
        class Base:
            def checkout_identity(self, root, target, branch): return target, branch
            def _run(self, argv, *, cwd, code): calls.append("verify"); raise failure
            def verify_seed_manifest(self, root): calls.append("manifest")
            def gpu_admission(self, root): calls.append("admission")
        with pytest.raises(type(failure)):
            tool.run_pre_admission(tmp_path, "0" * 40, tool.EXPECTED_BRANCH, Base())
        assert calls == ["verify"]


def test_parent_revalidates_every_worker_claim() -> None:
    tool = _tool(); row = tool.select_statistical_case(json.loads(BANK.read_text("ascii")))
    aggregate = deepcopy(row["expected"])
    geometries = {}
    for name, config in tool.GEOMETRIES.items():
        geometries[name] = {
            "config": {
                "batch_blocks": str(config["batch_blocks"]),
                "vram_budget_bytes": "default" if config["vram_budget_bytes"] is None else str(config["vram_budget_bytes"]),
            },
            "actual_capacity": str(config["capacity"]), "batches": tool.expected_batches_wire(name),
            "cpu_aggregate": aggregate, "cuda_aggregate": aggregate, "frozen_aggregate": aggregate,
            "statistics": tool.statistical_report(row, aggregate), "duration_ns": "1",
        }
    worker = {
        "statistical_case": {
            "id": row["id"], "seed": row["seed"], "requested_trials": row["requested_trials"],
            "hero_card_ids": ["12", "37"], "board_card_ids": ["0", "18", "33", "47", "48"],
            "opponent_count": "1", "canonical_case_hash_hex": row["canonical_case_hash_hex"],
        },
        "geometries": geometries,
        "provenance": {"qualification": "cuda-statistical-v1", "device_id": "cuda-uuid:" + "a" * 32, "kernel_id": "cuda-source-sha256:" + "b" * 64},
        "environment": {
            "python_version": "3.13.15", "cupy_version": "14.1.1", "cuda_driver_version": "13000",
            "cuda_runtime_version": "13000", "compute_capability": "12.0",
            "memory_free_before_bytes": "1", "memory_free_after_bytes": "1",
        },
        "installation": {"wheel_basename": "package.whl", "wheel_contents_verified": "true", "wheel_sha256": "c" * 64},
    }
    admission = {"nvidia_uuid": "GPU-" + "a" * 32, "compute_capability": "12.0"}
    wheel = {"basename": "package.whl", "sha256": "c" * 64, "size": "1"}
    assert tool.validate_worker_output(worker, row, admission, wheel, "b" * 64) == worker
    mutations = []
    extra = deepcopy(worker); extra["unexpected"] = "x"; mutations.append(extra)
    trace = deepcopy(worker); trace["geometries"]["budget_capacity_3"]["batches"][0]["trials"] = "1"; mutations.append(trace)
    result = deepcopy(worker); result["geometries"]["capacity_7"]["cuda_aggregate"]["unique_wins"] = "0"; mutations.append(result)
    report = deepcopy(worker); report["geometries"]["capacity_256"]["statistics"]["wilson"]["loss"]["status"] = "failed"; mutations.append(report)
    device = deepcopy(worker); device["provenance"]["device_id"] = "cuda-uuid:" + "d" * 32; mutations.append(device)
    installed = deepcopy(worker); installed["installation"]["wheel_sha256"] = "e" * 64; mutations.append(installed)
    for mutated in mutations:
        with pytest.raises(tool.QualificationError):
            tool.validate_worker_output(mutated, row, admission, wheel, "b" * 64)


def test_worker_is_private_and_parent_is_explicitly_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _tool(); monkeypatch.delenv("PKNG_STATISTICAL_QUALIFICATION_WORKER", raising=False)
    with pytest.raises(tool.QualificationError, match="WORKER"):
        tool._worker(BANK, Path("missing.whl"))
    monkeypatch.delenv("RUN_CUDA_STATISTICAL_QUALIFICATION", raising=False)
    arguments = SimpleNamespace(run_id="test")
    assert tool.qualify(arguments) == 2


def test_private_worker_records_all_actual_launches_with_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _tool(); row = tool.select_statistical_case(json.loads(BANK.read_text("ascii")))
    from poker_knight_ng import _cuda_runtime

    class FakeKernel:
        def __call__(self, grid, block, args): pass

    class FakeRuntime:
        def __init__(self, *, batch_blocks, vram_budget_bytes):
            self.batch_blocks, self.vram_budget_bytes = batch_blocks, vram_budget_bytes
        def _batch_capacity(self):
            return 3 if self.vram_budget_bytes == 4864 else self.batch_blocks
        def _kernels(self): return FakeKernel(), FakeKernel()
        def run(self, *, hero, board, opponents, key, first_simulation_id, count):
            simulate, reduce = self._kernels(); offset = 0
            while offset < count:
                trials = min(count - offset, self._batch_capacity() * 128); blocks = (trials + 127) // 128
                simulate((blocks,), (128,), (None, None, None, None, None, None, first_simulation_id + offset, trials, None))
                reduce((1,), (128,), (None, blocks, None)); offset += trials
            expected = row["expected"]
            categories = expected["hero_category_counts"]
            return SimpleNamespace(
                completed_trials=int(expected["completed_trials"]), unique_wins=int(expected["unique_wins"]),
                tie_by_other_winners=tuple(int(expected["tie_by_other_winners"][str(i)]) for i in range(1, 7)),
                losses=int(expected["losses"]), equity_share_units=int(expected["equity_share_units"]),
                hero_category_counts=tuple(int(categories[name]) for name in tool.CATEGORY_KEYS),
                rejection_count=int(expected["rejection_count"]),
            )
        def provenance(self): return "cuda-uuid:" + "a" * 32, "cuda-source-sha256:" + "b" * 64

    class Runtime:
        @staticmethod
        def getDevice(): return 0
        @staticmethod
        def getDeviceProperties(device): return {"major": 12, "minor": 0}
        @staticmethod
        def memGetInfo(): return 10_000_000, 20_000_000
        @staticmethod
        def driverGetVersion(): return 13000
        @staticmethod
        def runtimeGetVersion(): return 13000

    fake_cupy = SimpleNamespace(
        __version__="14.1.1",
        cuda=SimpleNamespace(runtime=Runtime, Device=lambda: SimpleNamespace(compute_capability="120")),
    )
    class Base:
        MAX_JSON_BYTES = 1_000_000
        @staticmethod
        def read_limited(path, limit, code): return path.read_bytes()
        @staticmethod
        def strict_json(raw): return json.loads(raw)
        @staticmethod
        def _verify_installed_wheel(distribution, wheel):
            return {"wheel_basename": "package.whl", "wheel_contents_verified": "true", "wheel_sha256": "c" * 64}
        @staticmethod
        def _device_uuid(properties): return "cuda-uuid:" + "a" * 32

    monkeypatch.setenv("PKNG_STATISTICAL_QUALIFICATION_WORKER", "1")
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(_cuda_runtime, "CupyDeterministicRuntime", FakeRuntime)
    monkeypatch.setattr(tool, "_base", lambda: Base())
    result = tool._worker(BANK, Path("package.whl"))
    assert set(result["geometries"]) == set(tool.GEOMETRIES)
    for name in tool.GEOMETRIES:
        assert result["geometries"][name]["batches"] == tool.expected_batches_wire(name)
        assert result["geometries"][name]["cuda_aggregate"] == row["expected"]


def test_harness_closure_binds_imported_base_and_machine_contract(tmp_path: Path) -> None:
    tool = _tool()
    assert "tools/qualify_gpu.py" in tool.HARNESS_BINDINGS
    assert "validation/holdem/v1/cuda_statistical_qualification.schema.json" in tool.HARNESS_BINDINGS
    contents = {name: name.encode() for name in tool.HARNESS_BINDINGS}
    class Base:
        MAX_JSON_BYTES = 1_000_000
        @staticmethod
        def read_limited(path, limit, code): return contents[path.relative_to(tmp_path).as_posix()]
    first = tool.harness_source_digest(tmp_path, Base())
    contents["tools/qualify_gpu.py"] += b"tamper"
    second = tool.harness_source_digest(tmp_path, Base())
    assert len(first) == 64 and first != second


def test_offline_verifier_binds_claimed_checkout_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _tool()
    run = tmp_path / "phase5c-test"
    run.mkdir()
    evidence = run / "qualification.json"
    evidence.write_bytes(tool.canonical({"status": "passed"}) + b"\n")
    source = {"git_sha": "a" * 40, "branch": tool.EXPECTED_BRANCH}
    record = {"run_id": run.name, "source": source}
    calls = []

    class Base:
        MAX_JSON_BYTES = 1_000_000

        @staticmethod
        def read_limited(path, limit, code):
            return path.read_bytes()

        @staticmethod
        def strict_json(raw):
            return json.loads(raw)

    def reject_checkout(base, root, claimed):
        calls.append((root, claimed))
        raise tool.QualificationError("SOURCE")

    monkeypatch.setattr(tool, "_base", lambda: Base())
    monkeypatch.setattr(tool, "_passed_shape", lambda value: record)
    monkeypatch.setattr(tool, "verify_checkout_identity", reject_checkout)
    assert tool.verify(evidence, tmp_path) == 1
    assert calls == [(tmp_path.resolve(), source)]
