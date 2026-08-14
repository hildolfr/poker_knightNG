"""Host-only fake coverage for the Phase 6C live worker."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "benchmark_equity.py"


def load_tool():
    name = f"benchmark_equity_live_test_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _paths(tmp_path):
    scenarios = tmp_path / "scenarios"; scenarios.mkdir()
    wheel = tmp_path / "poker_knight_ng-0.1.0-py3-none-any.whl"; wheel.write_bytes(b"wheel")
    return scenarios, wheel


def test_worker_requires_absolute_nonsymlink_existing_paths(tmp_path, monkeypatch):
    tool = load_tool(); scenarios, wheel = _paths(tmp_path)
    monkeypatch.setenv(tool.WORKER_MARKER, "1")
    monkeypatch.setattr(tool, "load_scenarios", lambda _: ())
    with pytest.raises(tool.BenchmarkError, match="WORKER_PATH"):
        tool.worker_main(("--worker", "--mode", "expected", "--scenario-dir", "relative", "--wheel", str(wheel)))
    link = tmp_path / "link"; link.symlink_to(wheel)
    with pytest.raises(tool.BenchmarkError, match="WORKER_PATH"):
        tool.worker_main(("--worker", "--mode", "expected", "--scenario-dir", str(scenarios), "--wheel", str(link)))


def test_expected_runs_exactly_one_normal_public_solve_per_cell(tmp_path, monkeypatch):
    tool = load_tool(); scenarios, wheel = _paths(tmp_path)
    monkeypatch.setenv(tool.WORKER_MARKER, "1")
    cells = tuple({"cell_id": f"c{n}", "hero_cards": ["As", "Kd"], "board_cards": [], "opponent_count": "1", "requested_trials": "10000", "seed": "0x0000000000000001"} for n in range(48))
    monkeypatch.setattr(tool, "load_scenarios", lambda _: ())
    monkeypatch.setattr(tool, "expand_matrix", lambda _: cells)
    calls = []
    class Request:
        @staticmethod
        def parse(raw): return raw
    monkeypatch.setattr(tool, "_worker_public_api", lambda: (Request, lambda request: calls.append(request) or {"request": request}, lambda result, request: {"timing": {"total_duration_ns": "9"}, "provenance": {"x": "y"}, "answer": request["requested_trials"]}))
    output = tool.worker_main(("--worker", "--mode", "expected", "--scenario-dir", str(scenarios), "--wheel", str(wheel)))
    assert len(calls) == 48
    assert set(output) == {"mode", "analytical_sha256s"}
    assert len(output["analytical_sha256s"]) == 48


def test_steady_has_one_warmup_and_thirty_normal_measurements_without_stage(tmp_path, monkeypatch):
    tool = load_tool(); scenarios, wheel = _paths(tmp_path)
    monkeypatch.setenv(tool.WORKER_MARKER, "1")
    cell = {"cell_id": "c", "hero_cards": ["As", "Kd"], "board_cards": [], "opponent_count": "1", "requested_trials": "10000", "seed": "0x0000000000000001"}
    monkeypatch.setattr(tool, "load_scenarios", lambda _: ())
    monkeypatch.setattr(tool, "expand_matrix", lambda _: (cell,))
    calls = []
    class Request:
        @staticmethod
        def parse(raw): return raw
    monkeypatch.setattr(tool, "_worker_public_api", lambda: (Request, lambda request: calls.append(request) or {"request": request}, lambda result, request: {"timing": {"total_duration_ns": "1"}, "provenance": {}, "answer": "x"}))
    output = tool.worker_main(("--worker", "--mode", "steady", "--scenario-dir", str(scenarios), "--wheel", str(wheel)))
    assert len(calls) == 31
    assert set(output) == {"mode", "cells"}
    assert set(output["cells"]["c"]) == {"warmup_duration_ns", "warmup_analytical_sha256", "durations_ns", "analytical_sha256s"}
    assert len(output["cells"]["c"]["durations_ns"]) == 30


def test_main_emits_one_canonical_line_and_suppresses_ordinary_failure(monkeypatch, capsys):
    tool = load_tool()
    monkeypatch.setattr(tool, "worker_main", lambda argv: {"mode": "expected"})
    assert tool.main(["--worker", "--mode", "expected", "--scenario-dir", "/x", "--wheel", "/y"]) == 0
    assert capsys.readouterr().out == '{"mode":"expected"}\n'
    monkeypatch.setattr(tool, "worker_main", lambda argv: (_ for _ in ()).throw(tool.BenchmarkError("bad")))
    assert tool.main(["bad"]) == 1
    assert capsys.readouterr().out == ""


def test_direct_script_guard_runs_only_after_stage_observer_is_defined():
    source = TOOL.read_text("utf-8")
    assert source.index("class CupyStageObserver:") < source.index(
        'if __name__ == "__main__":'
    )
