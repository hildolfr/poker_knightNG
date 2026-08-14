"""CPU-only worker/controller contracts for the Phase 6C benchmark tool."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "benchmark_equity.py"



def load_tool():
    name = f"benchmark_equity_worker_test_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_import_is_stdlib_only_cupy_inert_and_does_not_start_a_worker(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    before = set(sys.modules)

    tool = load_tool()

    assert calls == []
    assert "cupy" not in set(sys.modules) - before
    assert "poker_knight_ng" not in set(sys.modules) - before
    assert "cupy" not in tool.__dict__


@pytest.mark.parametrize("argv", [
    ("--worker", "--mode", "inventory", "--mode", "cold"),
    ("--worker", "--mode", "unknown"),
    ("--mode", "cold"),
])
def test_worker_main_refuses_missing_or_mixed_closed_mode(argv, monkeypatch):
    tool = load_tool()
    monkeypatch.setenv(tool.WORKER_MARKER, "1")

    with pytest.raises(tool.BenchmarkError, match="WORKER_MODE"):
        tool.worker_main(argv)


@pytest.mark.parametrize("marker", [None, "", "0", "yes", "1 "])
def test_require_private_worker_requires_exact_private_marker(marker, monkeypatch):
    tool = load_tool()
    monkeypatch.delenv(tool.WORKER_MARKER, raising=False)
    if marker is not None:
        monkeypatch.setenv(tool.WORKER_MARKER, marker)

    with pytest.raises(tool.BenchmarkError, match="PRIVATE_WORKER"):
        tool.require_private_worker("inventory")


def test_parse_worker_output_accepts_one_canonical_ascii_json_line_only():
    tool = load_tool()
    expected = {"mode": "cold", "result": {"duration_ns": "17"}}

    assert tool.parse_worker_output(tool.canonical(expected), b"") == expected
    for raw in (
        b'{"mode":"cold","mode":"warm"}\n',
        b'{"mode":"cold","result":{"duration_ns":NaN}}\n',
        b'{"mode":"cold"}\n{"mode":"warm"}\n',
        b'{"mode":"cold"}',
        b'{"mode":"cold"}\ntrailing',
        b'{"mode":"caf\xc3\xa9"}\n',
        b'{"mode":"' + b"x" * tool.MAX_WORKER_OUTPUT_BYTES + b'"}\n',
    ):
        with pytest.raises(tool.BenchmarkError):
            tool.parse_worker_output(raw, b"")
    with pytest.raises(tool.BenchmarkError, match="WORKER_OUTPUT"):
        tool.parse_worker_output(tool.canonical(expected), b"diagnostic")


def test_run_worker_uses_exact_isolated_subprocess_contract(tmp_path, monkeypatch):
    tool = load_tool()
    calls = []
    cache = tmp_path / "cache"
    scenarios = tmp_path / "scenarios"
    wheel = tmp_path / "poker_knight_ng.whl"
    expected = {"mode": "warm", "result": {"duration_ns": "17"}}

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return type("Completed", (), {
            "stdout": tool.canonical(expected),
            "stderr": b"",
        })()

    monkeypatch.setattr(tool, "run_bounded", fake_run)
    result = tool.run_worker(
        "warm",
        cwd=tmp_path,
        cache_dir=cache,
        scenario_directory=scenarios,
        wheel=wheel,
    )

    assert result == expected
    assert calls == [(
        tool.worker_argv("warm", scenario_directory=scenarios, wheel=wheel),
        {
            "cwd": tmp_path,
            "env": {
                "CUPY_CACHE_DIR": str(cache),
                "CUDA_CACHE_PATH": str(cache / "driver"),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                tool.WORKER_MARKER: "1",
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
            },
            "timeout_seconds": tool.WORKER_TIMEOUT_SECONDS,
            "output_limit": tool.MAX_WORKER_OUTPUT_BYTES,
        },
    )]


def test_steady_samples_exclude_warmup_and_stage_profile_never_becomes_latency_sample():
    tool = load_tool()
    calls = []

    def run_once(mode):
        calls.append(mode)
        return {
            "result": {"completed_trials": "100", "equity_units": "42000"},
            "duration_ns": str(len(calls)),
            "stage_profile": {"simulate_gpu_ns": "9"} if mode == "stage" else None,
        }

    samples = tool.collect_steady_samples(run_once)
    stage = tool.collect_stage_profile(run_once)

    assert calls == ["steady"] * 31 + ["stage"]
    assert samples == tuple(str(value) for value in range(2, 32))
    assert stage == {"simulate_gpu_ns": "9"}
    assert "stage" not in samples


def test_observer_stage_names_translate_to_the_exact_evidence_contract():
    tool = load_tool()
    profile = {
        "h2d_gpu_ns": "0",
        "simulate_gpu_ns": "11",
        "reduction_gpu_ns": "22",
        "d2h_gpu_ns": "33",
    }
    assert tool.evidence_stage_durations(profile) == {
        "h2d_ns": "0",
        "simulate_ns": "11",
        "reduction_ns": "22",
        "d2h_ns": "33",
    }
    for hostile in ({**profile, "extra": "1"}, {"h2d_gpu_ns": "1"}):
        with pytest.raises(tool.BenchmarkError, match="STAGE_PROFILE"):
            tool.evidence_stage_durations(hostile)
