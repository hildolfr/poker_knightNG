"""Host-only RED/GREEN contracts for the Phase 6C evidence controller."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "run_cuda_benchmark.py"
D = "a" * 64


def load_tool():
    name = f"benchmark_controller_test_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def admission():
    uuid = "GPU-" + "0" * 32
    gpu = f"{uuid}, 4096, 4096\n".encode("ascii")
    return {
        "gpu_uuid": uuid,
        "free_bytes": "4294967296",
        "total_bytes": "4294967296",
        "compute_applications": [],
        "gpu_snapshot_hex": gpu.hex(),
        "compute_snapshot_hex": "",
        "gpu_snapshot_sha256": hashlib.sha256(gpu).hexdigest(),
        "compute_snapshot_sha256": hashlib.sha256(b"").hexdigest(),
    }


def worker_outputs():
    cells = [
        f"v1-{street}-o{opponents}-n{trials}"
        for street in ("preflop", "flop", "turn", "river")
        for opponents in ("1", "3", "6")
        for trials in ("10000", "100000", "500000", "1000000")
    ]
    expected = {key: D for key in cells}
    steady = {
        key: {
            "warmup_duration_ns": "10",
            "warmup_analytical_sha256": D,
            "durations_ns": ["10"] * 30,
            "analytical_sha256s": [D] * 30,
        }
        for key in cells
    }
    stage = {
        key: {
            "durations": {
                "h2d_ns": "0", "simulate_ns": "1",
                "reduction_ns": "1", "d2h_ns": "1",
            },
            "analytical_sha256": D,
        }
        for key in cells
    }
    return {
        "inventory": {
            "mode": "inventory",
            "installation": {
                "wheel_basename": "wheel.whl",
                "wheel_contents_verified": "true",
                "wheel_sha256": D,
            },
            "environment": {
                "os": "Linux", "kernel": "6.1", "python_version": "3.11.0",
                "cuda_driver_version": "1", "cuda_runtime_version": "1",
                "cupy_version": "1.0.0", "gpu_name": "GPU",
                "gpu_uuid": "GPU-" + "0" * 32, "compute_capability": "1.0",
                "device_memory_bytes": "4294967296",
            },
        },
        "expected": {"mode": "expected", "analytical_sha256s": expected},
        "cold": {"mode": "cold", "duration_ns": "10", "analytical_sha256": D},
        "warm": {"mode": "warm", "duration_ns": "10", "analytical_sha256": D},
        "steady": {"mode": "steady", "cells": steady},
        "stage": {
            "mode": "stage",
            "planned_batch_blocks": "256",
            "batch_counts": {
                cell_id: str(
                    (int(cell_id.rsplit("-n", 1)[1]) + 256 * 128 - 1) // (256 * 128)
                )
                for cell_id in expected
            },
            "cells": stage,
        },
    }


def _config(tool, tmp_path):
    return tool.ControllerConfig(
        ROOT, tmp_path / "wheel.whl", tmp_path / "sdist.tar.gz",
        tmp_path / "lock", tmp_path / "scenarios", tmp_path / "manifest",
        tmp_path / "out.json",
    )


def _common_fakes(tool, tmp_path, monkeypatch, outputs):
    monkeypatch.setattr(tool, "_validate_config", lambda _config: None)
    monkeypatch.setattr(tool, "_verify_final_record", lambda _config, _record: None)
    monkeypatch.setattr(tool, "admit_gpu", lambda **_: admission())
    monkeypatch.setattr(
        tool,
        "_file",
        lambda path: {"basename": path.name, "sha256": D, "size_bytes": "1"},
    )
    monkeypatch.setattr(
        tool,
        "_source",
        lambda config: {
            "git_sha": "0" * 40, "branch": "main", "clean": "true",
            "benchmark_tool_sha256": D, "cuda_runtime_source_sha256": D,
        },
    )
    monkeypatch.setattr(
        tool,
        "_wheel_closure",
        lambda _path: [{"basename": "x", "sha256": D, "size_bytes": "1"}],
    )
    monkeypatch.setattr(tool, "prepare_cold_cache", lambda path: path.mkdir(mode=0o700))
    cache_seal = {
        "cold_result_sha256": D,
        "files": [{"path": "kernel.bin", "sha256": D, "size_bytes": "1"}],
        "format_version": "phase6c-cache-seal-v1",
    }
    monkeypatch.setattr(
        tool, "seal_cold_cache", lambda path, **_: cache_seal
    )
    monkeypatch.setattr(
        tool, "verify_warm_cache", lambda path: cache_seal
    )


def test_import_is_stdlib_only_and_cuda_inert():
    tool = load_tool()
    assert "cupy" not in sys.modules
    assert tool.__name__


def test_config_rejects_artifacts_outside_clean_repo_before_admission(tmp_path):
    tool = load_tool()
    config = _config(tool, tmp_path)
    for path in (config.wheel, config.sdist, config.lockfile, config.scenario_manifest):
        path.write_bytes(b"x")
    config.scenario_directory.mkdir()
    with pytest.raises(tool.ControllerError, match="CONFIG"):
        tool._validate_config(config)


def test_controller_runs_exact_six_order_and_shared_sealed_warm_cache(tmp_path, monkeypatch):
    tool = load_tool()
    outputs = worker_outputs()
    seen = []
    _common_fakes(tool, tmp_path, monkeypatch, outputs)

    def fake(mode, **kwargs):
        seen.append((mode, kwargs["cache_dir"]))
        return tool.WorkerCapture(outputs[mode], canonical(outputs[mode]), b"")

    monkeypatch.setattr(tool, "run_worker_capture", fake)
    config = _config(tool, tmp_path)
    record = tool.run(config)
    assert [mode for mode, _ in seen] == list(tool.MODES)
    assert seen[2][1] == seen[3][1]
    assert len({path for _, path in seen}) == 5
    assert [worker["mode"] for worker in record["workers"]] == list(tool.MODES)
    assert [worker["stdout_bytes"] for worker in record["workers"]] == [
        str(len(canonical(outputs[mode]))) for mode in tool.MODES
    ]
    assert {worker["output_limit_bytes"] for worker in record["workers"]} == {"2097152"}
    assert record["startup_cache"]["cold_seal"] == record["startup_cache"]["warm_verified_seal"]
    assert record["startup_cache"]["cold_seal"]["files"][0]["path"] == "kernel.bin"
    batch_counts = record["workers"][5]["payload"]["batch_counts"]
    assert {value: list(batch_counts.values()).count(value) for value in ("1", "4", "16", "31")} == {
        "1": 12, "4": 12, "16": 12, "31": 12,
    }
    assert config.output.read_bytes() == canonical(record)


def test_admission_refuses_compute_apps(tmp_path, monkeypatch):
    tool = load_tool()
    rows = [b"GPU-00000000000000000000000000000000, 4096, 4096\n", b"17, 1\n"]
    monkeypatch.setattr(
        tool,
        "run_bounded",
        lambda *args, **kwargs: type("P", (), {"stdout": rows.pop(0), "stderr": b""})(),
    )
    with pytest.raises(tool.ControllerError, match="ADMISSION"):
        tool.admit_gpu(cwd=tmp_path)


def test_admission_normalizes_canonical_nvidia_uuid_and_retains_snapshots(tmp_path, monkeypatch):
    tool = load_tool()
    gpu = b"GPU-01234567-89AB-cdef-0123-456789abcdef, 4096, 8192\n"
    rows = [gpu, b""]
    monkeypatch.setattr(
        tool,
        "run_bounded",
        lambda *args, **kwargs: type("P", (), {"stdout": rows.pop(0), "stderr": b""})(),
    )
    snapshot = tool.admit_gpu(cwd=tmp_path)
    assert snapshot["gpu_uuid"] == "GPU-0123456789abcdef0123456789abcdef"
    assert snapshot["gpu_snapshot_hex"] == gpu.hex()
    assert snapshot["compute_snapshot_hex"] == ""


def test_digest_mismatch_refuses_without_output(tmp_path, monkeypatch):
    tool = load_tool()
    outputs = worker_outputs()
    outputs["warm"]["analytical_sha256"] = "b" * 64
    _common_fakes(tool, tmp_path, monkeypatch, outputs)
    monkeypatch.setattr(
        tool,
        "run_worker_capture",
        lambda mode, **kwargs: tool.WorkerCapture(outputs[mode], canonical(outputs[mode]), b""),
    )
    config = _config(tool, tmp_path)
    with pytest.raises(tool.ControllerError, match="CONTROLLER"):
        tool.run(config)
    assert not config.output.exists()


def test_source_identity_change_refuses_without_output(tmp_path, monkeypatch):
    tool = load_tool()
    outputs = worker_outputs()
    _common_fakes(tool, tmp_path, monkeypatch, outputs)
    monkeypatch.setattr(
        tool,
        "run_worker_capture",
        lambda mode, **_kwargs: tool.WorkerCapture(outputs[mode], canonical(outputs[mode]), b""),
    )
    config = _config(tool, tmp_path)
    source_values = [
        {"git_sha": "a" * 40, "branch": "main", "clean": "true", "benchmark_tool_sha256": D, "cuda_runtime_source_sha256": D},
        {"git_sha": "b" * 40, "branch": "main", "clean": "true", "benchmark_tool_sha256": D, "cuda_runtime_source_sha256": D},
    ]
    monkeypatch.setattr(tool, "_source", lambda _config: source_values.pop(0))
    with pytest.raises(tool.ControllerError, match="SOURCE_CHANGED"):
        tool.run(config)
    assert not config.output.exists()


def test_warm_cache_manifest_change_refuses_before_warm_worker(tmp_path, monkeypatch):
    tool = load_tool()
    outputs = worker_outputs()
    _common_fakes(tool, tmp_path, monkeypatch, outputs)
    seen = []

    def capture(mode, **_kwargs):
        seen.append(mode)
        return tool.WorkerCapture(outputs[mode], canonical(outputs[mode]), b"")

    monkeypatch.setattr(tool, "run_worker_capture", capture)
    monkeypatch.setattr(
        tool,
        "verify_warm_cache",
        lambda _path: {
            "cold_result_sha256": D,
            "files": [{"path": "kernel.bin", "sha256": "f" * 64, "size_bytes": "1"}],
            "format_version": "phase6c-cache-seal-v1",
        },
    )
    config = _config(tool, tmp_path)
    with pytest.raises(tool.ControllerError, match="CACHE_CHANGED"):
        tool.run(config)
    assert seen == ["inventory", "expected", "cold"]
    assert not config.output.exists()


def test_exclusive_output_refuses_existing_path(tmp_path):
    tool = load_tool()
    path = tmp_path / "evidence.json"
    path.write_bytes(b"existing")
    with pytest.raises(tool.ControllerError, match="OUTPUT"):
        tool.write_evidence_exclusive(path, {"x": "y"})


def test_exclusive_output_removes_final_path_if_directory_fsync_fails(tmp_path, monkeypatch):
    tool = load_tool()
    path = tmp_path / "evidence.json"
    calls = 0
    real_fsync = tool.os.fsync

    def fail_directory_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        return real_fsync(fd)

    monkeypatch.setattr(tool.os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError):
        tool.write_evidence_exclusive(path, {"value": "x"})
    assert not path.exists()


def test_main_reports_generic_ordinary_failure_but_propagates_baseexception(monkeypatch, capsys):
    tool = load_tool()
    argv = [
        "--repo-root", "/repo", "--wheel", "/wheel.whl", "--sdist", "/source.tar.gz",
        "--lockfile", "/lock", "--scenario-dir", "/scenarios",
        "--scenario-manifest", "/manifest", "--output", "/private/evidence.json",
    ]
    monkeypatch.setattr(tool, "run", lambda _config: (_ for _ in ()).throw(RuntimeError("private")))
    assert tool.main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Phase 6C private benchmark evidence: FAIL\n"
    monkeypatch.setattr(tool, "run", lambda _config: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        tool.main(argv)
