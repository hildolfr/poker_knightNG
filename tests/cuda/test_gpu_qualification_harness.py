"""CPU-only contract tests for the exact-SHA GPU qualification harness."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "qualify_gpu.py"
TARGET_SHA = "a" * 40
BRANCH = "revival/phase-5-qualification-harness"


def qualify_argv(run_id, wheel, sdist, *extra):
    return [
        "qualify", "--run-id", run_id, "--target-sha", TARGET_SHA, "--branch", BRANCH,
        "--wheel", str(wheel), "--sdist", str(sdist), *extra,
    ]


def load_tool():
    name = f"qualify_gpu_test_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_import_is_stdlib_only_cupy_inert_and_side_effect_free(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    before = set(sys.modules)
    tool = load_tool()
    assert calls == []
    assert "cupy" not in set(sys.modules) - before
    assert "poker_knight_ng" not in set(sys.modules) - before
    assert "cupy" not in tool.__dict__


def test_subprocess_output_is_actively_bounded(tmp_path, monkeypatch):
    tool = load_tool()
    monkeypatch.setattr(tool, "MAX_PROCESS_OUTPUT_BYTES", 1024)
    completed = tool._run(
        [sys.executable, "-c", "print('bounded')"], cwd=tmp_path, code="SUBPROCESS",
    )
    assert completed.stdout == "bounded\n"
    with pytest.raises(tool.QualificationError, match="SUBPROCESS"):
        tool._run(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 4096)"],
            cwd=tmp_path, code="SUBPROCESS",
        )


def test_pipe_inheriting_descendant_is_group_killed_without_unbounded_join(tmp_path, monkeypatch):
    tool = load_tool()
    monkeypatch.setattr(tool, "PIPE_DRAIN_TIMEOUT_SECONDS", 0.2)
    pid_file = tmp_path / "descendant.pid"
    script = (
        "import pathlib,subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))"
    )
    started = time.monotonic()
    with pytest.raises(tool.QualificationError, match="SUBPROCESS"):
        tool._run([sys.executable, "-c", script, str(pid_file)], cwd=tmp_path, code="SUBPROCESS")
    assert time.monotonic() - started < 3
    descendant = int(pid_file.read_text())
    for _ in range(50):
        try:
            os.kill(descendant, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("pipe-inheriting descendant survived process-group cleanup")


def test_strict_json_is_closed_byte_canonical_and_rejects_hostile_encodings():
    tool = load_tool()
    value = {"a": ["0"], "z": "ok"}
    assert tool.strict_json(tool.canonical(value)) == value
    hostile = (
        b'{"a":"1","a":"2"}\n',
        b'{"a":"\\u00e9"}\n',
        b'\xef\xbb\xbf{}\n',
        b'{}\r\n',
        b'{}\n\n',
        b'{"x":NaN}\n',
        b'{ "x":"1"}\n',
    )
    for raw in hostile:
        with pytest.raises(tool.VerificationError):
            tool.strict_json(raw)
    with pytest.raises(tool.VerificationError):
        tool.closed({"extra": "1"}, {"required"})


def test_atomic_write_is_durable_exclusive_and_propagates_process_control(tmp_path, monkeypatch):
    tool = load_tool()
    parents = []
    monkeypatch.setattr(tool, "_fsync_parent", lambda path: parents.append(path))
    destination = tmp_path / "qualification.json"
    tool.atomic_write(destination, b"{}\n")
    assert destination.read_bytes() == b"{}\n"
    assert parents == [tmp_path]
    with pytest.raises(FileExistsError):
        tool.atomic_write(destination, b"{}\n")

    original_replace = tool.os.replace
    monkeypatch.setattr(tool.os, "replace", lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        tool.atomic_write(tmp_path / "interrupted.json", b"{}\n")
    assert not (tmp_path / "interrupted.json").exists()
    assert not list(tmp_path.glob(".interrupted.json.*"))
    monkeypatch.setattr(tool.os, "replace", original_replace)


def test_child_environment_uses_only_run_owned_cache_and_explicit_cuda_variables(tmp_path, monkeypatch):
    tool = load_tool()
    monkeypatch.setenv("SECRET_TOKEN", "must-not-propagate")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/cuda/lib")
    cache = tmp_path / "cache"
    environment = tool.child_environment(cache, force_ptx=True)
    assert environment["CUPY_CACHE_DIR"] == str(cache)
    assert environment["CUDA_CACHE_PATH"] == str(cache / "driver")
    assert environment["CUDA_FORCE_PTX_JIT"] == "1"
    assert environment["PKNG_QUALIFICATION_WORKER"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["LD_LIBRARY_PATH"] == "/cuda/lib"
    assert "SECRET_TOKEN" not in environment


def _smoke(tool, *, duration="1"):
    aggregate = {
        "case_hash": "a" * 64,
        "completed_trials": "64",
        "equity_share_units": "420",
        "hero_category_counts": {
            "flush": "0", "four_of_a_kind": "0", "full_house": "0", "high_card": "64",
            "one_pair": "0", "straight": "0", "straight_flush": "0", "three_of_a_kind": "0", "two_pair": "0",
        },
        "losses": "63",
        "rejection_count": "0",
        "requested_trials": "64",
        "seed": "0x0000000000000001",
        "tie_by_other_winners": {str(index): "0" for index in range(1, 7)},
        "ties": "0",
        "unique_wins": "1",
    }
    return {
        "cases": {"case-1": {"aggregate": aggregate, "engine_duration_ns": duration}},
        "device_id": "cuda-uuid:" + "b" * 32,
        "kernel_id": "cuda-source-sha256:" + "c" * 64,
        "qualification": "cuda-deterministic-v1",
        "source_sha256": "c" * 64,
    }


def _inventory(tool, wheel: Path, source_sha: str):
    return {
        "build": {
            "contract_version": "v1", "engine_build_id": "poker-knight-ng-0.1.0",
            "rng_algorithm_id": "poker-knight-ng/philox4x32-10", "rng_algorithm_version": "1",
        },
        "cuda": {
            "compiler_options": ["-std=c++17", "--gpu-architecture=compute_120"],
            "driver_version": "13030", "runtime_version": "13020", "source_sha256": source_sha,
        },
        "cupy_version": "14.1.1",
        "device": {
            "compute_capability": "12.0", "device_id": "cuda-uuid:" + "b" * 32,
            "memory_free_bytes": str(12 * 1024**3), "memory_total_bytes": str(16 * 1024**3),
        },
        "installation": {
            "wheel_basename": wheel.name, "wheel_contents_verified": "true",
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        },
        "poker_knight_ng_version": "0.1.0",
        "python_version": "3.13.15",
        "wall_duration_ns": "1",
    }


def test_run_worker_uses_exact_argv_cache_env_and_fresh_ptx_process(tmp_path, monkeypatch):
    tool = load_tool()
    root = tmp_path / "root"
    root.mkdir()
    seed = root / "seed.json"
    seed.write_text("{}\n")
    cache = tmp_path / "ptx-cache"
    calls = []

    def fake_run(argv, *, cwd, code, env=None):
        calls.append((argv, cwd, code, env))
        return subprocess.CompletedProcess(argv, 0, tool.canonical(_smoke(tool)).decode(), "")

    monkeypatch.setattr(tool, "_run", fake_run)
    monkeypatch.setattr(tool.time, "monotonic_ns", lambda: 10 if not calls else 25)
    result = tool.run_worker(root, seed, cache, mode="smoke", force_ptx=True)
    argv, cwd, code, environment = calls[0]
    assert argv == [sys.executable, str(TOOL), "worker", "--mode", "smoke", "--seed-bank", str(seed)]
    assert cwd == root and code == "SUBPROCESS"
    assert environment["CUPY_CACHE_DIR"] == str(cache)
    assert environment["CUDA_FORCE_PTX_JIT"] == "1"
    assert result["wall_duration_ns"] == "15"


def test_installed_distribution_is_byte_bound_to_supplied_wheel(tmp_path):
    tool = load_tool()
    installed = tmp_path / "installed"
    wheel = tmp_path / "poker_knight_ng-0.1.0-py3-none-any.whl"
    members = {
        "poker_knight_ng/__init__.py": b"version = 'good'\n",
        "poker_knight_ng-0.1.0.dist-info/METADATA": b"Name: poker-knight-ng\nVersion: 0.1.0\n\n",
        "poker_knight_ng-0.1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
        "poker_knight_ng-0.1.0.dist-info/top_level.txt": b"poker_knight_ng\n",
    }
    record_lines = []
    for name, data in members.items():
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record_lines.append(f"{name},sha256={encoded},{len(data)}")
    record_name = "poker_knight_ng-0.1.0.dist-info/RECORD"
    members[record_name] = ("\n".join(record_lines + [f"{record_name},,"]) + "\n").encode()
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    for name, data in members.items():
        if name == record_name:
            continue
        path = installed / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    class Distribution:
        version = "0.1.0"
        def locate_file(self, name):
            return installed / name

    record = tool._verify_installed_wheel(Distribution(), wheel)
    assert record == {
        "wheel_basename": wheel.name,
        "wheel_contents_verified": "true",
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }
    (installed / "poker_knight_ng/__init__.py").write_bytes(b"version = 'tampered'\n")
    with pytest.raises(RuntimeError, match="differs"):
        tool._verify_installed_wheel(Distribution(), wheel)
    (installed / "poker_knight_ng/__init__.py").write_bytes(members["poker_knight_ng/__init__.py"])
    (installed / "poker_knight_ng/extra.py").write_bytes(b"unverified\n")
    with pytest.raises(RuntimeError, match="unverified"):
        tool._verify_installed_wheel(Distribution(), wheel)

    sparse = tmp_path / "poker_knight_ng-0.1.0-py3-none-any-sparse.whl"
    with zipfile.ZipFile(sparse, "w") as archive:
        archive.writestr("poker_knight_ng/__init__.py", b"version = 'good'\n")
    with pytest.raises(RuntimeError, match="filename|closure"):
        tool._verify_installed_wheel(Distribution(), sparse)


def test_artifact_copy_uses_admitted_descriptor_across_path_swap(tmp_path, monkeypatch):
    tool = load_tool()
    source = tmp_path / "source.whl"
    replacement = tmp_path / "replacement.whl"
    destination = tmp_path / "copied.whl"
    source.write_bytes(b"admitted")
    replacement.write_bytes(b"replacement")
    real_open = tool.os.open
    swapped = False

    def swapping_open(path, flags, *args, **kwargs):
        nonlocal swapped
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == source and not swapped:
            source.unlink()
            source.symlink_to(replacement)
            swapped = True
        return descriptor

    monkeypatch.setattr(tool.os, "open", swapping_open)
    tool.copy_artifact(source, destination)
    assert destination.read_bytes() == b"admitted"


def test_harness_uses_the_actual_runtime_compiler_cache_api():
    tool = load_tool()
    from poker_knight_ng import _cuda_runtime

    assert callable(_cuda_runtime.compiler_cache_key)
    class Device:
        compute_capability = "120"

    fake_cupy = type("FakeCupy", (), {
        "__version__": "14.1.1",
        "cuda": type("Cuda", (), {"Device": Device}),
    })()
    cache_key = _cuda_runtime.compiler_cache_key(fake_cupy)
    assert cache_key[1:] == ("120", ("-std=c++17", "--gpu-architecture=compute_120"))
    source = TOOL.read_text()
    assert "compiler_cache_key(cp)[2]" in source
    assert "_compile_options" not in source


def test_gpu_admission_records_bounded_apps_and_rejects_low_vram_before_workers(tmp_path, monkeypatch):
    tool = load_tool()

    def rows(query, _root):
        if query.startswith("gpu:"):
            return [["RTX 5070 Ti", "GPU-" + "b" * 32, "12.0", "16303", "12000", "42.5", "2100"]]
        return [["12", "512"], ["34", "256"]]

    monkeypatch.setattr(tool, "_nvidia_rows", rows)
    inventory = tool.gpu_admission(tmp_path)
    assert inventory["memory_free_before_bytes"] == str(12000 * 1024 * 1024)
    assert inventory["compute_applications"] == [
        {"pid": "12", "used_memory_mib": "512"}, {"pid": "34", "used_memory_mib": "256"},
    ]

    monkeypatch.setattr(tool, "_nvidia_rows", lambda query, _root: (
        [["RTX", "GPU-" + "b" * 32, "12.0", "16303", "1024", "0", "0"]]
        if query.startswith("gpu:") else []
    ))
    with pytest.raises(tool.QualificationError, match="ADMISSION"):
        tool.gpu_admission(tmp_path)


def test_checkout_gate_requires_exact_sha_clean_branch_and_argv(tmp_path, monkeypatch):
    tool = load_tool()
    outputs = iter([TARGET_SHA + "\n", BRANCH + "\n", TARGET_SHA + "\n", ""])
    calls = []

    def fake_run(argv, *, cwd, code, env=None):
        calls.append((argv, cwd, code, env))
        return subprocess.CompletedProcess(argv, 0, next(outputs), "")

    monkeypatch.setattr(tool, "_run", fake_run)
    assert tool.checkout_identity(tmp_path, TARGET_SHA, BRANCH) == (TARGET_SHA, BRANCH)
    assert calls[2][0] == ["git", "rev-parse", f"refs/heads/{BRANCH}"]
    assert calls[3][0] == ["git", "status", "--porcelain", "--untracked-files=all"]
    assert all(call[3] is None for call in calls)

    monkeypatch.setattr(tool, "_run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "short\n", ""))
    with pytest.raises(tool.QualificationError, match="CHECKOUT"):
        tool.checkout_identity(tmp_path, TARGET_SHA, BRANCH)


def _prepare_root(tool, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "uv.lock").write_bytes(b"lock\n")
    seed = root / "validation/holdem/v1/rng_seed_bank.json"
    seed.parent.mkdir(parents=True)
    seed_case = {
        "board_card_ids": [], "canonical_case_hash_hex": "a" * 64,
        "expected": {
            "completed_trials": "64", "equity_share_units": "420",
            "hero_category_counts": {
                "flush": "0", "four_of_a_kind": "0", "full_house": "0", "high_card": "64",
                "one_pair": "0", "straight": "0", "straight_flush": "0", "three_of_a_kind": "0", "two_pair": "0",
            },
            "losses": "63", "rejection_count": "0",
            "tie_by_other_winners": {str(index): "0" for index in range(1, 7)},
            "unique_wins": "1",
        },
        "hero_card_ids": [0, 1], "id": "case-1", "opponent_count": 1,
        "requested_trials": "64", "seed": "0x0000000000000001",
    }
    seed.write_bytes(load_tool().canonical({"exact_vectors": [seed_case]}))
    manifest = root / "validation/holdem/v1/manifests/rng_seed_bank.sha256"
    manifest.parent.mkdir(parents=True)
    digest = hashlib.sha256(seed.read_bytes()).hexdigest()
    manifest.write_text(f"{digest}  validation/holdem/v1/rng_seed_bank.json\n", encoding="ascii")
    source = root / "src/poker_knight_ng/cuda-sources"
    source.mkdir(parents=True)
    for name in tool.SOURCE_NAMES:
        (source / name).write_bytes(name.encode("ascii"))
    wheel = tmp_path / "poker_knight_ng-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    sdist = tmp_path / "poker_knight_ng-0.1.0.tar.gz"
    sdist.write_bytes(b"sdist")
    return root, seed, manifest, wheel, sdist


def _gpu(tool):
    return {
        "clock_sm_mhz": "2100", "compute_applications": [], "compute_capability": "12.0",
        "memory_free_before_bytes": str(12 * 1024**3), "memory_total_bytes": str(16 * 1024**3),
        "name": "RTX 5070 Ti", "nvidia_uuid": "GPU-" + "b" * 32,
        "power_draw_w": "42.5",
    }


def test_low_vram_failure_writes_canonical_evidence_without_starting_worker(tmp_path, monkeypatch):
    tool = load_tool()
    root, _seed, _manifest, wheel, sdist = _prepare_root(tool, tmp_path)
    monkeypatch.setattr(tool, "repo_root", lambda: root)
    monkeypatch.setattr(tool, "checkout_identity", lambda _root, _target, _branch: (TARGET_SHA, BRANCH))
    monkeypatch.setattr(tool, "gpu_admission", lambda _root: (_ for _ in ()).throw(tool.QualificationError("ADMISSION")))
    monkeypatch.setattr(tool, "run_worker", lambda *args, **kwargs: pytest.fail("worker started below admission floor"))
    monkeypatch.setenv("RUN_CUDA_QUALIFICATION", "1")
    status = tool.main(qualify_argv("low-vram", wheel, sdist))
    evidence_path = root / "artifacts/qualification/low-vram/qualification.json"
    assert status == 1 and evidence_path.is_file()
    raw = evidence_path.read_bytes()
    evidence = tool.strict_json(raw)
    assert raw == tool.canonical(evidence)
    assert evidence["status"] == "failed" and evidence["error_codes"] == ["ADMISSION"]
    assert evidence["workers"] == {} and evidence["sanitizers"] == {}
    assert not any(str(tmp_path) in str(value) for value in evidence.values())


def test_ordinary_failure_is_sanitized_and_baseexception_is_not_converted(tmp_path, monkeypatch):
    tool = load_tool()
    root, _seed, _manifest, wheel, sdist = _prepare_root(tool, tmp_path)
    monkeypatch.setattr(tool, "repo_root", lambda: root)
    monkeypatch.setattr(tool, "checkout_identity", lambda *_args: (_ for _ in ()).throw(RuntimeError("secret /home/sam token")))
    monkeypatch.setenv("RUN_CUDA_QUALIFICATION", "1")
    assert tool.main(qualify_argv("ordinary", wheel, sdist)) == 1
    evidence = tool.strict_json((root / "artifacts/qualification/ordinary/qualification.json").read_bytes())
    assert evidence["error_codes"] == ["INTERNAL"]
    assert "secret" not in json.dumps(evidence)

    monkeypatch.setattr(tool, "checkout_identity", lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        tool.main(qualify_argv("interrupt", wheel, sdist))
    assert not (root / "artifacts/qualification/interrupt/qualification.json").exists()


def _run_success(tool, tmp_path, monkeypatch):
    root, seed, _manifest, wheel, sdist = _prepare_root(tool, tmp_path)
    source_sha = tool.source_digest(root)
    smoke = _smoke(tool)
    smoke["source_sha256"] = source_sha
    smoke["kernel_id"] = "cuda-source-sha256:" + source_sha
    calls = {"workers": [], "sanitizers": [], "pytest": []}
    admissions = [_gpu(tool), _gpu(tool)]
    monkeypatch.setattr(tool, "repo_root", lambda: root)
    monkeypatch.setattr(tool, "checkout_identity", lambda _root, _target, _branch: (TARGET_SHA, BRANCH))
    monkeypatch.setattr(tool, "gpu_admission", lambda _root: admissions.pop(0))
    monkeypatch.setattr(tool, "_bounded_version", lambda argv, _root, **_kwargs: " | ".join(argv))

    def fake_worker(_root, _seed, cache, *, mode, force_ptx=False, wheel=None):
        calls["workers"].append((cache, mode, force_ptx, wheel))
        if mode == "inventory":
            assert wheel is not None
            return _inventory(tool, wheel, source_sha)
        return {**json.loads(json.dumps(smoke)), "wall_duration_ns": "10"}

    monkeypatch.setattr(tool, "run_worker", fake_worker)

    def fake_run(argv, *, cwd, code, env=None):
        calls["pytest"].append((argv, cwd, code, env))
        junit_arg = next(value for value in argv if value.startswith("--junitxml="))
        Path(junit_arg.split("=", 1)[1]).write_text('<testsuite tests="587" failures="0" errors="0"/>', encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(tool, "_run", fake_run)

    def fake_sanitizer(_root, run, _seed, cache, sanitizer_tool):
        calls["sanitizers"].append((cache, sanitizer_tool))
        log = run / f"{sanitizer_tool}.log"
        log.write_text("========= " + tool.SANITIZER_ZERO_MARKERS[sanitizer_tool] + "\n", encoding="utf-8")
        return json.loads(json.dumps(smoke)), tool.file_record(log)

    monkeypatch.setattr(tool, "run_sanitizer", fake_sanitizer)
    monkeypatch.setenv("RUN_CUDA_QUALIFICATION", "1")
    status = tool.main(qualify_argv("success", wheel, sdist))
    return root, wheel, sdist, calls, status


def test_run_sanitizer_requires_exact_tool_specific_zero_summary(tmp_path, monkeypatch):
    tool = load_tool()
    root = tmp_path / "root"
    run = tmp_path / "run"
    cache = tmp_path / "cache"
    root.mkdir()
    run.mkdir()
    cache.mkdir()
    seed_bank = root / "seed-bank.json"
    seed_bank.write_text("{}", encoding="utf-8")
    markers = {
        "memcheck": "========= ERROR SUMMARY: 0 errors\n",
        "racecheck": "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n",
        "initcheck": "========= ERROR SUMMARY: 0 errors\n",
        "synccheck": "========= ERROR SUMMARY: 0 errors\n",
    }

    def fake_run(argv, *, cwd, code, env=None):
        sanitizer_tool = argv[argv.index("--tool") + 1]
        log = Path(argv[argv.index("--log-file") + 1])
        log.write_text(markers[sanitizer_tool], encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    monkeypatch.setattr(tool, "_run", fake_run)
    monkeypatch.setattr(tool, "parse_worker", lambda _text: {})
    for sanitizer_tool in tool.SANITIZERS:
        worker, record = tool.run_sanitizer(root, run, seed_bank, cache, sanitizer_tool)
        assert worker == {} and record["sha256"]

    markers["racecheck"] = markers["memcheck"]
    with pytest.raises(tool.QualificationError, match="SANITIZER"):
        tool.run_sanitizer(root, run, seed_bank, cache, "racecheck")


def test_run_sanitizer_rejects_unknown_tool_before_subprocess(tmp_path, monkeypatch):
    tool = load_tool()
    root = tmp_path / "root"
    run = tmp_path / "run"
    cache = tmp_path / "cache"
    root.mkdir()
    run.mkdir()
    cache.mkdir()
    seed_bank = root / "seed-bank.json"
    seed_bank.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tool, "_run", lambda *_args, **_kwargs: pytest.fail("unknown sanitizer executed"))
    with pytest.raises(tool.QualificationError, match="SANITIZER"):
        tool.run_sanitizer(root, run, seed_bank, cache, "unknown")


def test_success_orchestration_uses_distinct_cold_warm_ptx_caches_full_pytest_and_four_sanitizers(tmp_path, monkeypatch):
    tool = load_tool()
    root, wheel, sdist, calls, status = _run_success(tool, tmp_path, monkeypatch)
    assert status == 0
    assert [(mode, ptx) for _cache, mode, ptx, _wheel in calls["workers"]] == [
        ("inventory", False), ("smoke", False), ("smoke", False), ("smoke", True),
    ]
    inventory_cache, cold_cache, warm_cache, ptx_cache = [row[0] for row in calls["workers"]]
    assert inventory_cache != cold_cache == warm_cache
    assert inventory_cache.name == "inventory-cache"
    assert ptx_cache != cold_cache and ptx_cache.name == "force-ptx-cache"
    assert calls["workers"][0][3] == root / "artifacts/qualification/success/files" / wheel.name
    assert all(row[3] is None for row in calls["workers"][1:])
    assert [row[1] for row in calls["sanitizers"]] == list(tool.SANITIZERS)
    pytest_argv, pytest_cwd, pytest_code, pytest_env = calls["pytest"][0]
    assert pytest_argv[:4] == [sys.executable, "-m", "pytest", "-q"]
    assert pytest_cwd == root and pytest_code == "JUNIT" and "SECRET_TOKEN" not in pytest_env

    evidence_path = root / "artifacts/qualification/success/qualification.json"
    raw = evidence_path.read_bytes()
    evidence = tool.strict_json(raw)
    assert evidence["status"] == "passed" and evidence["error_codes"] == []
    assert evidence["source"]["git_sha"] == "a" * 40
    assert evidence["source"]["cuda_source_sha256"] == tool.source_digest(root)
    assert set(evidence["sanitizers"]) == set(tool.SANITIZERS)
    assert evidence["artifacts"]["wheel"]["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert evidence["artifacts"]["sdist"]["sha256"] == hashlib.sha256(sdist.read_bytes()).hexdigest()
    assert (evidence_path.parent / "files" / wheel.name).read_bytes() == wheel.read_bytes()
    assert all(str(tmp_path) not in value for value in _all_strings(evidence))
    assert tool.verify(evidence_path, root) == 0


def _all_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)


def test_verifier_rejects_worker_divergence_provenance_source_and_artifact_tampering(tmp_path, monkeypatch):
    tool = load_tool()
    root, _wheel, _sdist, _calls, status = _run_success(tool, tmp_path, monkeypatch)
    assert status == 0
    evidence_path = root / "artifacts/qualification/success/qualification.json"
    original = tool.strict_json(evidence_path.read_bytes())

    mutations = []
    worker_divergence = json.loads(json.dumps(original))
    worker_divergence["workers"]["warm"]["cases"]["case-1"]["aggregate"]["losses"] = "62"
    mutations.append(worker_divergence)
    provenance = json.loads(json.dumps(original))
    provenance["workers"]["cold"]["kernel_id"] = "cuda-source-sha256:" + "0" * 64
    mutations.append(provenance)
    source = json.loads(json.dumps(original))
    source["source"]["compiler_options"] = ["--use_fast_math"]
    mutations.append(source)
    extra = json.loads(json.dumps(original))
    extra["unexpected"] = "x"
    mutations.append(extra)

    for mutated in mutations:
        evidence_path.write_bytes(tool.canonical(mutated))
        assert tool.verify(evidence_path, root) == 1
    evidence_path.write_bytes(tool.canonical(original))
    (evidence_path.parent / "files" / original["artifacts"]["wheel"]["basename"]).write_bytes(b"tampered")
    assert tool.verify(evidence_path, root) == 1


def test_verifier_rejects_hash_consistent_generic_racecheck_marker(tmp_path, monkeypatch):
    tool = load_tool()
    root, _wheel, _sdist, _calls, status = _run_success(tool, tmp_path, monkeypatch)
    assert status == 0
    run = root / "artifacts/qualification/success"
    evidence_path = run / "qualification.json"
    evidence = tool.strict_json(evidence_path.read_bytes())
    racecheck_log = run / "racecheck.log"
    racecheck_log.write_text("========= ERROR SUMMARY: 0 errors\n", encoding="utf-8")
    evidence["artifacts"]["sanitizer_racecheck"] = tool.file_record(racecheck_log)
    evidence_path.write_bytes(tool.canonical(evidence))
    assert tool.verify(evidence_path, root) == 1


def test_run_sanitizer_uses_exact_argv_no_shell_and_requires_zero_error_summary(tmp_path, monkeypatch):
    tool = load_tool()
    root = tmp_path / "root"
    run = tmp_path / "run"
    cache = tmp_path / "cache"
    root.mkdir(); run.mkdir(); cache.mkdir()
    seed = root / "seed.json"
    seed.write_text("{}\n")
    smoke = _smoke(tool)
    calls = []

    def fake_run(argv, *, cwd, code, env=None):
        calls.append((argv, cwd, code, env))
        log_index = argv.index("--log-file") + 1
        Path(argv[log_index]).write_text(
            "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, tool.canonical(smoke).decode(), "")

    monkeypatch.setattr(tool, "_run", fake_run)
    observation, record = tool.run_sanitizer(root, run, seed, cache, "racecheck")
    argv, cwd, code, environment = calls[0]
    assert argv[:7] == ["compute-sanitizer", "--tool", "racecheck", "--error-exitcode", "91", "--log-file", str(run / "racecheck.log")]
    assert argv[7:] == [sys.executable, str(TOOL), "worker", "--mode", "smoke", "--seed-bank", str(seed)]
    assert cwd == root and code == "SANITIZER"
    assert environment["CUPY_CACHE_DIR"] == str(cache)
    assert observation == smoke and record["basename"] == "racecheck.log"

    def one_error(argv, **_kwargs):
        Path(argv[argv.index("--log-file") + 1]).write_text("========= ERROR SUMMARY: 1 error\n")
        return subprocess.CompletedProcess(argv, 0, tool.canonical(smoke).decode(), "")

    monkeypatch.setattr(tool, "_run", one_error)
    with pytest.raises(tool.QualificationError, match="SANITIZER"):
        tool.run_sanitizer(root, run, seed, cache, "memcheck")


def test_cli_rejects_missing_gate_unsafe_ids_symlinks_and_output_traversal(tmp_path, monkeypatch):
    tool = load_tool()
    root, _seed, _manifest, wheel, sdist = _prepare_root(tool, tmp_path)
    monkeypatch.setattr(tool, "repo_root", lambda: root)
    assert tool.main(qualify_argv("../bad", wheel, sdist)) == 2
    assert tool.main(qualify_argv("good", wheel, sdist)) == 2
    monkeypatch.setenv("RUN_CUDA_QUALIFICATION", "1")
    assert tool.main(qualify_argv("good", wheel, sdist, "--output-root", str(tmp_path / "escape"))) == 2
    link = tmp_path / "linked.whl"
    link.symlink_to(wheel)
    assert tool.main(qualify_argv("good", link, sdist)) == 2
    assert not (root / "artifacts/qualification/good").exists()


def test_qualification_namespace_rejects_symlinked_ancestor(tmp_path, monkeypatch):
    tool = load_tool()
    root, _seed, _manifest, wheel, sdist = _prepare_root(tool, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "artifacts").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(tool, "repo_root", lambda: root)
    monkeypatch.setenv("RUN_CUDA_QUALIFICATION", "1")
    assert tool.main(qualify_argv("escaped", wheel, sdist)) == 2
    assert not (outside / "qualification").exists()


def test_manifest_mismatch_is_detected_before_gpu_admission(tmp_path, monkeypatch):
    tool = load_tool()
    root, seed, _manifest, wheel, sdist = _prepare_root(tool, tmp_path)
    seed.write_bytes(b"tampered\n")
    monkeypatch.setattr(tool, "repo_root", lambda: root)
    monkeypatch.setattr(tool, "checkout_identity", lambda _root, _target, _branch: (TARGET_SHA, BRANCH))
    monkeypatch.setattr(tool, "gpu_admission", lambda _root: pytest.fail("GPU admission reached"))
    monkeypatch.setenv("RUN_CUDA_QUALIFICATION", "1")
    assert tool.main(qualify_argv("manifest", wheel, sdist)) == 1
    evidence = tool.strict_json((root / "artifacts/qualification/manifest/qualification.json").read_bytes())
    assert evidence["error_codes"] == ["MANIFEST"]


def test_failed_evidence_verifier_accepts_only_canonical_fixed_codes(tmp_path):
    tool = load_tool()
    root = tmp_path / "root"
    run = root / "artifacts/qualification/failed"
    run.mkdir(parents=True)
    evidence = tool.empty_evidence("failed")
    evidence["error_codes"] = ["ADMISSION"]
    path = run / "qualification.json"
    path.write_bytes(tool.canonical(evidence))
    assert tool.verify(path, root) == 0
    evidence["error_codes"] = ["raw /home/user secret"]
    path.write_bytes(tool.canonical(evidence))
    assert tool.verify(path, root) == 1


def test_gitignore_declares_qualification_artifacts():
    assert "/artifacts/" in (ROOT / ".gitignore").read_text("utf-8")
