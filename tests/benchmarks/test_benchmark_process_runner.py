import importlib.util
import os
from pathlib import Path
import signal
import sys
import time

import pytest


ROOT = Path(__file__).parents[2]


def load_tool():
    path = ROOT / "tools/benchmark_equity.py"
    spec = importlib.util.spec_from_file_location("benchmark_equity_process", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bounded_runner_returns_exact_independent_stream_bytes(tmp_path):
    tool = load_tool()
    completed = tool.run_bounded(
        [sys.executable, "-c", "import sys;sys.stdout.write('ok\\n')"],
        cwd=tmp_path,
        env={},
        timeout_seconds=2,
        output_limit=1024,
    )
    assert completed.returncode == 0
    assert completed.stdout == b"ok\n"
    assert completed.stderr == b""


def test_bounded_runner_kills_group_on_output_overflow_and_timeout(tmp_path):
    tool = load_tool()
    with pytest.raises(tool.BenchmarkError, match="PROCESS_OUTPUT"):
        tool.run_bounded(
            [sys.executable, "-c", "import sys;sys.stdout.write('x'*4096)"],
            cwd=tmp_path,
            env={},
            timeout_seconds=2,
            output_limit=128,
        )
    started = time.monotonic()
    with pytest.raises(tool.BenchmarkError, match="PROCESS_TIMEOUT"):
        tool.run_bounded(
            [sys.executable, "-c", "import time;time.sleep(60)"],
            cwd=tmp_path,
            env={},
            timeout_seconds=0.1,
            output_limit=128,
        )
    assert time.monotonic() - started < 2


def test_bounded_runner_rejects_pipe_inheriting_descendant(tmp_path):
    tool = load_tool()
    pid_file = tmp_path / "descendant.pid"
    script = (
        "import pathlib,subprocess,sys;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))"
    )
    with pytest.raises(tool.BenchmarkError, match="PROCESS_PIPE"):
        tool.run_bounded(
            [sys.executable, "-c", script, str(pid_file)],
            cwd=tmp_path,
            env={},
            timeout_seconds=2,
            output_limit=1024,
        )
    descendant = int(pid_file.read_text(encoding="ascii"))
    for _ in range(100):
        try:
            os.kill(descendant, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("pipe-inheriting descendant survived process-group cleanup")


def test_post_popen_setup_interrupt_propagates_after_group_cleanup(tmp_path, monkeypatch):
    tool = load_tool()
    real_thread = tool.threading.Thread
    calls = 0

    def interrupting_thread(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt()
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(tool.threading, "Thread", interrupting_thread)
    with pytest.raises(KeyboardInterrupt):
        tool.run_bounded(
            [sys.executable, "-c", "import time;time.sleep(60)"],
            cwd=tmp_path,
            env={},
            timeout_seconds=2,
            output_limit=1024,
        )
