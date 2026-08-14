import io
import json
from pathlib import Path
import subprocess
import sys

import jsonschema
import pytest


ROOT = Path(__file__).parents[2]
CORRELATION_ID = "pk_" + "1" * 32


def raw_request(*, backend="cpu_reference", trials="2"):
    return {
        "backend": backend,
        "board_cards": ["2s", "3h", "Td"],
        "contract_version": "v1",
        "hero_cards": ["As", "Ah"],
        "opponent_count": "2",
        "requested_trials": trials,
        "rng": {
            "algorithm_id": "poker-knight-ng/philox4x32-10",
            "algorithm_version": "1",
        },
        "seed": "0x0123456789abcdef",
    }


def canonical(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def invoke(args, payload, *, correlation_factory=lambda: CORRELATION_ID):
    from poker_knight_ng import cli

    stdout = io.BytesIO()
    stderr = io.BytesIO()
    exit_code = cli.run(
        args,
        io.BytesIO(payload),
        stdout,
        stderr,
        correlation_factory=correlation_factory,
    )
    return exit_code, stdout.getvalue(), stderr.getvalue()


def parse_problem(payload):
    value = json.loads(payload)
    schema = json.loads((ROOT / "contracts/v1/problem.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(value)
    return value


def test_cpu_cli_success_is_one_canonical_schema_valid_result_line():
    code, stdout, stderr = invoke(["solve"], canonical(raw_request()))
    assert code == 0
    assert stderr == b""
    assert stdout.endswith(b"\n") and not stdout.endswith(b"\n\n")
    parsed = json.loads(stdout)
    schema = json.loads(
        (ROOT / "contracts/v1/equity-result.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(parsed)
    assert stdout == canonical(parsed)
    assert parsed["backend"] == "cpu_reference"


def test_solve_command_keeps_cuda_unavailable_without_any_execution(monkeypatch):
    import poker_knight_ng.engine.local as local

    monkeypatch.setattr(
        local,
        "run_cpu_monte_carlo",
        lambda **_: pytest.fail("must not execute CPU"),
    )
    code, stdout, stderr = invoke(
        ["solve"],
        canonical(raw_request(backend="cuda")),
    )
    assert (code, stdout) == (4, b"")
    assert parse_problem(stderr)["code"] == "BACKEND_UNAVAILABLE"


def test_solve_cuda_is_explicit_and_never_falls_back(monkeypatch):
    from poker_knight_ng import cli
    import poker_knight_ng.engine as engine

    calls = []
    sentinel = object()

    class FakeCUDAEngine:
        def __init__(self):
            calls.append("constructed")

        def solve(self, request):
            calls.append(request.backend)
            return sentinel

    monkeypatch.setattr(engine, "CUDAEngine", FakeCUDAEngine)
    monkeypatch.setattr(
        cli,
        "serialize_equity_result",
        lambda result, request: {
            "backend": request.backend,
            "sentinel": result is sentinel,
        },
    )
    monkeypatch.setattr(
        cli,
        "solve",
        lambda _request: pytest.fail("must not call CPU solve"),
    )

    code, stdout, stderr = invoke(
        ["solve-cuda"],
        canonical(raw_request(backend="cuda")),
    )
    assert (code, stderr) == (0, b"")
    assert stdout == b'{"backend":"cuda","sentinel":true}\n'
    assert calls == ["constructed", "cuda"]


def test_solve_cuda_rejects_cpu_before_engine_construction(monkeypatch):
    import poker_knight_ng.engine as engine

    monkeypatch.setattr(
        engine,
        "CUDAEngine",
        lambda: pytest.fail("must not construct CUDA engine"),
    )
    code, stdout, stderr = invoke(
        ["solve-cuda"],
        canonical(raw_request()),
    )
    assert (code, stdout) == (3, b"")
    assert parse_problem(stderr)["code"] == "UNSUPPORTED_REQUEST"


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b"   \n",
        b"[]\n",
        b"\xff",
        b"\xef\xbb\xbf{}\n",
        b"{not-json}\n",
        b"{}{}\n",
        b'{"backend":"cpu_reference","backend":"cuda"}\n',
        b'{"rng":{"algorithm_id":"x","algorithm_id":"y"}}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
        b'{"value":-Infinity}\n',
    ),
)
def test_invalid_stdin_framing_is_closed_unsupported_request(payload):
    code, stdout, stderr = invoke(["solve"], payload)
    assert (code, stdout) == (2, b"")
    problem = parse_problem(stderr)
    assert problem["code"] == "UNSUPPORTED_REQUEST"
    assert problem["correlation_id"] == CORRELATION_ID
    assert b"Traceback" not in stderr


def test_whitespace_and_key_order_are_accepted_but_size_is_bounded():
    pretty = (json.dumps(raw_request(), indent=2) + "\n").encode()
    code, stdout, stderr = invoke(["solve"], pretty)
    assert code == 0 and stdout and stderr == b""

    from poker_knight_ng.cli import MAX_STDIN_BYTES

    compact = canonical(raw_request())
    exact = compact + b" " * (MAX_STDIN_BYTES - len(compact))
    assert len(exact) == MAX_STDIN_BYTES
    assert invoke(["solve"], exact)[0] == 0
    code, stdout, stderr = invoke(["solve"], exact + b" ")
    assert (code, stdout) == (2, b"")
    assert parse_problem(stderr)["code"] == "UNSUPPORTED_REQUEST"


def test_frozen_semantic_error_codes_and_exit_classes_are_preserved():
    unknown = raw_request()
    unknown["extra"] = "nope"
    code, stdout, stderr = invoke(["solve"], canonical(unknown))
    assert (code, stdout) == (3, b"")
    assert parse_problem(stderr)["code"] == "UNSUPPORTED_FIELD"

    invalid_card = raw_request()
    invalid_card["hero_cards"] = ["ZZ", "Ah"]
    code, stdout, stderr = invoke(["solve"], canonical(invalid_card))
    assert (code, stdout) == (3, b"")
    assert parse_problem(stderr)["code"] == "INVALID_CARD"


@pytest.mark.parametrize(
    ("problem_code", "exit_code"),
    (
        ("UNSUPPORTED_REQUEST", 3),
        ("BACKEND_UNAVAILABLE", 4),
        ("RESOURCE_EXHAUSTED", 4),
        ("RNG_REJECTION_EXHAUSTED", 5),
        ("INTERNAL_ERROR", 5),
    ),
)
def test_execution_problem_exit_map_is_exact(monkeypatch, problem_code, exit_code):
    from poker_knight_ng import cli
    from poker_knight_ng.contract.errors import problem

    def failed(_request):
        raise problem(problem_code)

    monkeypatch.setattr(cli, "solve", failed)
    code, stdout, stderr = invoke(["solve"], canonical(raw_request()))
    assert (code, stdout) == (exit_code, b"")
    assert parse_problem(stderr)["code"] == problem_code


def test_ordinary_execution_exception_is_closed_without_message_leak(monkeypatch):
    from poker_knight_ng import cli

    def failed(_request):
        raise RuntimeError("raw path /tmp/private and submitted As Ah")

    monkeypatch.setattr(cli, "solve", failed)
    code, stdout, stderr = invoke(["solve"], canonical(raw_request()))
    assert (code, stdout) == (5, b"")
    assert parse_problem(stderr)["code"] == "INTERNAL_ERROR"
    assert b"RuntimeError" not in stderr
    assert b"/tmp/private" not in stderr
    assert b"As Ah" not in stderr


def test_invocation_help_and_correlation_failure_are_frozen():
    from poker_knight_ng import cli

    for args in ([], ["unknown"], ["solve", "extra"]):
        code, stdout, stderr = invoke(args, b"")
        assert (code, stdout) == (2, b"")
        assert parse_problem(stderr)["code"] == "UNSUPPORTED_REQUEST"
        assert b"usage:" not in stderr.lower()

    code, stdout, stderr = invoke(["--help"], b"")
    assert (code, stderr) == (0, b"")
    assert stdout == cli.HELP

    def failed_entropy():
        raise RuntimeError("secret entropy path")

    code, stdout, stderr = invoke(
        ["solve"],
        b"not json",
        correlation_factory=failed_entropy,
    )
    problem = parse_problem(stderr)
    assert (code, stdout, problem["code"]) == (5, b"", "INTERNAL_ERROR")
    assert problem["correlation_id"] == "pk_" + "0" * 32
    assert b"entropy" not in stderr and b"RuntimeError" not in stderr


def test_baseexception_propagates_and_broken_pipes_are_silent(monkeypatch):
    from poker_knight_ng import cli

    def interrupted(_request):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "solve", interrupted)
    with pytest.raises(KeyboardInterrupt):
        invoke(["solve"], canonical(raw_request()))

    class BrokenWriter:
        def write(self, _payload):
            raise BrokenPipeError

    assert cli.run(
        ["--help"],
        io.BytesIO(),
        BrokenWriter(),
        io.BytesIO(),
        correlation_factory=lambda: CORRELATION_ID,
    ) == 1
    assert cli.run(
        ["solve"],
        io.BytesIO(b"bad"),
        io.BytesIO(),
        BrokenWriter(),
        correlation_factory=lambda: CORRELATION_ID,
    ) == 1


def test_console_and_module_entry_points_are_declared():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'poker-knight-ng = "poker_knight_ng.cli:main"' in pyproject
    assert (ROOT / "src/poker_knight_ng/__main__.py").is_file()

    request_bytes = canonical(raw_request(trials="1"))
    commands = (
        [sys.executable, "-m", "poker_knight_ng", "solve"],
        [sys.executable, "-m", "poker_knight_ng.cli", "solve"],
    )
    results = [
        subprocess.run(command, input=request_bytes, capture_output=True)
        for command in commands
    ]
    assert all(result.returncode == 0 for result in results)
    payloads = [json.loads(result.stdout) for result in results]
    durations = [payload["timing"].pop("total_duration_ns") for payload in payloads]
    assert all(value.isdecimal() for value in durations)
    assert payloads[0] == payloads[1]
    assert results[0].stderr == results[1].stderr == b""
