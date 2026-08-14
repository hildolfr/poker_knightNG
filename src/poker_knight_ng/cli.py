"""Bounded local JSON CLI with explicit CPU and CUDA routes."""

from collections.abc import Callable, Sequence
import json
import secrets
import sys
from typing import BinaryIO

from .contract import (
    ContractProblem,
    EquityRequest,
    serialize_equity_result,
)
from .contract.errors import problem
from .engine import solve, solve_cuda

MAX_STDIN_BYTES = 16_384
EMERGENCY_CORRELATION_ID = "pk_" + "0" * 32
HELP = (
    b"Poker Knight NG deterministic equity CLI\n"
    b"usage: poker-knight-ng {solve|solve-cuda}\n"
    b"stdin: one bounded v1 EquityRequest JSON object\n"
)
_RETRYABLE_OPERATIONAL = frozenset(
    {"BACKEND_UNAVAILABLE", "RESOURCE_EXHAUSTED"}
)
_EXECUTION_FAILURE = frozenset(
    {"INTERNAL_ERROR", "RNG_REJECTION_EXHAUSTED"}
)


class _InputError(Exception):
    """Private input-framing marker; no submitted values are retained."""


def _correlation_id() -> str:
    return "pk_" + secrets.token_hex(16)


def _duplicate_free(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _InputError
        value[key] = item
    return value


def _reject_constant(_value: str) -> object:
    raise _InputError


def _parse_request_bytes(payload: bytes) -> EquityRequest:
    if type(payload) is not bytes or not payload or len(payload) > MAX_STDIN_BYTES:
        raise _InputError
    try:
        text = payload.decode("utf-8")
        if text.startswith("\ufeff"):
            raise _InputError
        raw = json.loads(
            text,
            object_pairs_hook=_duplicate_free,
            parse_constant=_reject_constant,
        )
    except _InputError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise _InputError from exc
    if type(raw) is not dict:
        raise _InputError
    return EquityRequest.parse(raw)


def _canonical_json(value: object) -> bytes:
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


def _problem_exit(code: str) -> int:
    if code in _RETRYABLE_OPERATIONAL:
        return 4
    if code in _EXECUTION_FAILURE:
        return 5
    return 3


def _write(stream: BinaryIO, payload: bytes) -> bool:
    try:
        written = stream.write(payload)
        if written is not None and written != len(payload):
            return False
        flush = getattr(stream, "flush", None)
        if flush is not None:
            flush()
        return True
    except Exception:
        return False


def _emit_problem(
    failure: ContractProblem,
    stderr: BinaryIO,
    correlation_factory: Callable[[], str],
    *,
    exit_code: int | None = None,
) -> int:
    selected = failure
    selected_exit = _problem_exit(failure.code) if exit_code is None else exit_code
    try:
        correlation_id = correlation_factory()
        payload = selected.serialize(correlation_id)
        encoded = _canonical_json(payload)
    except Exception:
        selected = problem("INTERNAL_ERROR")
        selected_exit = 5
        encoded = _canonical_json(
            selected.serialize(EMERGENCY_CORRELATION_ID)
        )
    if not _write(stderr, encoded):
        return 1
    return selected_exit


def run(
    argv: Sequence[str],
    stdin: BinaryIO,
    stdout: BinaryIO,
    stderr: BinaryIO,
    *,
    correlation_factory: Callable[[], str] = _correlation_id,
) -> int:
    """Execute one bounded CLI invocation without catching BaseException."""
    if list(argv) == ["--help"]:
        return 0 if _write(stdout, HELP) else 1
    if list(argv) not in (["solve"], ["solve-cuda"]):
        return _emit_problem(
            problem("UNSUPPORTED_REQUEST"),
            stderr,
            correlation_factory,
            exit_code=2,
        )

    try:
        payload = stdin.read(MAX_STDIN_BYTES + 1)
        request = _parse_request_bytes(payload)
    except _InputError:
        return _emit_problem(
            problem("UNSUPPORTED_REQUEST"),
            stderr,
            correlation_factory,
            exit_code=2,
        )
    except ContractProblem as exc:
        return _emit_problem(exc, stderr, correlation_factory)
    except Exception:
        return _emit_problem(
            problem("INTERNAL_ERROR"),
            stderr,
            correlation_factory,
        )

    try:
        if argv[0] == "solve":
            result = solve(request)
        else:
            result = solve_cuda(request)
        response = serialize_equity_result(result, request)
        encoded = _canonical_json(response)
    except ContractProblem as exc:
        return _emit_problem(exc, stderr, correlation_factory)
    except Exception:
        return _emit_problem(
            problem("INTERNAL_ERROR"),
            stderr,
            correlation_factory,
        )
    return 0 if _write(stdout, encoded) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point."""
    arguments = sys.argv[1:] if argv is None else argv
    return run(
        arguments,
        sys.stdin.buffer,
        sys.stdout.buffer,
        sys.stderr.buffer,
    )


if __name__ == "__main__":
    raise SystemExit(main())
