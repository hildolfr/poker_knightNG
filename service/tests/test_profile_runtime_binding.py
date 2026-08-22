"""Cross-bind the frozen v1 service profile to the runtime constants that mirror it.

This test loads ``contracts/service/v1/http-service-profile.json`` (and the
supplementary ``unix-listener-construction.json``) and asserts that every
runtime constant which is *supposed* to mirror a profile value still equals it.
The point is to turn the previous "conformant by inspection only" state into an
automated, loud gate: if a runtime constant drifts from the frozen profile, the
test fails with a diff-style message showing the exact mismatch.

Only ``service/tests/**`` and ``tests/**`` are touched by this lane; the runtime
modules themselves are owned by other lanes and must not be edited here.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

SERVICE_SRC = (
    Path(__file__).resolve().parents[1] / "src" / "poker_knight_ng_service"
)
PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "contracts" / "service" / "v1" / "http-service-profile.json"
)
LISTENER_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "contracts" / "service" / "v1" / "unix-listener-construction.json"
)


def _load_profile() -> dict:
    with PROFILE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_listener_profile() -> dict:
    with LISTENER_PROFILE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _diff_message(label: str, expected: object, actual: object) -> str:
    return (
        f"\nPROFILE/RUNTIME DRIFT: {label}\n"
        f"  profile expects: {expected!r}\n"
        f"  runtime  actual: {actual!r}"
    )


def _assert_eq(label: str, expected: object, actual: object) -> None:
    assert expected == actual, _diff_message(label, expected, actual)


_FRAMING_LIMIT_MARKERS: dict[int, str] = {
    # Marker value -> limit key. The framing module compares against these
    # exact integer literals in its guard conditions.
    8192: "header_total_bytes",
    32: "header_count",
    128: "header_name_bytes",
    1024: "header_value_bytes",
    16384: "max_body_bytes",
}


def _framing_literals() -> dict[str, int]:
    """Extract the numeric framing limits enforced by ``framing.py`` source.

    The framing module hard-codes these limits inline (header total bytes,
    header count, header name bytes, header value bytes, max body bytes).
    Parsing the source AST binds the profile to the *actual* enforced numbers
    rather than a hand-copied copy that could silently drift.
    """
    source = (SERVICE_SRC / "framing.py").read_text("utf-8")
    tree = ast.parse(source)
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not isinstance(node.ops[0], ast.Gt):
            continue
        comparator = node.comparators[0]
        if not isinstance(comparator, ast.Constant):
            continue
        # max_body_bytes is enforced against a bytes literal (b"16384").
        if isinstance(comparator.value, bytes) and comparator.value.isdigit():
            found.setdefault("max_body_bytes", int(comparator.value))
            continue
        if not isinstance(comparator.value, int):
            continue
        key = _FRAMING_LIMIT_MARKERS.get(comparator.value)
        if key is not None:
            found.setdefault(key, comparator.value)
    return found


def test_profile_binds_to_runtime_constants() -> None:
    import poker_knight_ng_service.connection as connection
    import poker_knight_ng_service.listener as listener
    import poker_knight_ng_service.routing as routing
    import poker_knight_ng_service.runtime as runtime

    profile = _load_profile()
    listener_profile = _load_listener_profile()
    limits = profile["limits"]
    socket = profile["socket"]

    # -- sessions -------------------------------------------------------------
    _assert_eq(
        "limits.max_connections == runtime.DEFAULT_MAX_SESSIONS",
        limits["max_connections"],
        runtime.DEFAULT_MAX_SESSIONS,
    )

    # -- graceful drain -------------------------------------------------------
    drain_ms = listener_profile["l3_shutdown"]["connection_drain_deadline_ms"]
    _assert_eq(
        "l3_shutdown.connection_drain_deadline_ms/1000 == _DEFAULT_GRACEFUL_DRAIN_SECONDS",
        drain_ms / 1000,
        runtime._DEFAULT_GRACEFUL_DRAIN_SECONDS,
    )
    _assert_eq(
        "execution.shutdown == 'stop-admission-and-drain'",
        "stop-admission-and-drain",
        profile["execution"]["shutdown"],
    )

    # -- framing limits (extracted from the runtime source AST) ---------------
    framing_limits = _framing_literals()
    for key, profile_key in (
        ("header_total_bytes", "header_total_bytes"),
        ("header_count", "header_count"),
        ("header_name_bytes", "header_name_bytes"),
        ("header_value_bytes", "header_value_bytes"),
        ("max_body_bytes", "max_body_bytes"),
    ):
        assert key in framing_limits, f"runtime framing limit {key} not found in framing.py"
        _assert_eq(
            f"limits.{profile_key} == framing enforcement ({key})",
            limits[profile_key],
            framing_limits[key],
        )
    # header_total_bytes is also mirrored by connection._HEADER_LIMIT
    _assert_eq(
        "limits.header_total_bytes == connection._HEADER_LIMIT",
        limits["header_total_bytes"],
        connection._HEADER_LIMIT,
    )

    # -- timeouts -------------------------------------------------------------
    _assert_eq(
        "limits.body_read_timeout_ms/1000 == connection._READ_TIMEOUT_SECONDS",
        limits["body_read_timeout_ms"] / 1000,
        connection._READ_TIMEOUT_SECONDS,
    )
    _assert_eq(
        "limits.header_read_timeout_ms == limits.body_read_timeout_ms",
        limits["header_read_timeout_ms"],
        limits["body_read_timeout_ms"],
    )

    # -- socket path and modes ------------------------------------------------
    _assert_eq(
        "socket.default_path == listener._SOCKET_PATH",
        socket["default_path"],
        listener._SOCKET_PATH,
    )
    _assert_eq(
        "socket.socket_mode == oct(listener._SOCKET_MODE)",
        socket["socket_mode"],
        format(listener._SOCKET_MODE, "04o"),
    )
    _assert_eq(
        "socket.directory_mode == oct(listener._PARENT_MODE)",
        socket["directory_mode"],
        format(listener._PARENT_MODE, "04o"),
    )

    # -- routes ---------------------------------------------------------------
    expected_routes = {
        route["path"]: route["method"]
        for route in profile["routes"]
    }
    runtime_routes = {
        path.decode("ascii"): method.decode("ascii")
        for path, (method, _) in routing._ROUTES.items()
    }
    _assert_eq("profile.routes == routing._ROUTES", expected_routes, runtime_routes)


def test_profile_problem_mapping_keys_are_valid_runtime_problem_codes() -> None:
    from poker_knight_ng.contract.errors import PROBLEM_POLICIES

    profile = _load_profile()
    mapping = profile["responses"]["problem_mapping"]

    # The profile maps semantic failure keys to canonical problem codes. Every
    # referenced code must exist in the runtime's frozen problem-policy set.
    for semantic_key, code in mapping.items():
        assert (
            code in PROBLEM_POLICIES
        ), f"profile problem_mapping[{semantic_key!r}] references unknown problem code {code!r}"

    # The set of semantic keys is a frozen, known contract.
    expected_keys = {
        "cuda-route-cpu-backend",
        "engine-or-adapter-failure",
        "service-trial-cap",
        "solve-capacity-busy",
        "solve-route-invalid-backend",
    }
    _assert_eq(
        "profile.responses.problem_mapping keys == frozen semantic set",
        expected_keys,
        set(mapping),
    )

    # Each transport-failure status is within the runtime's allowed transport set.
    from poker_knight_ng_service.responses import _TRANSPORT_STATUSES

    for semantic_key, status in profile["responses"]["transport_failures"].items():
        assert (
            status in _TRANSPORT_STATUSES
        ), f"transport_failures[{semantic_key!r}] status {status} not in runtime transport set"
