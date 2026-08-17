"""Phase 7B service-package isolation and CI authority."""
from __future__ import annotations

import ast
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "service"
ADR = ROOT / "docs/adr/0006-isolated-http-service-package.md"
ROADMAP = ROOT / "docs/roadmap-status.md"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/04/4b/"
    "29cac41a4d98d144bf5f6d33995617b185d14b22401f75ca86f384e87ff1/"
    "h11-0.16.0-py3-none-any.whl"
)
WHEEL_SHA256 = "63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86"


def test_service_package_owns_exact_h11_without_root_dependency_drift() -> None:
    root_project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    service_project = tomllib.loads((SERVICE / "pyproject.toml").read_text("utf-8"))
    root_lock = (ROOT / "uv.lock").read_text("utf-8")
    service_lock = (SERVICE / "uv.lock").read_text("utf-8")

    assert root_project["project"]["dependencies"] == []
    assert 'name = "h11"' not in root_lock
    assert service_project["project"]["dependencies"] == [
        f"h11 @ {WHEEL_URL}",
        "poker-knight-ng==0.1.0",
    ]
    assert service_project["tool"]["uv"]["sources"] == {
        "poker-knight-ng": {"path": "..", "editable": True}
    }
    assert f'source = {{ url = "{WHEEL_URL}" }}' in service_lock
    assert f'hash = "sha256:{WHEEL_SHA256}"' in service_lock
    assert 'name = "poker-knight-ng"' in service_lock
    assert 'source = { editable = "../" }' in service_lock


def test_ci_verifies_frozen_service_environment_and_tests() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text("utf-8")

    assert "Verify bounded service package" in workflow
    assert "uv sync --project service --frozen --group dev" in workflow
    assert "uv run --project service --frozen pytest -q service/tests" in workflow
    assert "uv build --out-dir ci-service-dist" in workflow
    assert "uv build --project service --out-dir ci-service-dist" in workflow
    assert 'engine_wheel="$(printf \'%s\\n\' ci-service-dist/poker_knight_ng-*.whl)"' in workflow
    assert 'h11_wheel="$RUNNER_TEMP/h11-0.16.0-py3-none-any.whl"' in workflow
    assert WHEEL_URL in workflow
    assert WHEEL_SHA256 in workflow
    assert "UV_CACHE_DIR=\"$RUNNER_TEMP/phase7b-service-empty-cache\"" in workflow
    assert "uv pip install --offline --no-deps" in workflow
    assert '"$h11_wheel" "$engine_wheel" "$service_wheel"' in workflow
    assert 'uv pip check --python "$service_venv/bin/python"' in workflow
    assert (
        '"$service_venv/bin/python" -I -c "import '
        'poker_knight_ng_service.adapter, poker_knight_ng_service.admission, '
        'poker_knight_ng_service.async_execution, '
        'poker_knight_ng_service.connection, '
        'poker_knight_ng_service.execution, '
        'poker_knight_ng_service.framing, '
        'poker_knight_ng_service.identity, '
        'poker_knight_ng_service.listener, '
        'poker_knight_ng_service.responses, poker_knight_ng_service.routing, '
        'poker_knight_ng_service.stream_adapter, '
        'poker_knight_ng_service.session"'
        in workflow
    )


def test_packaging_adr_and_roadmap_preserve_no_listener_boundary() -> None:
    adr = ADR.read_text("utf-8")
    roadmap = ROADMAP.read_text("utf-8")
    phase7b = next(line for line in roadmap.splitlines() if line.startswith("| Phase 7B"))

    assert "Status: accepted" in adr
    assert "separate Python distribution" in adr
    assert "root engine distribution and root lock remain byte-unchanged" in adr
    assert "No listener, socket activation, engine invocation or deployment" in adr
    assert "08f97ac10337f945eaf1bf993be3b0faaa5a8955" in phase7b
    assert "31864664096" in phase7b
    assert "eea11831afe62674be7ec7572b264fb351c74461" in phase7b
    assert "31889215662" in phase7b
    assert "1c9a30b401c55b71a0788729bf2f67f061a65295" in phase7b
    assert "31890879197" in phase7b
    assert "b119c3d800ad667f2a82713799ba7f408772c085" in phase7b
    assert "31892394345" in phase7b
    assert "6296c808573b80f22b78dfd7a82f42237d070b7c" in phase7b
    assert "31894370720" in phase7b
    assert "509a0734ca0983bd360c539a17b3fc52c858ae13" in phase7b
    assert "31896238833" in phase7b
    assert "5e56547586f031a453fa00c8f9d6f558bf97546b" in phase7b
    assert "31897788088" in phase7b
    assert "b9fead1bdc645acadf35831e6a9c5eccdf288ee5" in phase7b
    assert "31901512049" in phase7b
    assert "99ab51d661f5226bd1f5ba961fa57c7bfbc707f4" in phase7b
    assert "31906009703" in phase7b
    assert "52a8e8fcce863406dcc5a1adf8bd7e9536c36b91" in phase7b
    assert "31908384542" in phase7b
    assert "ADR 0006" in phase7b
    assert "ADR 0007" in phase7b
    assert "incremental reader" in phase7b
    assert "raw framing" in phase7b
    assert "route and response-envelope" in phase7b
    assert "semantic request adapter" in phase7b
    assert "cancellation-safe async solve handoff" in phase7b
    assert "listener-free one-request session coordinator" in phase7b
    assert "weak-identity ownership" in phase7b
    assert "bounded Unix-listener construction authority" in phase7b
    assert "three-wheel service boundary" in phase7b
    assert "one-solve admission" in phase7b
    assert "zero-queue" in phase7b
    assert "engine-execution checkpoint" in phase7b
    assert "engine-construction-inert" in phase7b
    assert "listener" in phase7b.lower()


def test_phase7b_source_has_exact_engine_authority_and_no_listener() -> None:
    expected_imports = {
        "__init__.py": (),
        "adapter.py": (
            "from __future__ import annotations",
            "import json",
            "from dataclasses import dataclass",
            "from typing import Any",
            "from poker_knight_ng.contract.errors import problem",
            "from poker_knight_ng.contract.models import EquityRequest",
            "from .framing import AdmittedRequest",
            "from .routing import Route, select_route",
        ),
        "admission.py": (
            "from __future__ import annotations",
            "from collections.abc import Callable",
            "from threading import Lock",
            "from poker_knight_ng.contract.errors import problem",
        ),
        "async_execution.py": (
            "from __future__ import annotations",
            "import asyncio",
            "from threading import Event, Thread",
            "from poker_knight_ng.contract.errors import ContractProblem, problem",
            "from .adapter import AdaptedSolveRequest",
            "from .admission import SolveLease, admit_solve",
            "from .execution import _execute_admitted, _trusted_snapshot",
        ),
        "connection.py": (
            "from __future__ import annotations",
            "import asyncio",
            "from time import monotonic",
            "from typing import Protocol",
            "from .framing import AdmittedRequest, TransportFailure, _inspect_request_head, admit_request",
        ),
        "execution.py": (
            "from __future__ import annotations",
            "from poker_knight_ng.contract import ContractProblem, EquityRequest, serialize_equity_result",
            "from poker_knight_ng.contract.errors import problem",
            "from poker_knight_ng.engine import CPUReferenceEngine, CUDAEngine",
            "from .adapter import AdaptedSolveRequest",
            "from .admission import SolveLease, admit_solve",
            "from .routing import Route",
        ),
        "identity.py": (
            "from __future__ import annotations",
            "from grp import getgrnam as _getgrnam",
            "from pwd import getpwnam as _getpwnam",
            "from weakref import WeakKeyDictionary",
        ),
        "framing.py": (
            "from __future__ import annotations",
            "from dataclasses import dataclass",
            "import h11",
        ),
        "listener.py": (
            "from __future__ import annotations",
            "import asyncio",
            "import errno",
            "import fcntl",
            "import os",
            "import socket",
            "import stat",
            "from dataclasses import dataclass",
            "from typing import Protocol",
            "from .identity import ResolvedServiceIdentity, _identity_values",
        ),
        "responses.py": (
            "from __future__ import annotations",
            "import json",
            "import re",
            "import secrets",
            "from collections.abc import Callable",
            "from .framing import TransportFailure, serialize_response",
        ),
        "routing.py": (
            "from __future__ import annotations",
            "from enum import Enum",
            "from .framing import AdmittedRequest, TransportFailure",
        ),
        "session.py": (
            "from __future__ import annotations",
            "import json",
            "from collections.abc import Callable",
            "from threading import Lock",
            "from typing import Protocol",
            "from weakref import ReferenceType, ref",
            "from poker_knight_ng.contract.errors import ContractProblem, PROBLEM_POLICIES, problem",
            "from .adapter import adapt_solve_request",
            "from .async_execution import execute_solve_async",
            "from .connection import AsyncReader, read_admitted_request",
            "from .framing import AdmittedRequest, TransportFailure",
            "from .responses import EMERGENCY_REQUEST_ID, RequestIdGenerationFailure, generate_request_id, serialize_health_response, serialize_json_response, serialize_transport_failure",
            "from .routing import Route, select_route",
        ),
        "stream_adapter.py": (
            "from __future__ import annotations",
            "import asyncio",
            "import errno",
        ),
    }
    observed_imports: dict[str, tuple[str, ...]] = {}
    forbidden_calls: list[str] = []
    engine_calls: list[tuple[str, str]] = []
    async_calls: list[tuple[str, str]] = []
    dunder_accesses: list[tuple[str, str]] = []
    forbidden_name_calls = {
        "__import__",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "locals",
        "setattr",
        "solve",
        "solve_cuda",
        "start_next_cycle",
        "vars",
    }
    forbidden_attribute_calls = {
        "call_soon_threadsafe",
        "create_task",
        "getattr",
        "import_module",
        "run_in_executor",
        "shield",
        "solve_cuda",
        "start_next_cycle",
        "to_thread",
    }
    for path in sorted((SERVICE / "src/poker_knight_ng_service").glob("*.py")):
        tree = ast.parse(path.read_text("utf-8"))
        observed_imports[path.name] = tuple(
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        )
        forbidden_calls.extend(
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in forbidden_name_calls
        )
        forbidden_calls.extend(
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden_attribute_calls
        )
        engine_calls.extend(
            (path.name, node.func.id)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"CPUReferenceEngine", "CUDAEngine"}
        )
        engine_calls.extend(
            (path.name, ast.unparse(node.func))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "solve"
        )
        async_calls.extend(
            (path.name, node.func.id)
            for node in ast.walk(tree)
            if path.name == "async_execution.py"
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Thread"
        )
        async_calls.extend(
            (path.name, ast.unparse(node.func))
            for node in ast.walk(tree)
            if path.name == "async_execution.py"
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"is_alive", "is_set", "join", "set", "sleep", "start"}
        )
        dunder_accesses.extend(
            (ast.unparse(node.value), node.attr)
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and node.attr.endswith("__")
        )

    assert observed_imports == expected_imports
    assert forbidden_calls == []
    assert engine_calls == [
        ("execution.py", "CPUReferenceEngine"),
        ("execution.py", "CUDAEngine"),
        ("execution.py", "engine.solve"),
    ]
    assert async_calls == [
        ("async_execution.py", "Thread"),
        ("async_execution.py", "worker.join"),
        ("async_execution.py", "worker.start"),
        ("async_execution.py", "done.set"),
        ("async_execution.py", "done.is_set"),
        ("async_execution.py", "worker.join"),
        ("async_execution.py", "worker.is_alive"),
        ("async_execution.py", "done.is_set"),
        ("async_execution.py", "asyncio.sleep"),
        ("async_execution.py", "worker.is_alive"),
    ]
    assert dunder_accesses == [
        ("object", "__new__"),
        ("object", "__setattr__"),
        ("object", "__setattr__"),
        ("object", "__getattribute__"),
        ("object", "__getattribute__"),
        ("object", "__getattribute__"),
        ("object", "__getattribute__"),
        ("object", "__new__"),
        ("object", "__getattribute__"),
        ("object", "__getattribute__"),
        ("object", "__getattribute__"),
        ("super()", "__init__"),
        ("object", "__new__"),
        ("type(cleanup_failure)", "__name__"),
        ("super()", "__init__"),
    ]
