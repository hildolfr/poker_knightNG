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


def test_phase7b_source_preserves_engine_and_listener_authority_boundaries() -> None:
    """Assert policy-relevant authority boundaries, not incidental AST order."""
    package = SERVICE / "src/poker_knight_ng_service"
    imports: dict[str, set[str]] = {}
    forbidden_calls: list[str] = []
    engine_calls: list[tuple[str, str]] = []
    thread_calls: list[tuple[str, str]] = []
    forbidden_names = {"__import__", "compile", "delattr", "eval", "exec", "getattr", "globals", "locals", "setattr", "solve", "solve_cuda", "start_next_cycle", "vars"}
    forbidden_attributes = {"call_soon_threadsafe", "create_task", "getattr", "import_module", "run_in_executor", "shield", "solve_cuda", "start_next_cycle", "to_thread"}
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text("utf-8"))
        imports[path.name] = {ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                if node.func.id in forbidden_names:
                    forbidden_calls.append(f"{path.name}:{node.lineno}:{node.func.id}")
                if node.func.id in {"CPUReferenceEngine", "CUDAEngine"}:
                    engine_calls.append((path.name, node.func.id))
                if path.name == "async_execution.py" and node.func.id == "Thread":
                    thread_calls.append((path.name, node.func.id))
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in forbidden_attributes:
                    forbidden_calls.append(f"{path.name}:{node.lineno}:{node.func.attr}")
                if node.func.attr == "solve":
                    engine_calls.append((path.name, ast.unparse(node.func)))
    assert forbidden_calls == []
    assert "from poker_knight_ng.engine import CPUReferenceEngine, CUDAEngine" in imports["execution.py"]
    assert all("listener" not in item.lower() for item in imports["execution.py"])
    assert all("poker_knight_ng.engine" not in item and "listener" not in item.lower() for item in imports["adapter.py"])
    assert set(engine_calls) == {("execution.py", "CPUReferenceEngine"), ("execution.py", "CUDAEngine"), ("execution.py", "engine.solve")}
    assert len(engine_calls) == 3
    assert thread_calls == [("async_execution.py", "Thread")]
