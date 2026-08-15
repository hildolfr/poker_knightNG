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
    assert service_project["project"]["dependencies"] == [f"h11 @ {WHEEL_URL}"]
    assert f'source = {{ url = "{WHEEL_URL}" }}' in service_lock
    assert f'hash = "sha256:{WHEEL_SHA256}"' in service_lock


def test_ci_verifies_frozen_service_environment_and_tests() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text("utf-8")

    assert "Verify bounded service package" in workflow
    assert "uv sync --project service --frozen --group dev" in workflow
    assert "uv run --project service --frozen pytest -q service/tests" in workflow
    assert "uv build --project service --out-dir ci-service-dist" in workflow
    assert 'uv pip install --python "$service_venv/bin/python" "$service_wheel"' in workflow
    assert '"$service_venv/bin/python" -I -c "import poker_knight_ng_service.framing"' in workflow


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
    assert "ADR 0006" in phase7b
    assert "raw framing" in phase7b
    assert "no listener" in phase7b.lower()


def test_phase7b_a_source_has_no_listener_or_engine_adapter() -> None:
    source = (SERVICE / "src/poker_knight_ng_service/framing.py").read_text("utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert "socket" not in imported_roots
    assert "asyncio" not in imported_roots
    assert "poker_knight_ng" not in imported_roots
    assert "start_next_cycle" not in source
