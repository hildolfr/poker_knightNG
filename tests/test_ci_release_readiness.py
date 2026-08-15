"""Promotion and release-readiness documentation contracts."""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
RELEASE = ROOT / "docs" / "release-process.md"
ROADMAP = ROOT / "docs" / "roadmap-status.md"


def test_ci_is_bounded_pinned_and_preserves_historical_git_authority() -> None:
    text = WORKFLOW.read_text("utf-8")
    assert "permissions:\n  contents: read" in text
    assert "timeout-minutes:" in text
    assert "fetch-depth: 0" in text
    assert "uv sync --frozen --group dev" in text
    assert "uv run pytest -q" in text
    assert "tools/generate_rng_seed_bank.py --verify" in text
    assert "tools/verify_cuda_release_qualification.py" in text
    assert "tools/verify_cuda_statistical_release_qualification.py" in text
    assert "tools/project_cuda_benchmark_baseline.py verify" in text
    assert "uv build --out-dir" in text
    action_refs = re.findall(r"uses:\s*[^@\s]+@([^\s#]+)", text)
    assert action_refs and all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs)


def test_release_process_is_fail_closed_and_history_preserving() -> None:
    text = RELEASE.read_text("utf-8")
    lowered = text.lower()
    assert "explicit owner approval" in lowered
    assert re.search(r"never\s+squash", lowered)
    assert re.search(r"never\s+rebase", lowered)
    assert "fast-forward" in lowered
    assert "private evidence" in lowered
    assert "pypi" in lowered and "separate approval" in lowered
    assert "rollback" in lowered


def test_roadmap_tracks_ci_and_promotion_as_distinct_states() -> None:
    text = ROADMAP.read_text("utf-8")
    assert "Automated CPU CI" in text
    assert "Promotion to default `main`" in text
    assert "GitHub release" in text
