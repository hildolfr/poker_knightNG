"""Promotion and release-readiness documentation contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
RELEASE = ROOT / "docs" / "release-process.md"
ROADMAP = ROOT / "docs" / "roadmap-status.md"
PROTECTION = ROOT / "docs" / "evidence" / "main-branch-protection.json"
PROTECTION_MANIFEST = ROOT / "docs" / "evidence" / "main-branch-protection.sha256"


def roadmap_rows() -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for line in ROADMAP.read_text("utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 3 and cells[0] not in {"Item", "---"}:
            rows[cells[0]] = (cells[1], cells[2])
    return rows


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
    rows = roadmap_rows()
    promotion_status, promotion_detail = rows["Promotion to default `main`"]
    assert promotion_status == "**Complete**"
    assert "15e49a5e8d88bcca6395ec07c02aacf388996ac4" in promotion_detail
    assert "actions/runs/31859196286" in promotion_detail
    protection_status, protection_detail = rows["Main branch protection"]
    assert protection_status == "**Active**"
    assert "evidence/main-branch-protection.json" in protection_detail
    assert rows["Automated CPU CI"][0] == "**Active**"
    assert rows["Release procedure"][0] == "**Implemented**"
    assert rows["GitHub release"][0] == "**Untouched**"


def test_main_branch_protection_evidence_is_canonical_closed_and_manifest_bound() -> None:
    raw = PROTECTION.read_bytes()
    value = json.loads(raw)
    assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert set(value) == {
        "branch", "captured_at", "format_version", "head_sha", "private_response_sha256",
        "protection", "provider", "repository",
    }
    assert value["format_version"] == "github-branch-protection-v1"
    assert value["repository"] == "hildolfr/poker_knightNG"
    assert value["branch"] == "main"
    assert value["head_sha"] == "15e49a5e8d88bcca6395ec07c02aacf388996ac4"
    assert value["protection"] == {
        "allow_deletions": False,
        "allow_force_pushes": False,
        "enforce_admins": False,
        "required_linear_history": True,
        "required_pull_request_reviews": False,
        "required_status_checks": {"contexts": ["verify"], "strict": True},
    }
    digest = hashlib.sha256(raw).hexdigest()
    assert PROTECTION_MANIFEST.read_text("ascii") == f"{digest}  {PROTECTION.name}\n"
    assert re.fullmatch(r"[0-9a-f]{64}", value["private_response_sha256"])
