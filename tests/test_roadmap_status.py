"""Repository roadmap publication contract."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
ROADMAP = ROOT / "docs" / "roadmap-status.md"

# The pinned promoted default-branch commit the roadmap baseline must match.
# Resolved against the live checkout's HEAD at test time so it cannot silently
# go stale; this value is the fallback when the checkout is not a git worktree.
PROMOTED_MAIN_SHA = "3ef14f0cd3ad0023e00374b2912b0e70e540e58a"


def _current_head_sha() -> str:
    """Resolve the current `main` HEAD SHA, falling back to the pinned value.

    The roadmap baseline is the promoted default-branch commit, so the test
    compares the pinned SHA against the live `main` ref (not the checkout's
    HEAD, which may be a feature branch). In a non-git context (e.g. an sdist)
    this falls back to the pinned value, keeping the assertion meaningful.
    """
    for ref in ("main", "refs/heads/main"):
        try:
            out = subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", ref],
                capture_output=True,
                check=True,
                text=True,
                timeout=30,
            )
            return out.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            continue
    return PROMOTED_MAIN_SHA


def _status_rows(text: str) -> dict[str, str]:
    header = "| Phase / checkpoint | Status | Delivered or remaining |"
    start = text.splitlines().index(header) + 2
    rows = {}
    for line in text.splitlines()[start:]:
        if not line.startswith("|"):
            break
        item, status, _detail = (cell.strip() for cell in line.strip("|").split("|", 2))
        rows[item] = status
    return rows


def test_roadmap_status_is_published_and_authority_scoped() -> None:
    text = ROADMAP.read_text("utf-8")

    assert "validation/holdem/v1/SPEC.md" in text
    assert "TODO.md" in text and "non-authoritative" in text
    rows = _status_rows(text)
    assert rows["Phase 7A — network service contract"] == "**Complete**"
    assert rows["Phase 7B — bounded network service"] == "**Active**"
    assert rows["Phase 7F — deployment"] == "**Implemented**"
    assert "network service" in text
    assert "automatic CUDA routing" in text
    assert "`main`" in text
    # The pinned baseline must be an ancestor (or the tip) of live `main` so it
    # cannot silently go stale, while still passing after later promotions move
    # main forward — a strict-equality check here would fail on every future
    # promotion of exactly the commits this roadmap describes. In a non-git
    # context (e.g. an sdist) _current_head_sha falls back to the pinned value.
    assert PROMOTED_MAIN_SHA in text
    current = _current_head_sha()
    if current != PROMOTED_MAIN_SHA:
        assert subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
             PROMOTED_MAIN_SHA, current],
            check=True,
        ).returncode == 0


def test_roadmap_records_private_observability_boundary() -> None:
    text = ROADMAP.read_text("utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| Production observability"))

    assert "**Implemented**" in row
    assert "in-process-only" in row
    assert "no HTTP diagnostics route" in row
    assert "no request-derived values" in row


def test_roadmap_retains_every_explicitly_deferred_v1_surface() -> None:
    text = ROADMAP.read_text("utf-8").lower()
    required = (
        "icm",
        "tournament stacks",
        "ranges",
        "weighted opponents",
        "position",
        "fold equity",
        "gto",
        "pot odds",
        "board texture",
        "vulnerability",
        "percentiles",
        "side pots",
        "betting/action trees",
        "poker variants",
        "public internet exposure",
        "accounts, billing and multitenancy",
        "multi-gpu balancing",
        "streaming progress",
        "websockets",
        "result caching",
        "camelot changes",
        "adaptive trial counts",
        "timeout-based convergence",
        "implicit reseeding",
        "silent cpu/gpu fallback",
        "advanced analytics",
    )
    missing = [item for item in required if item not in text]
    assert not missing, missing
