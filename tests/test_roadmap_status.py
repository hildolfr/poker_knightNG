"""Repository roadmap publication contract."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
ROADMAP = ROOT / "docs" / "roadmap-status.md"


def test_roadmap_status_is_published_and_authority_scoped() -> None:
    text = ROADMAP.read_text("utf-8")

    assert "validation/holdem/v1/SPEC.md" in text
    assert "TODO.md" in text and "non-authoritative" in text
    assert "Phase 7A" in text and "Not started" in text
    assert "network service" in text
    assert "automatic CUDA routing" in text
    assert "`main`" in text
    assert "15e49a5e8d88bcca6395ec07c02aacf388996ac4" in text


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
