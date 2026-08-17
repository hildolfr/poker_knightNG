"""Repository roadmap publication contract."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
ROADMAP = ROOT / "docs" / "roadmap-status.md"


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
    assert "15e49a5e8d88bcca6395ec07c02aacf388996ac4" in text


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
