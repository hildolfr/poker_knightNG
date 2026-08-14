"""Test-only lifecycle predicate for the unqualified Phase 6C CUDA candidate."""
from __future__ import annotations

import hashlib
from pathlib import Path

CANDIDATE_CUDA_SOURCE_SHA256 = "8da8349bed65e782a18d29f83de884341b3838f40c1e83904d07860c2c4ade5a"
CUDA_SOURCE_NAMES = (
    "philox.cuh",
    "dealer.cuh",
    "cards.cuh",
    "evaluator.cuh",
    "simulate.cuh",
    "reduce.cuh",
    "deterministic_kernels.cu",
)


def cuda_source_digest(root: Path) -> str | None:
    """Return the exact approved closure digest, or None for any unreadable member."""
    directory = root / "src/poker_knight_ng/cuda-sources"
    digest = hashlib.sha256()
    try:
        for name in CUDA_SOURCE_NAMES:
            digest.update(name.encode("ascii"))
            digest.update(b"\0")
            digest.update((directory / name).read_bytes())
    except OSError:
        return None
    return digest.hexdigest()


def candidate_authority_pending(root: Path) -> bool:
    """True only for the exact unqualified candidate CUDA source closure."""
    return cuda_source_digest(root) == CANDIDATE_CUDA_SOURCE_SHA256
