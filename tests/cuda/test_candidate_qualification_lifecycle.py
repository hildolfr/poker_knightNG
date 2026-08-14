from __future__ import annotations

from pathlib import Path
import importlib.util
import shutil

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("_candidate_lifecycle_hostile", ROOT / "tools/candidate_qualification_lifecycle.py")
assert SPEC is not None and SPEC.loader is not None
LIFECYCLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIFECYCLE)


def test_candidate_lifecycle_requires_exact_cuda_source_closure(tmp_path: Path) -> None:
    source = ROOT / "src/poker_knight_ng/cuda-sources"
    destination = tmp_path / "src/poker_knight_ng/cuda-sources"
    destination.mkdir(parents=True)
    for name in LIFECYCLE.CUDA_SOURCE_NAMES:
        shutil.copyfile(source / name, destination / name)

    assert LIFECYCLE.cuda_source_digest(tmp_path) == LIFECYCLE.CANDIDATE_CUDA_SOURCE_SHA256
    assert LIFECYCLE.candidate_authority_pending(tmp_path)

    evaluator = destination / "evaluator.cuh"
    evaluator.write_bytes(evaluator.read_bytes() + b"\n// hostile drift\n")
    assert LIFECYCLE.cuda_source_digest(tmp_path) != LIFECYCLE.CANDIDATE_CUDA_SOURCE_SHA256
    assert not LIFECYCLE.candidate_authority_pending(tmp_path)

    evaluator.unlink()
    assert LIFECYCLE.cuda_source_digest(tmp_path) is None
    assert not LIFECYCLE.candidate_authority_pending(tmp_path)
