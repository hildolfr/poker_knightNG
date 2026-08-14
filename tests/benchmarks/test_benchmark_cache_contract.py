from pathlib import Path

import pytest

from test_benchmark_equity_worker import load_tool


def test_cold_cache_requires_a_new_directory_and_seals_exact_files(tmp_path):
    tool = load_tool()
    cache = tmp_path / "cache"
    tool.prepare_cold_cache(cache)
    assert cache.is_dir()
    assert (cache.stat().st_mode & 0o777) == 0o700
    assert list(cache.iterdir()) == []

    artifact = cache / "kernel.bin"
    artifact.write_bytes(b"compiled")
    manifest = tool.seal_cold_cache(cache, cold_result_sha256="a" * 64)
    assert manifest["cold_result_sha256"] == "a" * 64
    assert manifest["files"] == [{
        "path": "kernel.bin",
        "sha256": "64ebd267717810f9524a58c6d7715bd9502b3c9235b291a7104389b372aeec4b",
        "size_bytes": "8",
    }]
    assert tool.verify_warm_cache(cache) == manifest


@pytest.mark.parametrize("mutation", ["missing-marker", "modified", "extra", "symlink"])
def test_warm_cache_rejects_unsealed_or_changed_cache(tmp_path, mutation):
    tool = load_tool()
    cache = tmp_path / "cache"
    tool.prepare_cold_cache(cache)
    artifact = cache / "kernel.bin"
    artifact.write_bytes(b"compiled")
    tool.seal_cold_cache(cache, cold_result_sha256="a" * 64)

    if mutation == "missing-marker":
        (cache / tool.CACHE_SEAL_NAME).unlink()
    elif mutation == "modified":
        artifact.write_bytes(b"changed")
    elif mutation == "extra":
        (cache / "extra.bin").write_bytes(b"extra")
    else:
        (cache / "link").symlink_to(artifact)

    with pytest.raises(tool.BenchmarkError, match="WARM_CACHE"):
        tool.verify_warm_cache(cache)


def test_cold_cache_refuses_any_preexisting_path(tmp_path):
    tool = load_tool()
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(tool.BenchmarkError, match="COLD_CACHE"):
        tool.prepare_cold_cache(cache)
