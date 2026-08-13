"""RNG seed-bank publication recovery semantics."""
import importlib.util
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "tools/generate_rng_seed_bank.py"

def _generator():
    spec = importlib.util.spec_from_file_location("rng_seed_bank_generator", TOOL_PATH)
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader
    spec.loader.exec_module(module); return module

def _debris(directory): return list(directory.glob(".rng-seed-bank-*"))

@pytest.mark.parametrize("mode", (0o600, 0o644, 0o755))
def test_publication_preserves_existing_mode(tmp_path, mode):
    g = _generator(); path = tmp_path / "artifact"; path.write_bytes(b"old"); os.chmod(path, mode)
    g.atomic_write_paths({path: b"new"})
    assert path.read_bytes() == b"new" and path.stat().st_mode & 0o777 == mode

def test_publication_absent_is_0644_and_fsyncs_parent(tmp_path, monkeypatch):
    g = _generator(); path = tmp_path / "new"; calls=[]; real_fsync=g.os.fsync
    monkeypatch.setattr(g, "_fsync_parent", lambda parent: calls.append(parent))
    g.atomic_write_paths({path:b"new"})
    assert path.read_bytes()==b"new" and path.stat().st_mode & 0o777==0o644 and calls == [tmp_path, tmp_path]

def test_second_replace_rolls_back_bytes_modes_and_removes_debris(tmp_path, monkeypatch):
    g=_generator(); a,b=tmp_path/"a",tmp_path/"b"; a.write_bytes(b"old a");b.write_bytes(b"old b");os.chmod(a,0o600);os.chmod(b,0o755)
    real=g.os.replace; n=0
    def fail(src,dst):
        nonlocal n
        if Path(src).name.endswith(".tmp"):
            n+=1
            if n==2: raise OSError("publish")
        return real(src,dst)
    monkeypatch.setattr(g.os,"replace",fail)
    with pytest.raises(g.SeedBankError, match="publication failed"): g.atomic_write_paths({a:b"new a",b:b"new b"})
    assert (a.read_bytes(),b.read_bytes())==(b"old a",b"old b")
    assert (a.stat().st_mode&0o777,b.stat().st_mode&0o777)==(0o600,0o755) and not _debris(tmp_path)

def test_initially_absent_outputs_removed_after_failure(tmp_path, monkeypatch):
    g=_generator(); a,b=tmp_path/"a",tmp_path/"b"; real=g.os.replace; n=0
    def fail(src,dst):
        nonlocal n
        if Path(src).name.endswith(".tmp"):
            n+=1
            if n==2: raise OSError("publish")
        return real(src,dst)
    monkeypatch.setattr(g.os,"replace",fail)
    with pytest.raises(g.SeedBankError): g.atomic_write_paths({a:b"a",b:b"b"})
    assert not a.exists() and not b.exists() and not _debris(tmp_path)

def test_atomic_restore_failure_uses_direct_fallback(tmp_path, monkeypatch):
    g=_generator(); a,b=tmp_path/"a",tmp_path/"b";a.write_bytes(b"old a");b.write_bytes(b"old b"); real=g.os.replace;n=0
    def fail(src,dst):
        nonlocal n
        if Path(src).name.endswith(".tmp"):
            n+=1
            if n==2: raise OSError("publish")
        elif n>=2: raise OSError("restore")
        return real(src,dst)
    monkeypatch.setattr(g.os,"replace",fail)
    with pytest.raises(g.SeedBankError,match=r"^seed-bank publication failed$"): g.atomic_write_paths({a:b"new a",b:b"new b"})
    assert a.read_bytes()==b"old a" and b.read_bytes()==b"old b" and not _debris(tmp_path)

def test_fallback_failure_reports_incomplete_rollback_and_retains_backup(tmp_path, monkeypatch):
    g=_generator();a,b=tmp_path/"a",tmp_path/"b";a.write_bytes(b"old a");b.write_bytes(b"old b");real=g.os.replace;n=0
    def fail(src,dst):
        nonlocal n
        if Path(src).name.endswith(".tmp"):
            n+=1
            if n==2: raise OSError("publish")
        elif n>=2: raise OSError("restore")
        return real(src,dst)
    monkeypatch.setattr(g.os,"replace",fail)
    monkeypatch.setattr(g,"_write_recovery_bytes",lambda *args: (_ for _ in ()).throw(OSError("fallback")))
    with pytest.raises(g.SeedBankError,match="rollback incomplete"): g.atomic_write_paths({a:b"new a",b:b"new b"})
    backups=list(tmp_path.glob(".rng-seed-bank-*.bak")); assert backups and all(path.read_bytes()==b"old a" for path in backups)

def test_stage_failure_leaves_destinations_and_cleanup_failure_is_reported(tmp_path, monkeypatch):
    g=_generator();a,b=tmp_path/"a",tmp_path/"b";a.write_bytes(b"old a");b.write_bytes(b"old b"); real=g._stage_bytes;n=0
    def fail_stage(*args,**kwargs):
        nonlocal n
        n+=1
        if n==3: raise OSError("stage")
        return real(*args,**kwargs)
    monkeypatch.setattr(g,"_stage_bytes",fail_stage)
    with pytest.raises(g.SeedBankError): g.atomic_write_paths({a:b"new a",b:b"new b"})
    assert a.read_bytes()==b"old a" and b.read_bytes()==b"old b"

@pytest.mark.parametrize("failure_number", (1, 2), ids=("backup-stage", "payload-stage"))
def test_each_stage_failure_precedes_publication(tmp_path, monkeypatch, failure_number):
    g=_generator(); a,b=tmp_path/"a",tmp_path/"b"; a.write_bytes(b"old a"); b.write_bytes(b"old b"); real=g._stage_bytes; calls=0
    def fail_stage(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_number: raise OSError("stage")
        return real(*args, **kwargs)
    monkeypatch.setattr(g, "_stage_bytes", fail_stage)
    with pytest.raises(g.SeedBankError, match="publication failed"): g.atomic_write_paths({a:b"new a", b:b"new b"})
    assert (a.read_bytes(), b.read_bytes()) == (b"old a", b"old b") and not _debris(tmp_path)

def test_successful_publication_surfaces_cleanup_failure_and_retains_backup(tmp_path, monkeypatch):
    g=_generator(); a=tmp_path/"a"; a.write_bytes(b"old"); real=g._cleanup
    def fail_cleanup(paths):
        paths=list(paths); return [(paths[0], "OSError")] if paths else real(paths)
    monkeypatch.setattr(g, "_cleanup", fail_cleanup)
    with pytest.raises(g.SeedBankError, match="publication cleanup failed"): g.atomic_write_paths({a:b"new"})
    backups=list(tmp_path.glob(".rng-seed-bank-*.bak"))
    assert a.read_bytes() == b"new" and len(backups) == 1 and backups[0].read_bytes() == b"old"

def test_release_rejects_prospective_bundle_without_touching_live_outputs(tmp_path, monkeypatch):
    g=_generator(); bank=tmp_path/g.BANK_PATH; manifest=tmp_path/g.MANIFEST_PATH; bank.parent.mkdir(parents=True); manifest.parent.mkdir(parents=True); bank.write_bytes(b"live-bank"); manifest.write_bytes(b"live-manifest")
    monkeypatch.setattr(g, "build", lambda root: {g.BANK_NAME:b"future-bank", g.MANIFEST_NAME:b"future-manifest"})
    monkeypatch.setattr(g, "verify_bundle", lambda *args: (_ for _ in ()).throw(g.SeedBankError("prospective semantic failure")))
    with pytest.raises(g.SeedBankError, match="prospective semantic failure"): g.release(tmp_path)
    assert bank.read_bytes() == b"live-bank" and manifest.read_bytes() == b"live-manifest" and not _debris(tmp_path)

def test_stages_every_backup_and_payload_durably_before_first_publish(tmp_path, monkeypatch):
    g=_generator(); a,b=tmp_path/"a",tmp_path/"b"; a.write_bytes(b"old a"); b.write_bytes(b"old b"); events=[]; real=g.os.replace
    monkeypatch.setattr(g, "_fsync_parent", lambda parent: events.append(("fsync", parent)))
    monkeypatch.setattr(g.os, "replace", lambda src,dst: (events.append(("replace", Path(src), Path(dst))), real(src,dst))[1])
    g.atomic_write_paths({b:b"new b", a:b"new a"})
    first_publish=next(i for i,event in enumerate(events) if event[0]=="replace" and event[1].suffix==".tmp")
    assert [event[0] for event in events[:first_publish]] == ["fsync"] * 4
    assert [event[1] for event in events[:first_publish]] == [tmp_path] * 4

def test_absent_stage_and_rollback_unlink_are_parent_fsynced(tmp_path, monkeypatch):
    g=_generator(); a,b=tmp_path/"a",tmp_path/"b"; events=[]; real=g.os.replace; publish=0
    def fail_second_publish(src,dst):
        nonlocal publish
        if Path(src).suffix==".tmp":
            publish += 1
            if publish==2: raise OSError("publish")
        return real(src,dst)
    monkeypatch.setattr(g, "_fsync_parent", lambda parent: events.append(parent))
    monkeypatch.setattr(g.os,"replace",fail_second_publish)
    with pytest.raises(g.SeedBankError, match="publication failed"): g.atomic_write_paths({a:b"a",b:b"b"})
    assert events == [tmp_path] * 5

def test_atomic_restore_and_direct_fallback_cleanup_fsync_parent(tmp_path, monkeypatch):
    g=_generator(); a,b=tmp_path/"a",tmp_path/"b"; a.write_bytes(b"old a");b.write_bytes(b"old b"); events=[]; real=g.os.replace; n=0
    def fail_publish_and_atomic_restore(src,dst):
        nonlocal n
        if Path(src).suffix==".tmp":
            n+=1
            if n==2: raise OSError("publish")
        elif n>=2: raise OSError("restore")
        return real(src,dst)
    monkeypatch.setattr(g, "_fsync_parent", lambda parent: events.append(parent))
    monkeypatch.setattr(g.os,"replace",fail_publish_and_atomic_restore)
    with pytest.raises(g.SeedBankError,match=r"^seed-bank publication failed$"): g.atomic_write_paths({a:b"new a",b:b"new b"})
    # Four staging syncs, publication sync, then direct recovery and backup-removal syncs.
    assert len(events) >= 7 and events[-2:] == [tmp_path, tmp_path]

def test_atomic_backup_restore_fsyncs_parent_after_replace(tmp_path, monkeypatch):
    g=_generator(); a,b=tmp_path/"a",tmp_path/"b"; a.write_bytes(b"old a"); b.write_bytes(b"old b"); events=[]; real=g.os.replace; publish=0
    def fail_second_publish(src,dst):
        nonlocal publish
        if Path(src).suffix==".tmp":
            publish += 1
            if publish==2: raise OSError("publish")
        events.append(("replace", Path(src).suffix, Path(dst)))
        return real(src,dst)
    monkeypatch.setattr(g.os,"replace",fail_second_publish); monkeypatch.setattr(g,"_fsync_parent",lambda parent: events.append(("fsync", parent)))
    with pytest.raises(g.SeedBankError,match=r"^seed-bank publication failed$"): g.atomic_write_paths({a:b"new a",b:b"new b"})
    restore=next(i for i,event in enumerate(events) if event[0]=="replace" and event[1]==".bak")
    assert events[restore+1] == ("fsync", tmp_path)

def test_rollback_restore_parent_fsync_failure_is_incomplete_and_keeps_backup(tmp_path, monkeypatch):
    g=_generator(); a,b=tmp_path/"a",tmp_path/"b";a.write_bytes(b"old a");b.write_bytes(b"old b"); real_replace=g.os.replace; calls=0
    def fail_second_publish(src,dst):
        nonlocal calls
        if Path(src).suffix==".tmp":
            calls+=1
            if calls==2: raise OSError("publish")
        return real_replace(src,dst)
    def fsync_fail_after_publish(parent):
        if calls >= 2: raise OSError("directory sync")
    monkeypatch.setattr(g.os,"replace",fail_second_publish); monkeypatch.setattr(g,"_fsync_parent",fsync_fail_after_publish)
    with pytest.raises(g.SeedBankError, match="rollback incomplete") as error: g.atomic_write_paths({a:b"new a",b:b"new b"})
    assert str(tmp_path) in str(error.value) and list(tmp_path.glob(".rng-seed-bank-*.bak"))


def test_atomic_restore_parent_fsync_failure_retains_durable_recovery_authority(tmp_path, monkeypatch):
    g=_generator(); a,b=tmp_path/"a",tmp_path/"b"; a.write_bytes(b"old a"); b.write_bytes(b"old b"); os.chmod(a, 0o600); real_replace=g.os.replace; syncs=0; publishes=0
    def fail_second_publish(src, dst):
        nonlocal publishes
        if Path(src).suffix == ".tmp":
            publishes += 1
            if publishes == 2: raise OSError("publish")
        return real_replace(src, dst)
    def fail_only_restore_sync(parent):
        nonlocal syncs
        syncs += 1
        # backup/payload staging (4), first publication (1), recovery staging (1), restore
        if syncs == 7: raise OSError("restore directory sync")
    monkeypatch.setattr(g.os, "replace", fail_second_publish); monkeypatch.setattr(g, "_fsync_parent", fail_only_restore_sync)
    with pytest.raises(g.SeedBankError, match="rollback incomplete"):
        g.atomic_write_paths({a:b"new a", b:b"new b"})
    recovery=list(tmp_path.glob(".rng-seed-bank-*.recovery.bak"))
    assert a.read_bytes() == b"old a" and a.stat().st_mode & 0o777 == 0o600
    assert len(recovery) == 1 and recovery[0].read_bytes() == b"old a" and b.read_bytes() == b"old b"


def test_atomic_restore_success_cleans_recovery_authority(tmp_path, monkeypatch):
    g=_generator(); a,b=tmp_path/"a",tmp_path/"b"; a.write_bytes(b"old a"); b.write_bytes(b"old b"); real=g.os.replace; publishes=0
    def fail_second_publish(src, dst):
        nonlocal publishes
        if Path(src).suffix == ".tmp":
            publishes += 1
            if publishes == 2: raise OSError("publish")
        return real(src, dst)
    monkeypatch.setattr(g.os, "replace", fail_second_publish)
    with pytest.raises(g.SeedBankError, match=r"^seed-bank publication failed$"):
        g.atomic_write_paths({a:b"new a", b:b"new b"})
    assert (a.read_bytes(), b.read_bytes()) == (b"old a", b"old b") and not _debris(tmp_path)


def test_recovery_cleanup_fsync_failure_retains_intact_recovery_evidence(tmp_path, monkeypatch):
    g=_generator(); a,b=tmp_path/"a",tmp_path/"b"; a.write_bytes(b"old a"); b.write_bytes(b"old b"); real=g.os.replace; publishes=0; syncs=0
    def fail_second_publish(src, dst):
        nonlocal publishes
        if Path(src).suffix == ".tmp":
            publishes += 1
            if publishes == 2: raise OSError("publish")
        return real(src, dst)
    def fail_recovery_cleanup_sync(parent):
        nonlocal syncs
        syncs += 1
        # stages 4, publication 1, recovery 1, restore 1, cleanup guard 1, recovery cleanup 1
        if syncs == 9: raise OSError("recovery cleanup directory sync")
    monkeypatch.setattr(g.os, "replace", fail_second_publish); monkeypatch.setattr(g, "_fsync_parent", fail_recovery_cleanup_sync)
    with pytest.raises(g.SeedBankError, match="rollback incomplete"):
        g.atomic_write_paths({a:b"new a", b:b"new b"})
    recovery=list(tmp_path.glob(".rng-seed-bank-*.recovery.bak"))
    assert a.read_bytes() == b"old a" and any(path.read_bytes() == b"old a" for path in recovery)


def test_cleanup_authority_failure_is_retained_and_never_cleaned_twice(tmp_path, monkeypatch):
    g=_generator(); a,b=tmp_path/"a",tmp_path/"b"; a.write_bytes(b"old a"); b.write_bytes(b"old b"); os.chmod(a, 0o600); real_replace=g.os.replace; publishes=0; cleanup_calls=[]; real_cleanup=g._cleanup
    def fail_second_publish(src, dst):
        nonlocal publishes
        if Path(src).suffix == ".tmp":
            publishes += 1
            if publishes == 2: raise OSError("publish")
        return real_replace(src, dst)
    def record_and_fail_cleanup(paths):
        paths=list(paths); cleanup_calls.append(tuple(paths))
        if len(cleanup_calls) == 2:
            assert len(paths) == 1 and paths[0].suffix == ".bak"
            return [(paths[0], "OSError")]
        return real_cleanup(paths)
    monkeypatch.setattr(g.os, "replace", fail_second_publish)
    monkeypatch.setattr(g, "_cleanup", record_and_fail_cleanup)
    with pytest.raises(g.SeedBankError, match="rollback incomplete"):
        g.atomic_write_paths({a:b"new a", b:b"new b"})
    authority = cleanup_calls[1][0]
    assert a.read_bytes() == b"old a" and (a.stat().st_mode & 0o777) == 0o600 and b.read_bytes() == b"old b"
    assert authority.exists() and authority.read_bytes() == b"old a"
    assert sum(authority in paths for paths in cleanup_calls) == 1


def test_stage_directory_fsync_failure_prevents_publish_and_cleans_honestly(tmp_path, monkeypatch):
    g=_generator(); a=tmp_path/"a";a.write_bytes(b"old"); mode=a.stat().st_mode & 0o777; replace_calls=[]
    monkeypatch.setattr(g,"_fsync_parent",lambda parent: (_ for _ in ()).throw(OSError("stage directory sync")))
    monkeypatch.setattr(g.os,"replace",lambda src,dst: replace_calls.append((src,dst)))
    with pytest.raises(g.SeedBankError, match="publication failed"): g.atomic_write_paths({a:b"new"})
    assert a.read_bytes()==b"old" and a.stat().st_mode & 0o777==mode and not replace_calls and not _debris(tmp_path)
