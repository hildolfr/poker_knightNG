"""Secure L1 Unix-listener construction and lifecycle tests."""
from __future__ import annotations

import asyncio
import errno
import fcntl
import inspect
import os
import stat
from dataclasses import replace
from types import SimpleNamespace

import pytest


def _run(awaitable):
    return asyncio.run(awaitable)


class _FakeServer:
    def __init__(self, owner) -> None:
        self.owner = owner
        self.started = False
        self.close_calls = 0
        self.wait_closed_calls = 0

    async def start_serving(self) -> None:
        self.owner.trace.append("server-start")
        self.owner._raise("server-start")
        self.started = True

    def close(self) -> None:
        self.owner.trace.append("server-close")
        self.close_calls += 1
        self.owner._raise("server-close")

    async def wait_closed(self) -> None:
        self.owner.trace.append("server-wait-closed")
        self.wait_closed_calls += 1
        self.owner._raise("server-wait-closed")


class _Writer:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _ClosedSyscalls:
    def __init__(self) -> None:
        from poker_knight_ng_service.listener import _Snapshot

        self.trace: list[str] = []
        self.parent_fd = 10
        self.lock_fd = 11
        self.parent = _Snapshot(1, 10, stat.S_IFDIR | 0o750, 1201, 1202)
        self.lock = _Snapshot(1, 11, stat.S_IFREG | 0o600, 1201, 1202)
        self.target = None
        self.probe_result: int | None = errno.ECONNREFUSED
        self.replacements: dict[str, object] = {}
        self.faults: dict[str, BaseException] = {}
        self.closed_fds: list[int] = []
        self.parent_open: tuple[str, int] | None = None
        self.lock_open: tuple[str, int, int, int] | None = None
        self.bind: tuple[str, bool] | None = None
        self.callback = None
        self.server: _FakeServer | None = None
        self.unlink_calls = 0
        self.chmod_calls = 0

    def _raise(self, event: str) -> None:
        failure = self.faults.get(event)
        if failure is not None:
            raise failure

    def open_parent(self, path: str, flags: int) -> int:
        self.trace.append("open-parent")
        self.parent_open = (path, flags)
        self._raise("open-parent")
        return self.parent_fd

    def fstat(self, fd: int):
        event = "fstat-parent" if fd == self.parent_fd else "fstat-lock"
        self.trace.append(event)
        self._raise(event)
        return self.parent if fd == self.parent_fd else self.lock

    def open_lock(self, name: str, flags: int, mode: int, dir_fd: int) -> int:
        self.trace.append("open-lock")
        self.lock_open = (name, flags, mode, dir_fd)
        self._raise("open-lock")
        return self.lock_fd

    def flock(self, fd: int, operation: int) -> None:
        self.trace.append("flock")
        assert fd == self.lock_fd
        assert operation == fcntl.LOCK_EX | fcntl.LOCK_NB
        self._raise("flock")

    def inspect_target(self, name: str, dir_fd: int):
        self.trace.append("inspect-target")
        assert name == "service.sock"
        assert dir_fd == self.parent_fd
        self._raise("inspect-target")
        return self.target

    def boundary(self, name: str) -> None:
        self.trace.append(f"boundary:{name}")
        if name in self.replacements:
            self.target = self.replacements[name]
        self._raise(f"boundary:{name}")

    async def probe(self, path: str, timeout_seconds: float) -> int | None:
        assert path == "/run/poker-knight-ng/service.sock"
        assert timeout_seconds == 0.250
        self.trace.extend(("probe-open", "probe-connect"))
        try:
            self._raise("probe-connect")
            return self.probe_result
        finally:
            self.trace.append("probe-close")

    def unlink_target(self, name: str, dir_fd: int) -> None:
        self.trace.append("unlink-target")
        assert name == "service.sock"
        assert dir_fd == self.parent_fd
        self._raise("unlink-target")
        self.unlink_calls += 1
        self.target = None

    async def start_server(self, callback, path: str, start_serving: bool):
        from poker_knight_ng_service.listener import _Snapshot

        self.trace.append("start-server")
        self._raise("start-server")
        self.callback = callback
        self.bind = (path, start_serving)
        self.target = _Snapshot(1, 200, stat.S_IFSOCK | 0o755, 1201, 1202)
        self.server = _FakeServer(self)
        return self.server

    def chmod_target(self, name: str, mode: int, dir_fd: int) -> None:
        self.trace.append("chmod-target")
        assert name == "service.sock"
        assert mode == 0o660
        assert dir_fd == self.parent_fd
        self._raise("chmod-target")
        self.chmod_calls += 1
        assert self.target is not None
        self.target = replace(self.target, mode=stat.S_IFSOCK | mode)

    def close_fd(self, fd: int) -> None:
        self.trace.append(f"close-fd:{fd}")
        self.closed_fds.append(fd)
        self._raise(f"close-fd:{fd}")


def _resolved(monkeypatch):
    from poker_knight_ng_service import identity

    passwd = SimpleNamespace(pw_uid=1201, pw_gid=1202)
    group = SimpleNamespace(gr_gid=1202)
    monkeypatch.setattr(identity, "_getpwnam", lambda name: passwd)
    monkeypatch.setattr(identity, "_getgrnam", lambda name: group)
    return identity.resolve_production_identity()


def _socket(*, inode: int = 100, uid: int = 1201, gid: int = 1202, mode: int = 0o660):
    from poker_knight_ng_service.listener import _Snapshot

    return _Snapshot(1, inode, stat.S_IFSOCK | mode, uid, gid)


def test_public_constructor_is_canonical_and_rejects_forged_identity(monkeypatch) -> None:
    from poker_knight_ng_service import identity, listener

    assert tuple(inspect.signature(listener.construct_l1_listener).parameters) == ("identity",)
    source = inspect.getsource(listener.construct_l1_listener)
    assert "/run/poker-knight-ng/service.sock" not in source
    assert "path" not in inspect.signature(listener.construct_l1_listener).parameters

    fake = _ClosedSyscalls()
    forged = object.__new__(identity.ResolvedServiceIdentity)
    with pytest.raises(identity.IdentityResolutionError):
        _run(listener._construct_l1_listener(forged, fake))
    assert fake.trace == []


def test_success_holds_lock_through_paused_bind_and_postconditions(monkeypatch) -> None:
    from poker_knight_ng_service import listener

    fake = _ClosedSyscalls()
    built = _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))

    required_parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    required_lock_flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    assert fake.parent_open == ("/run/poker-knight-ng", required_parent_flags)
    assert fake.lock_open == ("service.lock", required_lock_flags, 0o600, fake.parent_fd)
    assert fake.bind == ("/run/poker-knight-ng/service.sock", False)
    assert fake.server is not None and fake.server.started
    assert fake.target == _socket(inode=200)
    assert fake.closed_fds == []
    assert fake.trace.index("flock") < fake.trace.index("inspect-target")
    assert fake.trace.index("chmod-target") < fake.trace.index("server-start")

    writer = _Writer()
    assert fake.callback is not None
    fake.callback(object(), writer)
    assert writer.close_calls == 1

    _run(built.close())
    assert fake.server.close_calls == 1
    assert fake.server.wait_closed_calls == 1
    assert fake.unlink_calls == 1
    assert fake.closed_fds == [fake.lock_fd, fake.parent_fd]
    assert fake.trace.index("server-close") < fake.trace.index("unlink-target")
    assert fake.trace.index("unlink-target") < fake.trace.index(f"close-fd:{fake.lock_fd}")


@pytest.mark.parametrize(
    ("component", "snapshot"),
    [
        ("parent-kind", lambda fake: replace(fake.parent, mode=stat.S_IFREG | 0o750)),
        ("parent-uid", lambda fake: replace(fake.parent, uid=9)),
        ("parent-gid", lambda fake: replace(fake.parent, gid=9)),
        ("parent-mode", lambda fake: replace(fake.parent, mode=stat.S_IFDIR | 0o755)),
        ("lock-kind", lambda fake: replace(fake.lock, mode=stat.S_IFSOCK | 0o600)),
        ("lock-uid", lambda fake: replace(fake.lock, uid=9)),
        ("lock-gid", lambda fake: replace(fake.lock, gid=9)),
        ("lock-mode", lambda fake: replace(fake.lock, mode=stat.S_IFREG | 0o640)),
    ],
)
def test_parent_and_lock_metadata_fail_before_target_inspection(
    monkeypatch,
    component: str,
    snapshot,
) -> None:
    from poker_knight_ng_service import listener

    fake = _ClosedSyscalls()
    if component.startswith("parent"):
        fake.parent = snapshot(fake)
    else:
        fake.lock = snapshot(fake)
    with pytest.raises(listener.ListenerConstructionError):
        _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))
    assert "inspect-target" not in fake.trace
    assert fake.closed_fds[-1] == fake.parent_fd


def test_lock_contention_fails_before_target_inspection(monkeypatch) -> None:
    from poker_knight_ng_service import listener

    fake = _ClosedSyscalls()
    fake.faults["flock"] = BlockingIOError(errno.EWOULDBLOCK, "held")
    with pytest.raises(listener.ListenerConstructionError):
        _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))
    assert "inspect-target" not in fake.trace
    assert fake.closed_fds == [fake.lock_fd, fake.parent_fd]


@pytest.mark.parametrize(
    "target",
    [
        lambda: None,
        lambda: _socket(uid=9),
        lambda: _socket(gid=9),
        lambda: _socket(mode=0o600),
    ],
)
def test_invalid_existing_targets_fail_without_probe_or_mutation(monkeypatch, target) -> None:
    from poker_knight_ng_service import listener

    fake = _ClosedSyscalls()
    fake.target = target()
    if fake.target is None:
        from poker_knight_ng_service.listener import _Snapshot

        fake.target = _Snapshot(1, 100, stat.S_IFREG | 0o660, 1201, 1202)
    with pytest.raises(listener.ListenerConstructionError):
        _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))
    assert "probe-open" not in fake.trace
    assert fake.unlink_calls == 0


@pytest.mark.parametrize("probe_result", [0, None, errno.EIO])
def test_only_connection_refused_is_stale(monkeypatch, probe_result: int | None) -> None:
    from poker_knight_ng_service import listener

    fake = _ClosedSyscalls()
    fake.target = _socket()
    fake.probe_result = probe_result
    with pytest.raises(listener.ListenerConstructionError):
        _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))
    assert fake.trace.count("probe-close") == 1
    assert fake.unlink_calls == 0


def test_connection_refused_unlinks_unchanged_stale_socket_under_lock(monkeypatch) -> None:
    from poker_knight_ng_service import listener

    fake = _ClosedSyscalls()
    fake.target = _socket()
    built = _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))
    assert fake.trace.count("probe-close") == 1
    assert fake.unlink_calls == 1
    assert fake.trace.index("flock") < fake.trace.index("unlink-target")
    _run(built.close())


@pytest.mark.parametrize(
    "boundary",
    [
        "check-to-stale-probe",
        "stale-probe-to-reinspect",
        "reinspect-to-unlink",
        "bind-to-chmod",
        "chmod-to-reinspect",
        "reinspect-to-start-serving",
    ],
)
def test_construction_replacement_boundaries_preserve_replacement(
    monkeypatch,
    boundary: str,
) -> None:
    from poker_knight_ng_service import listener

    fake = _ClosedSyscalls()
    if boundary.startswith(("check", "stale", "reinspect-to-unlink")):
        fake.target = _socket()
    replacement = _socket(inode=999)
    fake.replacements[boundary] = replacement

    with pytest.raises(listener.ListenerConstructionError):
        _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))
    assert fake.target == replacement
    assert fake.server is None or not fake.server.started


def test_cleanup_replacement_boundary_preserves_replacement_and_reports(monkeypatch) -> None:
    from poker_knight_ng_service import listener

    fake = _ClosedSyscalls()
    built = _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))
    replacement = _socket(inode=999)
    fake.replacements["cleanup-reinspect-to-unlink"] = replacement

    with pytest.raises(listener.ListenerCleanupError):
        _run(built.close())
    assert fake.target == replacement
    assert fake.closed_fds == [fake.lock_fd, fake.parent_fd]


def test_start_serving_failure_closes_server_and_removes_only_created_socket(monkeypatch) -> None:
    from poker_knight_ng_service import listener

    fake = _ClosedSyscalls()
    fake.faults["server-start"] = RuntimeError("start")
    with pytest.raises(listener.ListenerConstructionError):
        _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))
    assert fake.server is not None
    assert fake.server.close_calls == 1
    assert fake.server.wait_closed_calls == 1
    assert fake.target is None
    assert fake.closed_fds == [fake.lock_fd, fake.parent_fd]


def test_construction_process_control_signal_propagates_after_cleanup(monkeypatch) -> None:
    from poker_knight_ng_service import listener

    class Stop(BaseException):
        pass

    fake = _ClosedSyscalls()
    signal = Stop()
    fake.faults["server-start"] = signal
    with pytest.raises(Stop) as caught:
        _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))
    assert caught.value is signal
    assert fake.target is None
    assert fake.closed_fds == [fake.lock_fd, fake.parent_fd]


def test_construct_failure_propagates_cleanup_base_exception(monkeypatch) -> None:
    from poker_knight_ng_service import listener

    class Stop(BaseException):
        pass

    fake = _ClosedSyscalls()
    fake.faults["server-start"] = RuntimeError("construction")
    fake.faults["server-close"] = Stop("shutdown")

    with pytest.raises(Stop):
        _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))
    assert fake.closed_fds == [fake.lock_fd, fake.parent_fd]


def test_close_process_control_signal_propagates_after_cleanup(monkeypatch) -> None:
    from poker_knight_ng_service import listener

    class Stop(BaseException):
        pass

    fake = _ClosedSyscalls()
    built = _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))
    signal = Stop()
    fake.faults["server-close"] = signal
    with pytest.raises(Stop) as caught:
        _run(built.close())
    assert caught.value is signal
    assert fake.target is None
    assert fake.closed_fds == [fake.lock_fd, fake.parent_fd]
    _run(built.close())
    assert fake.server is not None and fake.server.close_calls == 1


def test_real_syscalls_enforce_dirfd_nofollow_flock_chmod_and_unlink(tmp_path) -> None:
    import socket

    from poker_knight_ng_service import listener

    real = listener._RealSyscalls()
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    parent_fd = real.open_parent(str(tmp_path), parent_flags)
    first_lock = real.open_lock(
        "service.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        parent_fd,
    )
    second_lock = real.open_lock(
        "service.lock",
        os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        parent_fd,
    )
    unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        real.flock(first_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            real.flock(second_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        socket_path = tmp_path / "service.sock"
        unix_socket.bind(str(socket_path))
        target = real.inspect_target("service.sock", parent_fd)
        assert target is not None and stat.S_ISSOCK(target.mode)
        real.chmod_target("service.sock", 0o660, parent_fd)
        target = real.inspect_target("service.sock", parent_fd)
        assert target is not None and stat.S_IMODE(target.mode) == 0o660
        real.unlink_target("service.sock", parent_fd)
        assert real.inspect_target("service.sock", parent_fd) is None

        (tmp_path / "target").write_text("x")
        (tmp_path / "service.sock").symlink_to("target")
        target = real.inspect_target("service.sock", parent_fd)
        assert target is not None and stat.S_ISLNK(target.mode)
    finally:
        unix_socket.close()
        real.close_fd(second_lock)
        real.close_fd(first_lock)
        real.close_fd(parent_fd)


def test_real_nonblocking_probe_distinguishes_live_and_refused(tmp_path) -> None:
    import socket

    from poker_knight_ng_service import listener

    async def scenario() -> None:
        real = listener._RealSyscalls()
        path = str(tmp_path / "probe.sock")
        live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        live.bind(path)
        live.listen(1)
        try:
            assert await real.probe(path, 0.250) == 0
        finally:
            live.close()
            os.unlink(path)

        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(path)
        stale.close()
        try:
            assert await real.probe(path, 0.250) == errno.ECONNREFUSED
        finally:
            os.unlink(path)

    _run(scenario())


def test_real_paused_asyncio_server_accepts_only_after_explicit_start(tmp_path) -> None:
    from poker_knight_ng_service import listener

    async def scenario() -> None:
        real = listener._RealSyscalls()
        path = str(tmp_path / "server.sock")
        accepted = asyncio.Event()

        def close_only(reader, writer) -> None:
            del reader
            writer.close()
            accepted.set()

        server = await real.start_server(close_only, path, False)
        assert not server.is_serving()
        await server.start_serving()
        assert server.is_serving()
        reader, writer = await asyncio.open_unix_connection(path)
        await asyncio.wait_for(accepted.wait(), timeout=1.0)
        assert await asyncio.wait_for(reader.read(1), timeout=1.0) == b""
        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()
        if os.path.exists(path):
            os.unlink(path)

    _run(scenario())



def test_cleanup_base_exception_from_target_reinspection_propagates(monkeypatch) -> None:
    from poker_knight_ng_service import listener

    class Stop(BaseException):
        pass

    fake = _ClosedSyscalls()
    built = _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))
    fake.faults["boundary:cleanup-reinspect-to-unlink"] = Stop()

    with pytest.raises(Stop):
        _run(built.close())



def test_cleanup_base_exception_from_close_fd_propagates_and_runs_all_fds(monkeypatch) -> None:
    from poker_knight_ng_service import listener

    class Stop(BaseException):
        pass

    fake = _ClosedSyscalls()
    built = _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))
    fake.faults[f"close-fd:{fake.lock_fd}"] = Stop()

    with pytest.raises(Stop):
        _run(built.close())

    assert fake.closed_fds == [fake.lock_fd, fake.parent_fd]


def test_real_probe_timeout_branches_nonblocking_safely(tmp_path) -> None:
    from poker_knight_ng_service import listener

    timed_out = {"seen": False}

    async def fake_wait_for(awaitable, timeout):
        timed_out["seen"] = True
        # The probe constructs a real sock_connect coroutine before calling
        # wait_for; a synthetic timeout must still close that awaitable so the
        # test does not leak an unawaited coroutine warning.
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        raise asyncio.TimeoutError

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(listener.asyncio, "wait_for", fake_wait_for)
        result = asyncio.run(listener._RealSyscalls().probe(str(tmp_path / "probe.sock"), 0.250))

    assert timed_out["seen"]
    assert result is None


def test_construct_rejects_inherited_listener(monkeypatch) -> None:
    from poker_knight_ng_service import listener

    fake = _ClosedSyscalls()
    monkeypatch.setenv("LISTEN_PID", str(os.getpid()))
    monkeypatch.setenv("LISTEN_FDS", "1")

    with pytest.raises(listener.ListenerConstructionError):
        _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))

    assert fake.trace == [
        "open-parent",
        "fstat-parent",
        "open-lock",
        "fstat-lock",
        "flock",
        "boundary:systemd-inherited-listener",
        "close-fd:11",
        "close-fd:10",
    ]
    # Cleanup closes only the constructor's own parent/lock fds; the
    # inherited-listener marker fd (3) is never adopted or closed.
    assert fake.closed_fds == [11, 10]


def test_inherited_listener_fd_respects_matching_pid_and_fds(monkeypatch) -> None:
    from poker_knight_ng_service import listener

    monkeypatch.setenv("LISTEN_PID", str(os.getpid()))
    monkeypatch.setenv("LISTEN_FDS", "1")

    assert listener._inherited_listener_fd() == 3


def test_inherited_listener_fd_ignores_bad_pid(monkeypatch) -> None:
    from poker_knight_ng_service import listener

    monkeypatch.setenv("LISTEN_PID", "999999")
    monkeypatch.setenv("LISTEN_FDS", "1")

    assert listener._inherited_listener_fd() is None


@pytest.mark.parametrize("count", ("0", "2", "not-a-count"))
def test_inherited_listener_fd_fails_closed_on_any_matching_activation_state(
    monkeypatch, count: str
) -> None:
    """A matching LISTEN_PID is forbidden regardless of the LISTEN_FDS value."""
    from poker_knight_ng_service import listener

    monkeypatch.setenv("LISTEN_PID", str(os.getpid()))
    monkeypatch.setenv("LISTEN_FDS", count)

    assert listener._inherited_listener_fd() == 3


def test_cleanup_mixed_failures_prefers_base_exception_over_ordinary(monkeypatch) -> None:
    from poker_knight_ng_service import listener

    class _OrdinaryFailure(RuntimeError):
        pass

    fake = _ClosedSyscalls()
    fake.faults["server-close"] = _OrdinaryFailure("ordinary")
    fake.faults["server-wait-closed"] = KeyboardInterrupt("control")

    built = _run(listener._construct_l1_listener(_resolved(monkeypatch), fake))
    with pytest.raises(KeyboardInterrupt):
        _run(built.close())

    assert fake.closed_fds == [fake.lock_fd, fake.parent_fd]
