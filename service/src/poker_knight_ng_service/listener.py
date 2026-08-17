"""Secure construction of the canonical paused Unix listener."""
from __future__ import annotations

import asyncio
import errno
import fcntl
import os
import socket
import stat
from dataclasses import dataclass
from typing import Protocol

from .identity import ResolvedServiceIdentity, _identity_values

_PARENT_PATH = "/run/poker-knight-ng"
_SOCKET_PATH = "/run/poker-knight-ng/service.sock"
_SOCKET_BASENAME = "service.sock"
_LOCK_BASENAME = "service.lock"
_PARENT_MODE = 0o750
_LOCK_MODE = 0o600
_SOCKET_MODE = 0o660
_PROBE_TIMEOUT_SECONDS = 0.250


class ListenerConstructionError(RuntimeError):
    """The canonical listener could not be constructed safely."""


class ListenerCleanupError(RuntimeError):
    """The listener closed but safe pathname cleanup was not completed."""


@dataclass(frozen=True, slots=True)
class _Snapshot:
    dev: int
    inode: int
    mode: int
    uid: int
    gid: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _Snapshot:
        return cls(value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid)


class _Server(Protocol):
    def close(self) -> None: ...

    async def wait_closed(self) -> None: ...

    async def start_serving(self) -> None: ...


class _L1Syscalls(Protocol):
    def open_parent(self, path: str, flags: int) -> int: ...

    def fstat(self, fd: int) -> _Snapshot: ...

    def open_lock(self, name: str, flags: int, mode: int, dir_fd: int) -> int: ...

    def flock(self, fd: int, operation: int) -> None: ...

    def inspect_target(self, name: str, dir_fd: int) -> _Snapshot | None: ...

    def boundary(self, name: str) -> None: ...

    async def probe(self, path: str, timeout_seconds: float) -> int | None: ...

    def unlink_target(self, name: str, dir_fd: int) -> None: ...

    async def start_server(self, callback, path: str, start_serving: bool) -> _Server: ...

    def chmod_target(self, name: str, mode: int, dir_fd: int) -> None: ...

    def close_fd(self, fd: int) -> None: ...


class _RealSyscalls:
    def open_parent(self, path: str, flags: int) -> int:
        return os.open(path, flags)

    def fstat(self, fd: int) -> _Snapshot:
        return _Snapshot.from_stat(os.fstat(fd))

    def open_lock(self, name: str, flags: int, mode: int, dir_fd: int) -> int:
        return os.open(name, flags, mode, dir_fd=dir_fd)

    def flock(self, fd: int, operation: int) -> None:
        fcntl.flock(fd, operation)

    def inspect_target(self, name: str, dir_fd: int) -> _Snapshot | None:
        try:
            value = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        return _Snapshot.from_stat(value)

    def boundary(self, name: str) -> None:
        del name

    async def probe(self, path: str, timeout_seconds: float) -> int | None:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_NONBLOCK)
        try:
            try:
                await asyncio.wait_for(
                    asyncio.get_running_loop().sock_connect(probe, path),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                return None
            except OSError as failure:
                return failure.errno
            return 0
        finally:
            probe.close()

    def unlink_target(self, name: str, dir_fd: int) -> None:
        os.unlink(name, dir_fd=dir_fd)

    async def start_server(self, callback, path: str, start_serving: bool) -> _Server:
        return await asyncio.start_unix_server(
            callback,
            path=path,
            start_serving=start_serving,
        )

    def chmod_target(self, name: str, mode: int, dir_fd: int) -> None:
        os.chmod(name, mode, dir_fd=dir_fd, follow_symlinks=False)

    def close_fd(self, fd: int) -> None:
        os.close(fd)


def _close_accepted(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    del reader
    writer.close()


def _mode(snapshot: _Snapshot) -> int:
    return stat.S_IMODE(snapshot.mode)


def _is_exact(
    snapshot: _Snapshot | None,
    kind,
    uid: int,
    gid: int,
    mode: int,
) -> bool:
    return (
        snapshot is not None
        and kind(snapshot.mode)
        and snapshot.uid == uid
        and snapshot.gid == gid
        and _mode(snapshot) == mode
    )


def _same_object(left: _Snapshot | None, right: _Snapshot) -> bool:
    return (
        left is not None
        and left.dev == right.dev
        and left.inode == right.inode
        and stat.S_IFMT(left.mode) == stat.S_IFMT(right.mode)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ListenerConstructionError(message)


def _require_same_socket(
    current: _Snapshot | None,
    expected: _Snapshot,
    uid: int,
    gid: int,
    *,
    exact_mode: int | None,
) -> _Snapshot:
    valid = (
        _same_object(current, expected)
        and current is not None
        and stat.S_ISSOCK(current.mode)
        and current.uid == uid
        and current.gid == gid
        and (exact_mode is None or _mode(current) == exact_mode)
    )
    _require(valid, "Unix socket target changed or failed validation")
    assert current is not None
    return current


async def _cleanup_target(
    syscalls: _L1Syscalls,
    parent_fd: int,
    expected: _Snapshot,
) -> None:
    current = syscalls.inspect_target(_SOCKET_BASENAME, parent_fd)
    if not _same_object(current, expected) or current is None or not stat.S_ISSOCK(current.mode):
        raise ListenerCleanupError("socket cleanup target is not owned by this listener")
    syscalls.boundary("cleanup-reinspect-to-unlink")
    current = syscalls.inspect_target(_SOCKET_BASENAME, parent_fd)
    if not _same_object(current, expected) or current is None or not stat.S_ISSOCK(current.mode):
        raise ListenerCleanupError("socket cleanup target changed")
    syscalls.unlink_target(_SOCKET_BASENAME, parent_fd)


async def _close_server(server: _Server, failures: list[BaseException]) -> None:
    try:
        server.close()
    except BaseException as failure:
        failures.append(failure)
    try:
        await server.wait_closed()
    except BaseException as failure:
        failures.append(failure)


def _close_fd(syscalls: _L1Syscalls, fd: int | None, failures: list[BaseException]) -> None:
    if fd is None:
        return
    try:
        syscalls.close_fd(fd)
    except BaseException as failure:
        failures.append(failure)


def _report_cleanup_failures(failures: list[BaseException]) -> None:
    if not failures:
        return
    for failure in failures:
        if not isinstance(failure, Exception):
            raise failure
    raise ListenerCleanupError("listener cleanup failed") from failures[0]


class L1Listener:
    """Retained server, namespace lease, and created socket identity."""

    __slots__ = (
        "_server",
        "_syscalls",
        "_parent_fd",
        "_lock_fd",
        "_created",
        "_closed",
    )

    def __init__(
        self,
        server: _Server,
        syscalls: _L1Syscalls,
        parent_fd: int,
        lock_fd: int,
        created: _Snapshot,
    ) -> None:
        self._server = server
        self._syscalls = syscalls
        self._parent_fd = parent_fd
        self._lock_fd = lock_fd
        self._created = created
        self._closed = False

    async def close(self) -> None:
        """Close serving, safely remove the owned socket, then release the lease."""

        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        await _close_server(self._server, failures)
        try:
            await _cleanup_target(
                self._syscalls,
                self._parent_fd,
                self._created,
            )
        except BaseException as failure:
            failures.append(failure)
        _close_fd(self._syscalls, self._lock_fd, failures)
        _close_fd(self._syscalls, self._parent_fd, failures)
        _report_cleanup_failures(failures)


async def _cleanup_failed_construction(
    syscalls: _L1Syscalls,
    server: _Server | None,
    parent_fd: int | None,
    lock_fd: int | None,
    created: _Snapshot | None,
) -> list[BaseException]:
    failures: list[BaseException] = []
    if server is not None:
        await _close_server(server, failures)
    if created is not None and parent_fd is not None:
        try:
            await _cleanup_target(syscalls, parent_fd, created)
        except BaseException as failure:
            failures.append(failure)
    _close_fd(syscalls, lock_fd, failures)
    _close_fd(syscalls, parent_fd, failures)
    return failures


async def _construct_l1_listener(
    identity: ResolvedServiceIdentity,
    syscalls: _L1Syscalls,
) -> L1Listener:
    uid, gid = _identity_values(identity)
    parent_fd: int | None = None
    lock_fd: int | None = None
    server: _Server | None = None
    created: _Snapshot | None = None

    try:
        parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        parent_fd = syscalls.open_parent(_PARENT_PATH, parent_flags)
        parent = syscalls.fstat(parent_fd)
        _require(
            _is_exact(parent, stat.S_ISDIR, uid, gid, _PARENT_MODE),
            "runtime parent failed validation",
        )

        lock_flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
        lock_fd = syscalls.open_lock(_LOCK_BASENAME, lock_flags, _LOCK_MODE, parent_fd)
        lock = syscalls.fstat(lock_fd)
        _require(
            _is_exact(lock, stat.S_ISREG, uid, gid, _LOCK_MODE),
            "instance lock failed validation",
        )
        syscalls.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        existing = syscalls.inspect_target(_SOCKET_BASENAME, parent_fd)
        if existing is not None:
            _require(
                _is_exact(existing, stat.S_ISSOCK, uid, gid, _SOCKET_MODE),
                "existing socket target failed validation",
            )
            syscalls.boundary("check-to-stale-probe")
            current = syscalls.inspect_target(_SOCKET_BASENAME, parent_fd)
            _require_same_socket(current, existing, uid, gid, exact_mode=_SOCKET_MODE)
            result = await syscalls.probe(_SOCKET_PATH, _PROBE_TIMEOUT_SECONDS)
            _require(result == errno.ECONNREFUSED, "existing socket is not proven stale")
            syscalls.boundary("stale-probe-to-reinspect")
            current = syscalls.inspect_target(_SOCKET_BASENAME, parent_fd)
            _require_same_socket(current, existing, uid, gid, exact_mode=_SOCKET_MODE)
            syscalls.boundary("reinspect-to-unlink")
            current = syscalls.inspect_target(_SOCKET_BASENAME, parent_fd)
            _require_same_socket(current, existing, uid, gid, exact_mode=_SOCKET_MODE)
            syscalls.unlink_target(_SOCKET_BASENAME, parent_fd)

        server = await syscalls.start_server(_close_accepted, _SOCKET_PATH, False)
        created = syscalls.inspect_target(_SOCKET_BASENAME, parent_fd)
        _require(
            created is not None
            and stat.S_ISSOCK(created.mode)
            and created.uid == uid
            and created.gid == gid,
            "bound socket failed initial validation",
        )
        assert created is not None
        syscalls.boundary("bind-to-chmod")
        current = syscalls.inspect_target(_SOCKET_BASENAME, parent_fd)
        _require_same_socket(current, created, uid, gid, exact_mode=None)
        syscalls.chmod_target(_SOCKET_BASENAME, _SOCKET_MODE, parent_fd)
        syscalls.boundary("chmod-to-reinspect")
        current = syscalls.inspect_target(_SOCKET_BASENAME, parent_fd)
        created = _require_same_socket(current, created, uid, gid, exact_mode=_SOCKET_MODE)
        syscalls.boundary("reinspect-to-start-serving")
        current = syscalls.inspect_target(_SOCKET_BASENAME, parent_fd)
        created = _require_same_socket(current, created, uid, gid, exact_mode=_SOCKET_MODE)
        await server.start_serving()
        return L1Listener(server, syscalls, parent_fd, lock_fd, created)
    except BaseException as failure:
        cleanup_failures = await _cleanup_failed_construction(
            syscalls,
            server,
            parent_fd,
            lock_fd,
            created,
        )
        for cleanup_failure in cleanup_failures:
            try:
                failure.add_note(f"cleanup failure: {type(cleanup_failure).__name__}")
            except Exception:
                pass
        if not isinstance(failure, Exception):
            raise
        if isinstance(failure, ListenerConstructionError):
            raise
        raise ListenerConstructionError("canonical listener construction failed") from None


async def construct_l1_listener(identity: ResolvedServiceIdentity) -> L1Listener:
    """Construct the one canonical production listener with no injectable path."""

    return await _construct_l1_listener(identity, _RealSyscalls())
