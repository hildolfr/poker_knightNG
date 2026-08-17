# ADR 0007: bounded Unix-listener construction

- Status: accepted
- Construction profile: `poker-knight-ng-unix-listener-construction-v1`
- Decision date: 2026-08-15

## Context

ADR 0005 freezes a private AF_UNIX HTTP/1.1 service at
`/run/poker-knight-ng/service.sock`, owned by the dedicated
`poker-knight-ng` service identity, with parent mode `0750`, socket mode
`0660`, no inherited listener, one response per connection, at most sixteen
open connections, one solve in flight and bounded graceful shutdown. ADR 0006
places the HTTP runtime in the isolated service distribution.

The framing, connection, routing, adaptation, admission, execution and
one-request session boundaries are already implemented. A listener still
requires construction rules that ADR 0005 did not make executable:

1. the production identity cannot be chosen by an arbitrary caller;
2. stale pathname cleanup is unsafe without single-instance ownership;
3. pathname checks are not atomic in a namespace writable by another process;
4. public `asyncio.StreamReader` has no supported synchronous buffer inspector;
5. the writer-side peer-loss and close lifecycle need a closed taxonomy; and
6. bind, session admission and shutdown are separate security surfaces.

The adjacent machine-readable authority is
`contracts/service/v1/unix-listener-construction.json`, with its exact bytes
bound by `unix-listener-construction.sha256`.

## Decision

### 1. Staged checkpoints

Listener work remains split into three separately reviewed checkpoints:

1. **L1 — secure bind and stream adapter**: resolve the fixed service identity,
   acquire the lifetime instance lock, validate and prepare the AF_UNIX path,
   bind without accepting before postconditions, and implement the public
   stream adapter. Accepted streams are closed without session scheduling.
2. **L2 — bounded session manager**: own accepted-session tasks, enforce the
   sixteen-open-connection cap with immediate close and zero queue, invoke
   `handle_one_session()` exactly once, and reap tasks.
3. **L3 — graceful lifecycle drain**: stop admission, close the listener, drain
   ordinary accepted work for up to five seconds. If an admitted
   non-cancellable solve remains after that deadline, wait without a deadline;
   forced solve termination is forbidden. This preserves ADR 0005 exactly.

No deployment, systemd unit, logging, metrics, automatic routing, TCP, TLS or
Phase 7C behavior belongs to L1. **Socket activation is forbidden**: this
runtime rejects inherited listeners, and only its own
`construct_listener_with_callback` path may construct the production listener.

### 2. Opaque resolved service identity

Production resolution uses the fixed account and group names
`poker-knight-ng`. The resolver verifies both records exist, verifies the
account's primary GID equals the resolved group GID, and validates every
numeric value with `type(value) is int and value >= 0`; booleans, `IntEnum`
values and integer subclasses are rejected. Resolution failure cannot fall
back to the effective process identity.

Successful resolution returns an opaque resolved service identity token whose
constructor is not public. The bind constructor accepts that token, not raw
UID/GID values. Arbitrary caller-selected numeric identities are forbidden.
Tests replace the fixed name-resolution dependencies; they do not bypass the
token authority.

Production L1 has one canonical production socket path:
`/run/poker-knight-ng/service.sock`. The production bind constructor has no
path argument, and no caller may select another parent or instance lock.
Deterministic unit tests use a closed syscall harness that substitutes
filesystem operations; this harness cannot override the production path or
mint a production identity token.

### 3. Exclusive namespace and lifetime instance lock

The Unix permission model treats every process with the dedicated service UID
as the same trusted principal. A hostile process already running under that
UID is outside this boundary because it can modify any service-owned runtime
state regardless of this listener. The parent hierarchy is provisioned by an
administrator, every ancestor is a non-symlink directory not writable by the
service identity, and the immediate parent is the only service-writable
component. It must be a non-symlink directory owned by the resolved service
UID/GID with exact mode `0750`.

Construction opens the immediate parent with
`O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`, verifies it with `fstat`, and retains that
directory FD for the complete listener lifecycle. Because the production path
is fixed, this parent defines the one service-instance lock namespace. L1 then
opens `service.lock` relative to that FD with `O_CREAT|O_CLOEXEC|O_NOFOLLOW`, mode
`0600`, verifies a regular file with the expected UID/GID and exact mode, and
acquires `flock(LOCK_EX|LOCK_NB)` before inspecting the socket basename. This
lifetime instance lock remains held through bind, serving, shutdown, pathname
cleanup and final FD close.

The lock is the ownership/lease protocol for cooperating service invocations.
A bound or starting invocation holds it, so another invocation cannot classify
that socket as stale merely because it is not accepting yet. Failure to
acquire it is terminal and performs no socket inspection or mutation.

Manual checks, chmod and unlink operations are descriptor-anchored and relative
to the retained parent FD with no symlink following. The stdlib
`asyncio.start_unix_server(path=..., start_serving=False)` bind is necessarily
path-based; it is permitted only while the lifetime lock is held and the
exclusive namespace preconditions above remain true. A target replacement
detected before any mutation fails without mutation. Under the declared
exclusive namespace, no cooperating process can replace the target between a
check and operation.

### 4. Existing target and bounded stale proof

With the lifetime lock held, the socket basename is inspected with
`lstat`/`stat(..., follow_symlinks=False)` relative to the parent FD.

- absence permits bind;
- a symlink, non-socket, wrong UID/GID or wrong mode fails without mutation;
- a matching socket is probed using `AF_UNIX`,
  `SOCK_STREAM|SOCK_NONBLOCK`, an exact 250 ms monotonic deadline and mandatory
  probe-socket close;
- a successful connection proves a live listener and fails without unlink;
- only `ECONNREFUSED`, while the lifetime lock is held, proves a stale artifact;
- timeout and every other result fail closed without unlink.

Before stale unlink, the target is re-inspected relative to the retained parent
FD and must still be the same device/inode, socket type, UID/GID and mode. The
unlink is relative to that FD while the lock is held. Any observed replacement
fails without removal. Transaction fault tests must inject replacement before
every reinspection/mutation boundary; no replacement may be removed.

### 5. Paused bind, postconditions and cleanup

L1 creates the server with
`asyncio.start_unix_server(..., start_serving=False)`. Acceptance remains
paused. While holding the lock, construction sets exact mode `0660` without
following symlinks and re-inspects the path. The target must be the same socket
created by this invocation, with expected device/inode, UID/GID and mode.
Only then may `Server.start_serving()` run.

A postcondition failure closes the server first. Cleanup is attempted only
while the lifetime lock is held and only when descriptor-relative inspection
still identifies the same device/inode socket created by this invocation.
Otherwise cleanup leaves the pathname untouched and reports failure rather
than risking removal of another object. The lock FD closes only after listener
cleanup completes.

### 6. Public stream adapter and one-byte read-ahead

The adapter wraps only public `asyncio.StreamReader` and
`asyncio.StreamWriter` interfaces. Access to `_buffer`, transport internals or
other private state is forbidden.

`read(limit)` accepts only exact integers in the existing bounded reader range.
When no adapter-owned overflow exists, it requests at most `limit + 1` bytes
from `StreamReader.read()`, returns at most `limit`, and retains at most one
surplus byte. When overflow exists, it is returned before another underlying
read. EOF remains `b""`. Synchronous `read_buffered(limit)` consumes only this
adapter-owned overflow. This adapter-owned one-byte read-ahead lets the current
connection boundary reject already-buffered pipelining without private
`StreamReader` access.

`send_response(response)` validates bytes, calls `writer.write(response)`
exactly once, then `await writer.drain()` exactly once. `EPIPE`, `ECONNRESET`,
`ECONNABORTED` and `ENOTCONN` from write, drain or wait-closed are peer loss and
return `False`; no retry or second write occurs. Every unlisted ordinary
failure propagates. `BaseException` always propagates.

`close()` calls `writer.close()` at most once; repeated adapter close calls are
no-ops. Lifecycle ownership then invokes `await writer.wait_closed()` at most
once. The same peer-loss errno set is non-fatal there; unlisted ordinary
failures and `BaseException` propagate. L2/L3 will define how a simultaneous
primary failure and wait-closed failure are prioritized.

### 7. L1 accept behavior

L1 may prove bind/accept only with an internal callback that closes the writer
without adapting a request, allocating a request ID, acquiring solve admission,
constructing an engine or scheduling a session task. Connection-cap admission
is L2. Graceful shutdown admission and drain are L3.

No listener code is authorized by this decision alone. L1 implementation
starts only after this ADR, profile, manifest and authority test pass
independent specification and adversarial review.

### 8. Required L1 fault matrix

L1 tests must reject `bool`, integer subclasses and `IntEnum` values for every
numeric authority field. They must cover unknown account or group names,
primary-group mismatch and attempted raw-numeric identity construction.

Deterministic hooks must inject target replacement at each named boundary:
check-to-stale-probe, stale-probe-to-reinspect, reinspect-to-unlink,
bind-to-chmod, chmod-to-reinspect, reinspect-to-start-serving and
cleanup-reinspect-to-unlink. Every observed replacement fails without removing
it. Stale probing must cover connection success, `ECONNREFUSED`, timeout,
unexpected errno and mandatory probe close.

Stream tests must cover one-byte read-ahead, every listed peer-loss errno at
write/drain/wait-closed, unlisted ordinary failures, `BaseException`, repeated
close and repeated wait-closed.

## Consequences

The lifetime lock and explicit Unix-principal threat model make stale cleanup
implementable without claiming protection from a hostile process that already
has the service UID. Descriptor anchoring limits pathname operations to the
validated parent. The opaque identity token prevents a production caller from
choosing convenient UID/GID values. The reader and writer contracts are both
machine-bound before adapter code exists.

The staged plan intentionally spends separate reviews on namespace mutation,
accepted-session concurrency and shutdown drain. L1 cannot be described as the
complete ADR 0005 service.
