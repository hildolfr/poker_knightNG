# ADR 0005: private Unix-socket HTTP service boundary

- Status: accepted
- Contract version: `poker-knight-ng-private-http-v1`
- Decision date: 2026-08-15

## Context

Poker Knight NG now has deterministic CPU and explicitly selected CUDA engines,
a bounded local CLI, qualified artifacts, and protected release history. The next
roadmap checkpoint is a service boundary. ADR 0001 still excludes public internet
exposure, accounts, multitenancy, streaming, WebSockets, caching, adaptive trial
counts, and silent fallback. ADR 0004 deliberately stops at a process-local API
and CLI.

A conventional TCP API cannot be added honestly without new authentication,
proxy trust, cancellation, transport-error, and deployment contracts. The frozen
v1 problem schema has no 401, 403, 404, 405, 408, 413, 415, 429, 431, or
cancellation problem branches. Mapping those conditions to unrelated v1 problem
codes would weaken the closed contract.

This decision therefore freezes a smaller local service: HTTP/1.1 carried only
over a protected Unix-domain socket. It is private IPC, not a public or remote
network API. The machine-readable authority is
`contracts/service/v1/http-service-profile.json`, bound by its adjacent SHA-256
manifest.

## Decision

### 1. Authority and compatibility

This ADR does not modify the frozen v1 request, result, or problem schemas. The
service profile binds their exact paths and SHA-256 digests. A successful solve
body is the canonical v1 result JSON line. A domain or engine failure body is the
canonical v1 problem JSON line. The root package remains service-runtime inert
and keeps zero base dependencies.

The service adds transport semantics only. Automatic CUDA routing, fallback,
retry, trial reduction, partial success, and result-schema extensions remain
forbidden.

### 2. Private transport and authorization

The listener MUST be one HTTP/1.1 server on an AF_UNIX socket. Its default path
is `/run/poker-knight-ng/service.sock`. TCP, UDP, HTTP/2, TLS termination,
reverse proxy exposure, container port publication, WebSockets, upgrades, CORS,
and public internet exposure are forbidden by this profile.

The dedicated service user and group are both `poker-knight-ng`. The socket
parent directory MUST be an owned, non-symlink directory with mode `0750`. The
socket MUST be a real socket, never a symlink, and mode `0660`. Membership of the
dedicated local group is the authorization boundary. There is no HTTP bearer
token, account, tenant, forwarded identity, or client-supplied authorization
metadata.

An implementation MUST fail closed before listening if the configured path is
relative, its parent or socket is a symlink, ownership/mode is wrong, the path is
already occupied by a non-socket, or a TCP/inherited network listener is supplied.
A stale socket may be removed only after proving it is an owned socket at the
exact configured path and no listener accepts a connection.

### 3. Routes and explicit backend selection

Exactly three routes exist:

| method | path | accepted backend | behavior |
|---|---|---|---|
| `GET` | `/healthz` | none | return 204 with an empty body; never import/probe CUDA or report capacity |
| `POST` | `/v1/solve` | `cpu_reference` | call the existing CPU `solve` route |
| `POST` | `/v1/solve-cuda` | `cuda` | call the existing explicit `solve_cuda` route |

A valid `cuda` request sent to `/v1/solve` fails with canonical
`BACKEND_UNAVAILABLE` before CPU or CUDA execution, preserving ADR 0004. A valid
`cpu_reference` request sent to `/v1/solve-cuda` fails with canonical
`UNSUPPORTED_REQUEST` before either engine is constructed. There is no
backend-neutral route, selection header, retry route, or automatic CUDA routing.
Unknown paths and wrong methods are transport failures with empty 404 and 405
responses respectively; they are not fabricated v1 problem documents.

### 4. HTTP framing

Every connection handles exactly one request and then closes. Keep-alive,
pipelining, upgrades, and request reuse are forbidden.

Headers are bounded before body admission:

- 8,192 aggregate bytes;
- at most 32 fields;
- field names at most 128 bytes;
- field values at most 1,024 bytes; and
- five seconds to complete the header block.

POST requests require exactly one canonical decimal `Content-Length` in
`1..16384`, no `Transfer-Encoding`, and no non-identity `Content-Encoding`.
Accepted media types are exactly `application/json` and
`application/json; charset=utf-8`. A body has five seconds to complete. The
implementation reads exactly the declared bytes and rejects incomplete,
oversized, conflicting, or surplus framing.

Header/framing failures return empty 400; header overflow returns empty 431;
payload overflow returns empty 413; body-read timeout returns empty 408; and
unsupported media or encoding returns empty 415. Transport failures do not claim
to satisfy `problem.schema.json`.

The POST entity uses the ADR 0004 JSON profile: required UTF-8, no BOM, one object
root, no duplicate members at any depth, no non-finite constants, and no trailing
document. Once framing is accepted, malformed JSON maps to canonical
`UNSUPPORTED_REQUEST`. Semantic parsing remains `EquityRequest.parse()` and
preserves all frozen v1 validation codes.

### 5. Resource admission

The service admits at most 16 open connections, zero queued solves, and one
in-flight solve globally across CPU and CUDA. Admission is atomic and occurs
before engine construction. A second solve while the token is held fails with
canonical `RESOURCE_EXHAUSTED`; it is never queued, retried, rerouted, or reduced.
Health checks do not consume the solve token.

The service profile accepts at most 1,000,000 requested trials. This is a service
policy ceiling, not a change to the wider process-local v1 request schema. The
value matches the largest preregistered and qualified Phase 5C/6C workload.
A schema-valid request above the service ceiling fails canonical
`UNSUPPORTED_REQUEST` before engine construction. `RESOURCE_EXHAUSTED` remains
reserved for transient occupied capacity or backend resource failure.

Increasing the trial ceiling, connection count, solve concurrency, or queue size
requires a new reviewed profile with CPU/CUDA resource evidence. It is not a
runtime tuning knob.

### 6. Disconnect, timeout, and shutdown

There is no execution timeout and no service-level cancellation operation. Once
a solve is admitted, a client disconnect does not cancel, shorten, retry, or
reroute it. The solve runs to its ordinary terminal result/problem; if the peer is
gone, the serialized response is discarded and only bounded metadata may be
logged.

Graceful shutdown stops new admissions, closes idle connections, and waits to
drain the one in-flight solve without a deadline. Forced process termination may
produce no response; it MUST NOT emit a partial or successful result. A future
bounded-cancellation design requires an engine/process-isolation contract and a
new explicit non-success semantic before it can replace drain behavior.

### 7. Responses and correlation

Accepted solve responses use `Content-Type: application/json`,
`Cache-Control: no-store`, `X-Content-Type-Options: nosniff`, and
`Connection: close`. JSON is canonical, ASCII-escaped, finite, compact,
sorted-key output with one trailing newline.

Every accepted solve receives a server-generated `pk_` plus 32 lowercase
hexadecimal correlation ID in `X-Poker-Knight-Request-ID`. On a v1 problem, the
header value equals the body `correlation_id`. A successful result is not extended
with a correlation field. ID generation failure uses the existing emergency-ID
and `INTERNAL_ERROR` behavior. Transport-only empty failures do not fabricate a
v1 correlation body.

Existing problem mappings remain authoritative:

- `cuda` on the CPU route: `BACKEND_UNAVAILABLE` / 503;
- `cpu_reference` on the CUDA route, malformed accepted-route JSON, or service
  trial cap: `UNSUPPORTED_REQUEST` / 400;
- unavailable explicit CUDA after accepted routing: `BACKEND_UNAVAILABLE` / 503;
- occupied solve token or backend resource exhaustion: `RESOURCE_EXHAUSTED` /
  503;
- deterministic rejection exhaustion: `RNG_REJECTION_EXHAUSTED` / 422; and
- unexpected ordinary adapter/engine/serialization failure: `INTERNAL_ERROR` /
  500.

`BaseException` process-control signals are never converted into public problems.

### 8. Logging and health privacy

Logs are bounded structured events containing only correlation ID, normalized
route name, and HTTP status. They MUST NOT contain request bodies, cards, seeds,
canonical hashes, socket peers, host paths, exception text, stack traces, backend
diagnostics, device identity, memory/capacity inventory, or raw headers.

`GET /healthz` proves only that the local HTTP loop can answer. It returns 204 and
an empty body through the same filesystem authorization boundary. It does not
initialize engines, inspect CUDA, report readiness/capacity, expose build or host
identity, or allocate a solve correlation ID.

### 9. Phase 7B implementation gate

This ADR freezes behavior but ships no listener. Phase 7B MUST begin with a
runtime selection spike that proves, before maintained implementation:

1. AF_UNIX-only binding and refusal of inherited/TCP listeners;
2. raw duplicate/framing/header/body limits and single-request connection close;
3. bounded slow-header/slow-body behavior;
4. exact empty transport statuses versus canonical v1 problem responses;
5. no root-import or base-dependency regression;
6. one-token admission under concurrent CPU/CUDA attempts;
7. disconnect continuation and shutdown drain behavior; and
8. socket ownership, mode, symlink, stale-path, and cleanup safety.

The chosen HTTP runtime MUST live in an optional service dependency group and be
pinned before implementation. If no candidate can satisfy the framing and Unix
socket contract without unsafe custom parsing, Phase 7B remains blocked. No
listener may be deployed during the spike.

## Consequences

The approved surface is intentionally smaller than a conventional web API. Local
operators gain a scriptable private service boundary while frozen equity
semantics remain unchanged. Remote access, mTLS, bearer credentials, reverse
proxy trust, bounded cancellation, greater concurrency, automatic CUDA routing,
and public exposure require separate future decisions and evidence.
