# ADR 0008: automatic CUDA routing by request backend

- Status: accepted
- Contract version: `poker-knight-ng-private-http-v1`
- Decision date: 2026-08-17

## Context

ADR 0005 froze the initial service profile as an explicit-route-only design: `/v1/solve`
was CPU-only and `/v1/solve-cuda` explicit.

This ADR does not modify the v1 request, result, or problem schema.

Roadmap phase 7C requires automatic CUDA routing so clients can keep a single solve
endpoint and still preserve deterministic engine selection. This ADR scopes automatic
routing without weakening any schema:

This also defines explicit runtime selection by request `backend` while preserving
all existing transport constraints.

- do not change `contracts/v1/equity-request.schema.json`
- do not add any new request/transport fields
- keep `/v1/solve-cuda` explicit route for direct CUDA invocation
- keep `/v1/solve` as automatic CPU vs CUDA by request backend

## Amendment (supersession of ADR 0005 routing constraints)

This ADR, as accepted and implemented, **supersedes** the following previously
binding text in ADR 0005:

- **ADR 0005 §3 ("Routes and explicit backend selection")** routing text, in
  particular the frozen route table that bound `POST /v1/solve` to
  `cpu_reference` only and declared "There is no backend-neutral route,
  selection header, retry route, or automatic CUDA routing." Under this ADR,
  `/v1/solve` is the automatic backend-selection route and binds by request
  `backend` (`cpu_reference` -> CPU, `cuda` -> CUDA), with no fallback, no trial
  reduction, and no cross-backend retry.
- **ADR 0005 §7 ("Responses and correlation")** problem-mapping bullet
  "`cuda` on the CPU route: `BACKEND_UNAVAILABLE` / 503" for the
  cuda-on-`/v1/solve` case. Under this ADR, the `cuda` backend is a valid
  selection on `/v1/solve`, so a `cuda` request routed to `/v1/solve` is not a
  backend-mismatch failure; the retained mapping for that case is
  solve-route-invalid-backend -> `UNSUPPORTED_REQUEST` / 400, per the frozen
  profile. The `cpu_reference` on `/v1/solve-cuda` -> `UNSUPPORTED_REQUEST`
  mapping remains unchanged.

All other ADR 0005 constraints remain binding and are not modified by this
amendment: the private AF_UNIX-only transport, authorization boundary, single
request per connection, framing/header/body bounds, 16-connection admission
with zero queued and one in-flight solve, no execution timeout, drain shutdown,
correlation IDs, logging privacy, healthz semantics, and the 1,000,000 trial
ceiling. ADR 0005 §9's Phase 7B implementation gate is satisfied by the
listener work delivered under this phase and is superseded only as to its
"no listener may be deployed" pre-gate language, which no longer applies once
the bounded listener is implemented.

## Decision

1. `/v1/solve` now binds by request `backend`:
   - `cpu_reference` -> CPU engine path
   - `cuda` -> CUDA engine path
   No route fallback, no backend trial reduction, and no cross-backend retry.
2. `/v1/solve-cuda` remains explicit and still requires `backend`=`cuda`.
3. The profile `request_backend` for `/v1/solve` is `auto` in
   `contracts/service/v1/http-service-profile.json`.
4. The contract remains closed and private:
   - AF_UNIX only, private group auth, zero queue, one in-flight solve
   - no fallback, no retry, no partial success, no streaming
   - explicit execution timeout remains `no execution timeout` and 1,000,000 max `requested_trials`
   - shutdown is `client disconnect` then admission drain behavior
   - protocol exposure remains limited to AF_UNIX: TCP, UDP, HTTP/2, reverse proxy exposure, and public internet exposure stay forbidden
5. Mapping retains canonical v1 problem codes from runtime (`RESOURCE_EXHAUSTED`, `UNSUPPORTED_REQUEST`) with no fallback.

## Consequences

- This is a contract-internal routing decision only; v1 request, result, and problem
  schemas remain unchanged.
- Route execution errors remain canonical v1 problems and statuses, mapped through
  existing failure handling.
- Roadmap checkpoints Phase 7B (listener runtime admission baseline), 7C (contract), and 7D (runtime implementation) are
  satisfied together under this branch and remain backward-compatible with existing
  `/v1/solve-cuda` callers.