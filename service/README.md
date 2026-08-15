# Poker Knight NG private service

This separate distribution contains the bounded private HTTP/1.1 service runtime for Poker Knight NG.

Current Phase 7B scope contains:

- raw request admission and exact single-request h11 parsing;
- incremental header/body reads with separate absolute monotonic deadlines;
- deterministic rejection of malformed, oversized, timed-out, surplus, pipelined, upgraded, transfer-encoded, or unsupported requests;
- exact route and method selection for `/healthz`, `/v1/solve`, and `/v1/solve-cuda`;
- strict UTF-8 JSON decoding with duplicate-member, BOM, non-finite, root, and trailing-document enforcement;
- semantic parsing through the frozen root `EquityRequest.parse()` contract;
- the 1,000,000-trial service ceiling and explicit CPU/CUDA route/backend binding;
- an atomic process-global one-solve lease with immediate zero-queue `RESOURCE_EXHAUSTED` rejection;
- exact CPU/CUDA engine construction and synchronous execution only while that lease is held;
- frozen request-bound result serialization and closed operational/internal problem mapping;
- async handoff to one directly started non-daemon worker thread only after event-loop admission, with no executor-side solve queue;
- cancellation-aware polling and join drain that keeps admission held until non-cancellable engine work ends;
- listener-free one-request session coordination across framing, routing, adaptation, execution, canonical response, send, and close boundaries;
- reload-stable weak-identity ownership that admits exactly one coordinator for each session object;
- peer-gone response discard after admitted solve completion without engine cancellation;
- empty health and transport-failure envelopes;
- canonical JSON solve envelopes with fixed cache, security, correlation, length, and close headers; and
- exact request-ID generation with fail-closed emergency-ID signaling.

The service distribution owns `h11==0.16.0` through the exact selected wheel URL and lockfile SHA-256. It now also owns an exact `poker-knight-ng==0.1.0` dependency as authorized by ADR 0006. Development resolves that dependency from the repository root. CI hash-verifies the selected h11 wheel, builds the engine and service wheels, and installs exactly those three local artifacts offline with an empty cache before checking dependency consistency and isolated imports.

The root engine distribution metadata and root lock remain unchanged. Public root import is engine-construction inert; the service's single reviewed execution module selects only `CPUReferenceEngine` or `CUDAEngine` after admission.

No listener, socket activation, socket disconnect integration, structured logging, deployment, or automatic backend routing exists yet. The execution boundary deliberately adds no service execution timeout and never cancels admitted engine work. Those remain later reviewed checkpoints under ADR 0005.
