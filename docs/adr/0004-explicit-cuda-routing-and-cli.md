# ADR 0004: explicit CUDA routing and command-line boundary

- Status: accepted
- Contract version: `v1`
- Decision date: 2026-08-14

## Context

The package has a qualified deterministic CUDA engine, but `solve()` is intentionally the CPU-reference convenience route. Phase 6 makes CUDA usable from the public Python API and a local command-line interface without changing that default or introducing automatic routing, retry, or fallback.

The frozen v1 request, result, and problem schemas remain authoritative. This ADR defines process-local routing and CLI framing only; it does not add contract fields or problem codes.

## Decision

### Python API

The package root exports `Engine`, `CPUReferenceEngine`, `CUDAEngine`, `solve`, `solve_cuda`, and `serialize_equity_result` in addition to the existing contract exports.

- `solve(request)` always delegates to the CPU-reference engine. A valid request whose `backend` is `cuda` fails with `BACKEND_UNAVAILABLE` before CPU simulation.
- `solve_cuda(request)` accepts only an exact, valid `EquityRequest` whose backend is `cuda`. It creates `CUDAEngine` only after fixed-authority request validation and routing checks. It never calls the CPU engine and never falls back.
- `serialize_equity_result(result, request)` accepts exact model types, reconstructs the complete v1 result wire from authoritative fields, and revalidates it with `EquityResult.parse(raw, request=request)`. Ordinary failures map to `INTERNAL_ERROR`; process-control `BaseException` signals propagate.
- Importing the package root, engine classes, or CLI does not import CuPy, compile CUDA, inspect a device, or construct a CUDA runtime.

### CLI commands

The wheel and sdist provide equivalent entry points:

```text
poker-knight-ng solve
poker-knight-ng solve-cuda
python -m poker_knight_ng solve
python -m poker_knight_ng solve-cuda
```

`--help` is the only non-JSON stdout mode. Commands accept no options. Missing or unknown commands are invocation failures.

Routing is fixed:

| command | request backend | behavior |
|---|---|---|
| `solve` | `cpu_reference` | CPU-reference execution |
| `solve` | `cuda` | `BACKEND_UNAVAILABLE`; no CUDA or CPU execution |
| `solve-cuda` | `cuda` | explicit CUDA execution |
| `solve-cuda` | `cpu_reference` | `UNSUPPORTED_REQUEST`; no engine construction |

### Stdin profile

A solve command reads one request from stdin as bytes:

- maximum `16,384` bytes, enforced by reading at most `16,385` bytes;
- valid UTF-8 without a byte-order mark;
- one JSON object and no trailing non-whitespace document;
- duplicate members rejected at every depth;
- `NaN`, positive or negative `Infinity`, and malformed JSON rejected;
- ordinary JSON whitespace and object-member order are accepted;
- semantic validation is delegated to `EquityRequest.parse()`.

Malformed framing, invalid UTF-8, duplicate members, non-object roots, and oversized input map to `UNSUPPORTED_REQUEST`. Frozen request validation errors keep their existing problem codes.

### Output and errors

Success writes exactly one canonical v1 result JSON line to stdout and nothing to stderr. Canonical JSON uses sorted keys, compact separators, ASCII escaping, finite values only, and one trailing newline.

Handled failure writes no stdout and exactly one canonical `ContractProblem.serialize(correlation_id)` line to stderr. It includes no `field_errors`, traceback, raw exception text, input values, paths, device diagnostics, or CLI metadata.

A cryptographically random `pk_` plus 32 lowercase hexadecimal correlation ID is allocated for each handled failure. If the ID factory itself raises an ordinary exception, the failure becomes `INTERNAL_ERROR` and uses the reserved emergency ID `pk_00000000000000000000000000000000`.

Exit statuses are closed:

| exit | meaning |
|---:|---|
| `0` | solve success or explicit help |
| `1` | silent terminal write failure |
| `2` | invocation or stdin framing failure |
| `3` | non-retryable request/contract failure |
| `4` | retryable operational failure: `BACKEND_UNAVAILABLE` or `RESOURCE_EXHAUSTED` |
| `5` | `INTERNAL_ERROR` or `RNG_REJECTION_EXHAUSTED` |

`KeyboardInterrupt`, `SystemExit`, `GeneratorExit`, and other `BaseException` process-control signals are never converted into contract errors. Broken stdout/stderr writes terminate silently with exit 1 and never trigger a second response.

## Consequences

CUDA use is explicit in both Python and shell interfaces. CPU remains the stable default. The CLI is local and synchronous; it adds no network service. Strict framing and closed output make the interface scriptable without exposing runtime details, while accepting normal JSON whitespace keeps it usable by humans and ordinary tooling.
