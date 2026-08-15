# Spike 001: Phase 7B HTTP runtime selection

## Verdict: VALIDATED

Direct **h11 0.16.0** is selected as the sans-I/O HTTP/1.1 parser/serializer for the Phase 7B private Unix-socket service. It is not a server and does not satisfy ADR 0005 by itself. The implementation must put bounded **raw framing admission** in front of h11, own AF_UNIX I/O and deadlines, emit one response, and close the connection.

This spike opened **no listener**, bound no TCP or Unix socket, constructed no service, and invoked no equity engine. Its authority base is protected `main` commit `6c75cf7b260a1a83f2fbc8cba7cd81c5d1198d70`.

## Candidate matrix

| Candidate | Version | Verdict | Reason |
|---|---:|---|---|
| direct h11 | **h11 0.16.0** | FIT | Pure-Python sans-I/O parser, no dependencies, exact I/O/lifecycle remains under the adapter's control |
| Uvicorn | **Uvicorn 0.52.3** with h11 | NO FIT | ASGI begins after security-relevant normalization; no supported header deadline/raw aggregate accounting; parser errors emit non-empty plaintext; keep-alive and generic concurrency behavior conflict |
| aiohttp | **aiohttp 3.14.3** | NO FIT | Supported APIs cannot enforce the raw aggregate/name/header deadline before parsing; parser errors and pipeline queue conflict; substantially larger compiled dependency surface |

The selected h11 wheel is `h11-0.16.0-py3-none-any.whl`, 37,515 bytes, SHA-256 `63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86`. It requires Python 3.8 or newer and declares no dependencies. Poker Knight NG retains `dependencies = []`; the root `pyproject.toml` and `uv.lock` remain byte-identical because they are part of the published Phase 5 authority closure. CI downloads the exact wheel URL, verifies size and SHA-256, installs that local artifact into a disposable venv, and executes the probe with that interpreter. Service dependency packaging is deferred to the implementation boundary and must not weaken historical verifiers.

## Reproduced behavior

The committed no-listener probe confirms:

- a valid POST emits `Request`, body `Data`, then `EndOfMessage`;
- malformed request headers raise `RemoteProtocolError`;
- conflicting `Content-Length` fields are rejected;
- equal duplicate `Content-Length` fields are accepted and normalized to one field;
- `Transfer-Encoding: chunked` is accepted and decoded;
- `Transfer-Encoding: chunked` plus `Content-Length` is accepted;
- a second pipelined request is retained in `Connection.trailing_data` after the first request pauses;
- an explicit `Connection: close` response is serialized as directed.

The permissive cases are intentional negative evidence. They prove the adapter cannot treat “h11 accepted it” as “ADR 0005 admitted it.”

## Mandatory adapter boundary

Before any request model parsing or engine construction, Phase 7B implementation must:

1. read a bounded raw header block under the five-second monotonic deadline;
2. enforce the 8,192-byte aggregate, 32-field, 128-byte name and 1,024-byte value limits incrementally;
3. require HTTP/1.1 and an allowed exact method/path;
4. require exactly one canonical decimal `Content-Length` from 1 through 16,384 for POST;
5. reject every `Transfer-Encoding`, unsupported `Content-Encoding`, upgrade/proxy/CORS surface, and unsupported media type;
6. feed the same admitted bytes to h11 with `max_incomplete_event_size=8192` as defense in depth;
7. collect exactly the declared body under a separate five-second deadline;
8. reject incomplete, oversized or surplus bytes before request parsing;
9. inspect `Connection.trailing_data`, reject any nonempty bytes, and **do not call `start_next_cycle()`**;
10. serialize explicit `Connection: close` and close the underlying stream after exactly one response.

The raw scanner is security-sensitive but intentionally narrow: it owns only request-line/header framing admission. h11 remains the HTTP grammar/event authority after those stricter checks. The implementation must regression-test segmented CRLF, whitespace, duplicate fields, every limit boundary, body packetization, surplus bytes and timeout boundaries.

## Deferred implementation gates

This spike validates runtime selection only. It does **not** authorize a listener. Before any AF_UNIX bind, implementation still requires:

- RED/GREEN framing and lifecycle tests;
- proof of the 16-connection admission limit and one global no-queue solve token;
- filesystem ownership/mode/symlink/stale-socket tests;
- disconnect continuation and shutdown-drain tests;
- root-import and base-wheel dependency inertness;
- independent security and compatibility review.

Uvicorn and aiohttp must not be introduced as alternative service paths under the v1 profile without a new reviewed decision and equivalent raw-boundary proof.

## Reproduction

From a frozen development environment:

```text
wheel=/tmp/h11-0.16.0-py3-none-any.whl
venv=/tmp/phase7b-h11-venv
curl --fail --location \
  https://files.pythonhosted.org/packages/04/4b/29cac41a4d98d144bf5f6d33995617b185d14b22401f75ca86f384e87ff1/h11-0.16.0-py3-none-any.whl \
  --output "$wheel"
test "$(stat -c %s "$wheel")" = 37515
printf '%s  %s\n' 63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86 "$wheel" | sha256sum --check -
uv venv --python 3.13 "$venv"
uv pip install --python "$venv/bin/python" "$wheel"
"$venv/bin/python" tools/spikes/phase7b_h11_probe.py
```

The probe prints canonical JSON. Its exact expected bytes are committed at `validation/service/v1/phase7b_runtime_spike.json` and bound by `phase7b_runtime_spike.sha256`.
