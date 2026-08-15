# Poker Knight NG private service

This separate distribution contains the bounded private HTTP/1.1 service runtime for Poker Knight NG.

Current Phase 7B scope contains:

- raw request admission and response framing in `framing.py`;
- one-request incremental assembly in `connection.py`;
- one absolute five-second monotonic budget for the header and a fresh five-second budget for a declared body;
- bounded reads of at most the remaining payload plus one surplus-detection byte;
- a required nonblocking `read_buffered(1)` probe before semantic admission;
- the exact three-route table in `routing.py`, with unknown paths and wrong methods closed as empty 404/405 transport failures; and
- fixed health, transport-failure and canonical JSON response envelopes in `responses.py`, including closed status sets and correlation-ID binding.

The reader contract is intentionally abstract. `read(limit)` waits for at most `limit` bytes; `read_buffered(limit)` returns only bytes already buffered and never waits.

The package does **not** contain a socket listener, socket activation, semantic request adapter, engine adapter, engine invocation or deployment configuration.

The service uses the exact `h11 0.16.0` wheel selected by Spike 001. Its independent lock preserves the qualified root engine package and root lock byte-for-byte.
