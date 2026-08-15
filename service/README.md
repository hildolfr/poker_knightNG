# Poker Knight NG private service

This separate distribution contains the bounded private HTTP/1.1 service runtime for Poker Knight NG.

Current Phase 7B-A scope is limited to raw request admission and response framing. It does **not** contain a socket listener, engine adapter, activation unit or deployment configuration.

The service uses the exact `h11 0.16.0` wheel selected by Spike 001. Its independent lock preserves the qualified root engine package and root lock byte-for-byte.
