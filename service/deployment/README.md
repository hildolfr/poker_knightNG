# Service deployment artifacts

This directory contains rollout assets for Phase 7F delivery work.

## What is included

- `systemd/poker-knight-ng.service` — direct runtime unit for `poker-knight-ng-service`.
- `systemd/poker-knight-ng.socket` — socket-activation unit stub for later hardening.

## Runtime unit behavior (current)

The provided systemd unit starts the service directly via
`poker-knight-ng-service --max-sessions 16 --graceful-drain-seconds 5`.
This is compatible with the current bounded listener contract and enforces:

- fixed, exact ownership modes on `/run/poker-knight-ng`
- restart policy for transient failures
- a non-privileged runtime profile boundary

## Current roadmap constraints

Socket activation remains implemented as a **disabled roadmap artifact** until
runtime conformance for shutdown/admission boundaries completes under ADR 0005.

## Local install sketch

From the repository root:

```bash
mkdir -p /opt/poker-knight-ng
cp service/deployment/systemd/poker-knight-ng.service /etc/systemd/system/
# optional for later: cp service/deployment/systemd/poker-knight-ng.socket ...
systemctl daemon-reload
systemctl enable --now poker-knight-ng.service
```

## Tag/release artifact notes

Use the project Git workflow tags and release process for immutable artifacts.
Roadmap-phase 7F tracks this as the next checkpoint before PyPI publication
and full runtime activation policy are implemented.
