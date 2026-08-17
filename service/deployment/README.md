# Service deployment artifacts

This directory contains rollout assets for Phase 7F delivery work.

## What is included

- `systemd/poker-knight-ng.service` — direct runtime unit for `poker-knight-ng-service`.
- `systemd/poker-knight-ng.socket` — socket-activation companion for on-demand startup.

## Runtime unit behavior (current)

The provided systemd unit starts the service directly and is wired to the socket
unit as a socket-activated on-demand service. In either launch mode, runtime
starts with:

`poker-knight-ng-service --max-sessions 16 --graceful-drain-seconds 5`.
This is compatible with the current bounded listener contract and enforces:

- fixed, exact ownership modes on `/run/poker-knight-ng`
- restart policy for transient failures
- a non-privileged runtime profile boundary

## Current roadmap state

Socket activation is active and bound to `poker-knight-ng.socket`, but the full
production hardening checklist in the roadmap remains the next checkpoint
before PyPI publication and automation layers.

## Local install sketch

From the repository root:

```bash
mkdir -p /opt/poker-knight-ng
cp service/deployment/systemd/poker-knight-ng.service /etc/systemd/system/
cp service/deployment/systemd/poker-knight-ng.socket /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now poker-knight-ng.socket
systemctl start poker-knight-ng.service
```

## Tag/release artifact notes

Use the project Git workflow tags and release process for immutable artifacts.
