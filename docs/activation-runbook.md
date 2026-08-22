# Activation Runbook — Poker Knight NG deployment-model conformance

**Owner of this runbook:** the project owner / operator who holds root on the
deployment host. Steps marked **[OPERATOR]** must be performed by a human; they
cannot and must not be automated by the conformance harness or any CI.

**Branch context:** `revival/phase8-conformance-prep`. Everything below is
**repo-side prep plus a future live session**. Nothing in this file activates
the service; activation happens only after conformance evidence passes **and**
the owner approves, using the explicit sequence at the bottom.

**Source of truth for the gate:** `service/deployment/README.md` "Activation
gate (not yet satisfied by this scaffold)". The conformance harness
`tools/conformance_run.py` proves items 1–4 under the deployment process model
and emits schema-conformant evidence.

---

## 0. What conformance proves

The harness launches the **real** service entrypoint
(`python -m poker_knight_ng_service`) under the dedicated `poker-knight-ng`
identity, binds the canonical ADR 0007 path `/run/poker-knight-ng/service.sock`,
and runs every L2/L3 check over a real AF_UNIX HTTP/1.1 client:

| Check | Verifies |
|---|---|
| `healthz_204` | `GET /healthz` → HTTP 204 |
| `valid_solve_round_trip` | `POST /v1/solve` → HTTP 200 + valid v1 equity result |
| `socket_file_mode_owner_after_bind` | socket `0660 poker-knight-ng:poker-knight-ng`; parent `/run/poker-knight-ng` `0750` |
| `seventeenth_connection_rejected` | L2: 16 admitted, the 17th concurrent connection closed/rejected |
| `stale_socket_recovery` | a pre-placed stale socket is probed, unlinked, and rebound safely |
| `stop_during_idle_closes_promptly` | L3: stop() closes an idle service promptly (≤ 5 s grace) |
| `stop_with_admitted_solve_waits_past_grace` | L3: stop() drains an admitted solve past the 5 s grace indefinitely |
| `admission_stopped_during_shutdown` | L3: during shutdown, new requests fail fast |

The harness is **safe by default**: it refuses to run whenever the systemd
scaffold no longer has `RefuseManualStart=yes` / lacks `[Install]` (i.e. whenever
removing `RefuseManualStart` would be implied), never enables/starts/stops a
systemd unit, and touches nothing outside `/run/poker-knight-ng` and its own
output files under `validation/service/v1/`.

---

## 1. Provisioning the dedicated identity **[OPERATOR]**

Run as root on the target host.

```bash
# 1.1 Service user + group (system accounts, no interactive shell).
groupadd --system poker-knight-ng
useradd --system --home-dir /run/poker-knight-ng \
    --shell /usr/sbin/nologin poker-knight-ng

# 1.2 Runtime directory created at boot by systemd-tmpfiles, with the exact
#     owner/mode the listener requires (0750, poker-knight-ng:poker-knight-ng).
#     Drop this into /etc/tmpfiles.d/poker-knight-ng.conf:
#         d /run/poker-knight-ng 0750 poker-knight-ng poker-knight-ng -
printf '%s\n' \
    'd /run/poker-knight-ng 0750 poker-knight-ng poker-knight-ng -' \
    > /etc/tmpfiles.d/poker-knight-ng.conf

# 1.3 Apply now (and it will re-apply on every boot).
systemd-tmpfiles --create /etc/tmpfiles.d/poker-knight-ng.conf

# 1.4 Verify the provisioned posture.
getent passwd poker-knight-ng
getent group  poker-knight-ng
ls -ld /run/poker-knight-ng        # drwxr-x--- poker-knight-ng poker-knight-ng
```

**Verify:** `/run/poker-knight-ng` is owned by `poker-knight-ng:poker-knight-ng`
with mode `0750`.

---

## 2. One-command conformance run **[OPERATOR]**

Run from a checkout of this branch on the deployment host, as **root**.

```bash
cd /path/to/poker_knightNG
python3 tools/conformance_run.py
```

The script:
1. Validates preconditions (root/sudo, identity, runtime dir owner/mode,
   no live service, scaffold still inactive/unenableable). On any failure it
   prints operator-facing **remediation text** and exits nonzero.
2. Launches the real service, runs all 8 checks, and writes
   `validation/service/v1/deployment_conformance.json` (+ `.sha256` sidecar)
   conforming to `validation/service/v1/deployment_conformance.schema.json`.
3. Exits `0` only if every precondition and every check passes.

A `--dry-run` flag checks preconditions and writes evidence without launching
any service.

---

## 3. Expected evidence output

On a passing run the harness prints:

```
conformance: PASS
  [precondition] privileged: PASS - ...
  [precondition] identity_user: PASS - ...
  [precondition] identity_group: PASS - ...
  [precondition] runtime_dir: PASS - ...
  [precondition] no_live_service: PASS - ...
  [precondition] deployment_posture: PASS - ...
  [check] healthz_204: PASS - HTTP 204 (empty)
  [check] valid_solve_round_trip: PASS - HTTP 200 with valid v1 equity result
  [check] socket_file_mode_owner_after_bind: PASS - ...
  [check] seventeenth_connection_rejected: PASS - ...
  [check] stale_socket_recovery: PASS - ...
  [check] stop_during_idle_closes_promptly: PASS - ...
  [check] stop_with_admitted_solve_waits_past_grace: PASS - ...
  [check] admission_stopped_during_shutdown: PASS - ...
evidence: .../validation/service/v1/deployment_conformance.json
sidecar:  .../validation/service/v1/deployment_conformance.json.sha256
```

The evidence JSON carries: UTC timestamp, host kernel, effective UID/GID,
service UID/GID resolved numerically, per-check pass/fail + measured
`duration_ms`, the reviewed commit SHA (`git rev-parse HEAD`), and a summary.
The `.sha256` sidecar is the SHA-256 of the evidence file.

**Verify evidence integrity:**
```bash
cd validation/service/v1
sha256sum -c deployment_conformance.json.sha256
```

**Review gate:** the owner must inspect `deployment_conformance.json` and
confirm every check is `"passed": true` and `"summary": { "passed": true }`
before proceeding. **Conformance passing is a prerequisite, not a substitute
for owner approval.**

---

## 4. Activation (only after conformance passes AND owner approves) **[OPERATOR]**

These steps remove the inactive posture and start the service. **Never run them
before a passing conformance record and explicit owner sign-off.**

### 4.1 Remove the RefuseManualStart guard and add install target

Edit `service/deployment/systemd/poker-knight-ng.service`:

- Remove the line `RefuseManualStart=yes`.
- Add an `[Install]` section with a targeted target (private local service; the
  ADR 0007 listener binds its own AF_UNIX socket — **no `.socket` unit**):
  ```ini
  [Install]
  WantedBy=multi-user.target
  ```

Commit this change to the release branch with a reference to the conformance
evidence (reviewed commit SHA + `validation/service/v1/deployment_conformance.json`).

### 4.2 Validate, install, enable, start

```bash
# Static validation of the changed unit (no side effects).
systemd-analyze verify service/deployment/systemd/poker-knight-ng.service

# Install into the system unit directory.
install -o root -g root -m 0644 \
    service/deployment/systemd/poker-knight-ng.service \
    /etc/systemd/system/poker-knight-ng.service

systemctl daemon-reload
systemctl enable poker-knight-ng.service
systemctl start poker-knight-ng.service

# Confirm health over the canonical socket.
systemctl status poker-knight-ng.service --no-pager
curl --unix-socket /run/poker-knight-ng/service.sock \
    http://localhost/healthz -i   # expect HTTP 204
```

> **Important:** the service performs its own descriptor-anchored AF_UNIX bind.
> Do **not** add a `.socket` unit or `Sockets=` wiring; the runtime rejects
> inherited descriptors (ADR 0007) and the deployment validator enforces this.

### 4.3 Rollback

If activation fails or the service misbehaves:

```bash
systemctl stop poker-knight-ng.service
systemctl disable poker-knight-ng.service
rm -f /etc/systemd/system/poker-knight-ng.service
systemctl daemon-reload
# Restore RefuseManualStart=yes and remove the [Install] section in the repo,
# then re-run the conformance harness to reconfirm the inactive scaffold.
```

---

## 5. Safety boundaries (enforced)

- **No `RefuseManualStart` removal is ever implied by running conformance.**
  The harness refuses to run if the scaffold was already mutated toward
  activation.
- **The harness never** enables, starts, stops, reloads, or creates a systemd
  unit; it only inspects the checkout's unit text.
- **The harness touches nothing** outside `/run/poker-knight-ng` and its own
  output files (`validation/service/v1/*.json`, `*.json.sha256`,
  `conformance_run.log`, `conformance_instance*.log`). Runtime evidence files
  are git-ignored so fabricated proof is never committed.
- **Stale-socket recovery** is exercised against a socket the harness itself
  places inside `/run/poker-knight-ng`; it is removed after the check.
