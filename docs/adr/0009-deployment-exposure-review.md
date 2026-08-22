# ADR 0009: consolidated deployment exposure review

- Status: accepted-with-evidence
- Scope: `poker-knight-ng-deployment-exposure-v1`
- Decision date: 2026-08-21

## Context

ADR 0005 freezes a private AF_UNIX HTTP/1.1 service, ADR 0006 isolates its
distribution, ADR 0007 freezes bounded Unix-listener construction, and ADR 0008
adds automatic CUDA routing without opening a transport. The roadmap lists a
security exposure review as an outstanding review item, and `service/deployment/README.md`
item 4 requires an operator-approved rollout and rollback plan before any
activation.

This ADR is the consolidated, evidence-bound exposure review. It does **not**
add new security requirements. Every control it cites is already implemented
and enforced by a named test or validator artifact. Its purpose is to enumerate
the currently forbidden deployment surfaces, state the control that keeps each
closed, and name the exact artifact that enforces it, so that a later activation
decision has a single reviewable inventory rather than scattered claims.

The review scope is:

1. **framing / HTTP layer** — the transport framing and route surface;
2. **listener construction** — how the production AF_UNIX listener is built;
3. **systemd unit** — the deployment scaffold's hardening and activation posture;
4. **diagnostics surface** — the operator diagnostics snapshot; and
5. **filesystem boundary** — the runtime directory, socket, and lock namespace.

## Decision

### 1. Forbidden deployment surfaces

The following surfaces are **forbidden** and must remain absent or rejected by
the enforcing artifact named in the table below:

| forbidden surface | meaning | primary enforcing artifact |
|---|---|---|
| TCP | any AF_INET/AF_INET6 stream listener | `service/deployment/validate-systemd.sh` (`RestrictAddressFamilies must be AF_UNIX`) + `service/deployment/systemd/poker-knight-ng.service` (`RestrictAddressFamilies=AF_UNIX`) |
| UDP | any AF_INET/AF_INET6 datagram listener | same as TCP |
| TLS | any TLS/HTTPS terminator in front of the service | unit exposes no `ListenStream`/`ListenDatagram`/`ListenNetlink`/`Sockets`/`Bind`/`FileDescriptorName`; enforced by `test_shipped_unit_has_no_net_listening_directives` |
| proxy | reverse-proxy / load-balancer trust boundary | `service/tests/test_deployment_exposure.py::test_shipped_unit_has_no_net_listening_directives` |
| inherited listener | systemd socket activation / passed-in listener fds | `listener.py::_inherited_listener_fd` + `listener.py::_construct_l1_listener` (raises on activation state); `tests/test_listener.py::test_construct_rejects_inherited_listener` and `test_inherited_listener_fd_fails_closed_on_any_matching_activation_state`; validator rejects any `*.socket` unit and `Sockets=` wiring |
| public exposure | container port publication, public internet reachability | unit has no `[Install]`, `RefuseManualStart=yes`; `service/deployment/validate-systemd.sh` |
| multi-instance | a second service instance sharing the namespace | `listener.py` lifetime `flock(LOCK_EX|LOCK_NB)` on `service.lock` under the validated parent; single canonical path constant `_SOCKET_PATH` |

### 2. Threat table

Each threat names the vector, the control that blocks it, and the exact
artifact that enforces the control.

| threat | vector | control | enforcing artifact |
|---|---|---|---|
| Remote network compromise | a TCP/UDP listener reachable off-host | service binds only the AF_UNIX path `_SOCKET_PATH = "/run/poker-knight-ng/service.sock"`; systemd restricts to `AF_UNIX` | `listener.py::_SOCKET_PATH`; `poker-knight-ng.service` `RestrictAddressFamilies=AF_UNIX`; `validate-systemd.sh` exact-value check |
| Socket-activation fd injection | systemd passes a listener fd that the service adopts | fail closed before bind when activation state is present | `listener.py::_inherited_listener_fd`, `_construct_l1_listener`; `tests/test_listener.py` inherited-fd tests |
| Untrusted client already on the host reaching an unauthenticated socket | a non-privileged local process connects | socket mode `0660`, group `poker-knight-ng` membership is the authorization boundary | `listener.py::_SOCKET_MODE`, `_require_same_socket`; `poker-knight-ng.service` `User`/`Group` |
| Stale or symlink pathname tricking the listener into unlinking a foreign object | a hostile replacement at the socket pathname | descriptor-anchored parent FD, `O_NOFOLLOW`, stat re-inspection before every unlink; cleanup only removes the same device/inode socket this invocation created | `listener.py::_cleanup_target`, `_require_same_socket`, `_same_object`; `tests/test_listener.py` fault matrix |
| Second instance racing cleanup / stale proof | two invocations sharing the namespace | lifetime instance lock held through bind, serving, shutdown, cleanup | `listener.py` `flock(LOCK_EX|LOCK_NB)`; `service.lock` |
| HTTP request smuggling / framing confusion | `Transfer-Encoding`, duplicate `Content-Length`, pipelining, upgrades | strict framing before body admission; single request per connection | `framing.py::_inspect_request_head`, `framing.py::admit_request`; `tests/test_framing.py` (`test_transfer_encoding_is_empty_400_before_h11`, `test_duplicate_equal_content_length_is_empty_400`, `test_surplus_pipelined_bytes_are_empty_400`, `test_upgrade_attempt_is_empty_400`) |
| Header/body resource exhaustion | oversized headers, too many fields, oversized or incomplete bodies | bounded headers (8,192 bytes / 32 fields / 128-byte names / 1,024-byte values), `Content-Length` in `1..16384`, monotonic deadlines | `tests/test_framing.py`, `tests/test_connection.py` |
| Diagnostics information leak | a snapshot exposes peer identity, request data, card/seed material, or host paths | snapshot is a fixed RAM-only aggregate with no request-derived or path values | `runtime.py::ServiceRuntime.diagnostics_snapshot` (fixed key set); `tests/test_runtime_diagnostics.py::test_runtime_diagnostics_do_not_expose_identifiers_secrets_or_paths`; `tests/test_deployment_exposure.py::test_diagnostics_snapshot_exposes_no_peer_request_card_seed_or_path_fields` |
| Manual or unreviewed activation | an operator starts or enables the scaffold before conformance evidence exists | `RefuseManualStart=yes`, no `[Install]` section | `validate-systemd.sh`; `poker-knight-ng.service`; `tests/test_deployment_exposure.py::test_deployment_scaffolding_cannot_be_enabled_or_started_manually` |

### 3. Scope reviews

#### 3.1 Framing / HTTP layer

The HTTP surface is frozen to HTTP/1.1 over the AF_UNIX socket, exactly one
request per connection. TCP, UDP, HTTP/2, TLS, WebSockets, upgrades, CORS, and
public exposure are forbidden. Framing is admitted only through
`framing.py::admit_request`, which rejects non-identity `Content-Encoding`,
`Transfer-Encoding`, duplicate `Content-Length`, upgrades, and surplus
pipelined bytes before a body is admitted. The enforcing artifacts are
`tests/test_framing.py` and `tests/test_connection.py`.

#### 3.2 Listener construction

The production listener is constructed only through
`listener.py::construct_listener_with_callback` (and the no-callback
`construct_l1_listener`); there is no injectable path, and inherited listeners
fail closed. `listener.py::_inherited_listener_fd` returns a presence marker
when `LISTEN_PID` resolves to the current process regardless of the
`LISTEN_FDS` value (0, 1, 2, or malformed), and `_construct_l1_listener` raises
`ListenerConstructionError` before bind. Enforcing artifacts:
`tests/test_listener.py::test_construct_rejects_inherited_listener` and
`test_inherited_listener_fd_fails_closed_on_any_matching_activation_state`.

#### 3.3 systemd unit

`service/deployment/systemd/poker-knight-ng.service` is a reviewed scaffold
that must stay inactive. It sets `RestrictAddressFamilies=AF_UNIX`,
`RefuseManualStart=yes`, the non-privileged `poker-knight-ng` account/group,
`RuntimeDirectoryMode=0750`, and a process/filesystem hardening block
(`NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome=yes`, `PrivateTmp`,
`PrivateDevices`, system-call filtering). It contains no `[Install]` section
and no `Network`/`Socket`-related sections. `service/deployment/validate-systemd.sh`
enforces exact values and rejects any `*.socket` unit and any `Sockets=`
wiring. Enforcing artifacts: `tests/test_deployment_exposure.py`.

#### 3.4 Diagnostics surface

`runtime.py::ServiceRuntime.diagnostics_snapshot` returns a fixed, in-process
only aggregate with keys `schema_version`, `readiness`, `active_sessions`,
`max_sessions`, and `rejected_sessions`. It is deliberately not an HTTP route
(the frozen 204 `/healthz` is unchanged), and it exposes no peer, request,
card, seed, or path fields. Enforcing artifacts:
`tests/test_runtime_diagnostics.py::test_runtime_diagnostics_are_fixed_schema_and_not_ready_before_listener`
and `test_diagnostics_have_no_http_route`, plus
`tests/test_deployment_exposure.py::test_diagnostics_snapshot_exposes_no_peer_request_card_seed_or_path_fields`.

#### 3.5 Filesystem boundary

The runtime namespace is `/run/poker-knight-ng` (parent mode `0750`, owned by
`poker-knight-ng`), with `service.lock` (mode `0600`) and `service.sock` (mode
`0660`). The parent is opened descriptor-anchored with
`O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`; all manual checks, chmods, and unlinks are
relative to the retained parent FD. Enforcing artifacts: `listener.py`
constants and the `tests/test_listener.py` fault matrix (target replacement at
every check-to-operation boundary fails without mutation).

## Review verdict

Every control in this review is already implemented and is enforced by a named
test or validator artifact. No new security requirement is introduced. The
following artifact inventory binds the verdict:

- `service/deployment/validate-systemd.sh` enforces the exact systemd hardening
  values, the absence of `[Install]`, and the rejection of any `*.socket` unit
  and `Sockets=` wiring.
- `service/tests/test_deployment_exposure.py` enforces the forbidden-surface
  list, the validator's rejection of missing/`AF_INET`/`AF_INET6`
  `RestrictAddressFamilies`, the absence of net-listening directives and
  network-related sections, and the diagnostics snapshot's absence of
  peer/request/card/seed/path fields.
- `service/tests/test_listener.py` enforces inherited-listener rejection under
  all `LISTEN_FDS` variants and the filesystem fault matrix.
- `service/tests/test_framing.py` and `service/tests/test_connection.py`
  enforce the framing and resource-admission surface.
- `service/tests/test_runtime_diagnostics.py` enforces the fixed diagnostics
  schema and the absence of an HTTP diagnostics route.

## Consequences

The deployment remains a reviewed, inactive scaffold: `RefuseManualStart=yes`
and the missing `[Install]` section prevent manual or unreviewed activation.
Remote access, mTLS, bearer credentials, reverse-proxy trust, greater
concurrency, bounded cancellation, and public exposure each require separate
future decisions with their own evidence, exactly as ADR 0005 §Consequences
states. This review does not authorize starting, enabling, or shipping either
unit; `service/deployment/README.md` item 4 still requires an operator-approved
rollout and rollback plan, health check procedure, and privacy-safe evidence
record before that posture changes.
