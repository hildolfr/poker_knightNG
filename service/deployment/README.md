# Service deployment scaffold

These files are **not an installation or activation procedure**. They provide a
reviewable systemd shape and an isolated validation check for Phase 7F while
runtime conformance remains the prerequisite for any operator activation.

## Safety posture

- The only declared endpoint is the ADR 0007 canonical AF_UNIX path:
  `/run/poker-knight-ng/service.sock`.
- The socket and service units both set `RefuseManualStart=yes` and deliberately
  contain no `[Install]` section. They cannot be enabled with `systemctl enable`
  and must not be started or copied into a live system as a rollout action.
- No TCP listener, port publication, proxy, TLS terminator, or public exposure
  is included or authorized.
- The service unit retains the non-privileged account, AF_UNIX-only address
  family, and filesystem/process hardening declarations. It intentionally does
  not alter service-package runtime behavior.

## Isolated validation

Run this from a checkout; it does not contact systemd, create `/run` paths,
copy files into `/etc`, or enable/start a unit:

```bash
sh service/deployment/validate-systemd.sh
```

When available, the script invokes `systemd-analyze verify` against the checkout
files. Its mandatory checks also assert the fixed path/modes, manual-start
refusal, absence of enablement targets, and absence of unvalidated `Sockets=`
wiring.

## Activation gate (not yet satisfied by this scaffold)

Do **not** remove `RefuseManualStart=yes`, add `[Install]`, start, enable, or
ship either unit until a reviewed runtime-conformance checkpoint has evidence
for all of the following:

1. L2's 16-connection rejection path and L3's admission-stop, five-second
   graceful drain, and indefinite admitted-solve wait have passed under the
   deployment process model.
2. The listener's canonical-path, UID/GID, `0750` parent, `0660` socket,
   descriptor-anchored cleanup, and stale-socket safety checks have passed with
   the dedicated `poker-knight-ng` identity.
3. Any socket-activation design is separately reconciled with ADR 0007's
   no-inherited-listener authority and is independently reviewed; this scaffold
   does not claim that authorization.
4. An operator-approved rollout and rollback plan, health check procedure, and
   privacy-safe evidence record are accepted. Observability and external
   exposure remain future work.

Record the passing command output, reviewed commit, and approval in the future
conformance evidence before changing this posture.
