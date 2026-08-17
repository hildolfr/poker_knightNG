#!/bin/sh
# Validate deployment artifacts without installing or enabling units.
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
service_unit="$repo_root/service/deployment/systemd/poker-knight-ng.service"
socket_unit="$repo_root/service/deployment/systemd/poker-knight-ng.socket"

fail() {
    printf '%s\n' "deployment validation failed: $*" >&2
    exit 1
}

[ -f "$service_unit" ] || fail "missing service unit"
[ -f "$socket_unit" ] || fail "missing socket unit"

grep -Fqx 'RefuseManualStart=yes' "$service_unit" || fail "service must refuse manual activation"
grep -Fqx 'RefuseManualStart=yes' "$socket_unit" || fail "socket must refuse manual activation"
grep -Fqx 'ListenStream=/run/poker-knight-ng/service.sock' "$socket_unit" || fail "socket path is not canonical"
grep -Fqx 'SocketMode=0660' "$socket_unit" || fail "socket mode is not ADR 0005 exact mode"
grep -Fqx 'DirectoryMode=0750' "$socket_unit" || fail "runtime directory mode is not exact"
grep -Fqx 'Service=poker-knight-ng.service' "$socket_unit" || fail "socket target is not canonical"
! grep -Eq '^\[Install\]|^WantedBy=|^RequiredBy=' "$service_unit" || fail "service must not be enableable"
! grep -Eq '^\[Install\]|^WantedBy=|^RequiredBy=' "$socket_unit" || fail "socket must not be enableable"
! grep -Eq '^Sockets=' "$service_unit" || fail "service must not declare unvalidated socket wiring"

if command -v systemd-analyze >/dev/null 2>&1; then
    # `verify` parses the checkout files only; it neither contacts a manager nor
    # installs, enables, starts, or creates either unit.
    systemd-analyze verify "$service_unit" "$socket_unit"
fi

printf '%s\n' 'deployment validation passed: scaffold remains inactive and unenableable'
