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

python3 - "$service_unit" "$socket_unit" <<'PY2'
from collections import defaultdict
from pathlib import Path
import sys

def parse_unit(path: str):
    data = defaultdict(list)
    section = ""
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section and "=" in line:
            key, value = line.split("=", 1)
            data[section].append((key.strip(), value.strip()))
    return data

def pick(values, key):
    return [value for (k, value) in values if k == key]

service = parse_unit(sys.argv[1])
socket = parse_unit(sys.argv[2])

if "Install" in service:
    fail = True
    raise SystemExit("service unit must not define [Install]")
if "Install" in socket:
    raise SystemExit("socket unit must not define [Install]")

if pick(service["Unit"], "RefuseManualStart") != ["yes"]:
    raise SystemExit("service RefuseManualStart must be yes")
if pick(socket["Unit"], "RefuseManualStart") != ["yes"]:
    raise SystemExit("socket RefuseManualStart must be yes")

if pick(socket["Socket"], "ListenStream") != ["/run/poker-knight-ng/service.sock"]:
    raise SystemExit("socket must listen only on /run/poker-knight-ng/service.sock")
if any(k == "ListenDatagram" for k, _ in socket["Socket"]):
    raise SystemExit("socket must not define ListenDatagram")
if any(k == "ListenSequentialPacket" for k, _ in socket["Socket"]):
    raise SystemExit("socket must not define ListenSequentialPacket")
if pick(socket["Socket"], "SocketMode") != ["0660"]:
    raise SystemExit("socket SocketMode must be exactly 0660")
if pick(socket["Socket"], "DirectoryMode") != ["0750"]:
    raise SystemExit("socket DirectoryMode must be exactly 0750")
if pick(socket["Socket"], "SocketUser") != ["poker-knight-ng"]:
    raise SystemExit("socket SocketUser must be poker-knight-ng")
if pick(socket["Socket"], "SocketGroup") != ["poker-knight-ng"]:
    raise SystemExit("socket SocketGroup must be poker-knight-ng")
if pick(socket["Socket"], "Service") != ["poker-knight-ng.service"]:
    raise SystemExit("socket Service binding must be poker-knight-ng.service")

if pick(service["Service"], "User") != ["poker-knight-ng"]:
    raise SystemExit("service User must be poker-knight-ng")
if pick(service["Service"], "Group") != ["poker-knight-ng"]:
    raise SystemExit("service Group must be poker-knight-ng")
if pick(service["Service"], "RuntimeDirectory") != ["poker-knight-ng"]:
    raise SystemExit("service RuntimeDirectory must be poker-knight-ng")
if pick(service["Service"], "RuntimeDirectoryMode") != ["0750"]:
    raise SystemExit("service RuntimeDirectoryMode must be 0750")
if pick(service["Service"], "RestrictAddressFamilies") != ["AF_UNIX"]:
    raise SystemExit("service RestrictAddressFamilies must be AF_UNIX")
if pick(service["Service"], "Sockets"):
    raise SystemExit("service must not define Sockets")

for key in ("ListenStream", "SocketMode", "DirectoryMode", "SocketUser", "SocketGroup", "Service"):
    if len(pick(socket["Socket"], key)) != 1:
        raise SystemExit(f"socket directive '{key}' must appear exactly once")
for key in ("RefuseManualStart",):
    if len(pick(service["Unit"], key)) != 1:
        raise SystemExit(f"service Unit directive '{key}' must appear exactly once")
    if len(pick(socket["Unit"], key)) != 1:
        raise SystemExit(f"socket Unit directive '{key}' must appear exactly once")
PY2

if command -v systemd-analyze >/dev/null 2>&1; then
    # `verify` parses the checkout files only; it neither contacts a manager nor
    # installs, enables, starts, or creates either unit.
    systemd-analyze verify "$service_unit" "$socket_unit"
fi

printf '%s\n' 'deployment validation passed: scaffold remains inactive and unenableable'
