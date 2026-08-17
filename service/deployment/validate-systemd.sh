#!/bin/sh
# Validate deployment artifacts without installing or enabling units.
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
deployment_dir="$repo_root/service/deployment/systemd"
service_unit="$deployment_dir/poker-knight-ng.service"

fail() {
    printf '%s\n' "deployment validation failed: $*" >&2
    exit 1
}

[ -f "$service_unit" ] || fail "missing service unit"
# Socket activation is incompatible with ADR 0007: the runtime rejects inherited
# listener descriptors and must perform its own authenticated AF_UNIX bind.
if find "$deployment_dir" -maxdepth 1 -type f -name '*.socket' -print -quit | grep -q .; then
    fail "socket activation units are forbidden; the service binds its own listener"
fi

python3 - "$service_unit" <<'PY2'
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
if "Install" in service:
    raise SystemExit("service unit must not define [Install]")
if pick(service["Unit"], "RefuseManualStart") != ["yes"]:
    raise SystemExit("service RefuseManualStart must be exactly yes")
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
PY2

if command -v systemd-analyze >/dev/null 2>&1; then
    # `verify` parses the checkout file only; it neither contacts a manager nor
    # installs, enables, starts, or creates the unit.
    systemd-analyze verify "$service_unit"
fi

printf '%s\n' 'deployment validation passed: direct-bind scaffold remains inactive and unenableable'
