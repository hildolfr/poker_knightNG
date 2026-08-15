"""Phase 7B-L0 Unix-listener construction authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADR = ROOT / "docs/adr/0007-bounded-unix-listener-construction.md"
PROFILE = ROOT / "contracts/service/v1/unix-listener-construction.json"
MANIFEST = PROFILE.with_suffix(".sha256")


def test_listener_construction_authority_is_hash_bound_and_closed() -> None:
    raw = PROFILE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    profile = json.loads(raw)
    adr = ADR.read_text("utf-8")
    assert MANIFEST.read_text("ascii") == f"{digest}  {PROFILE.name}\n"
    assert set(profile) == {
        "authority", "checkpoint_order", "constructor", "filesystem",
        "format_version", "l1_required_tests", "l1_scope", "l3_shutdown",
        "stream_adapter",
    }
    assert profile["format_version"] == "poker-knight-ng-unix-listener-construction-v1"
    assert profile["checkpoint_order"] == [
        "secure-bind-and-stream-adapter",
        "bounded-session-manager",
        "graceful-lifecycle-drain",
    ]
    assert profile["constructor"] == {
        "accepted_socket_injection": "forbidden",
        "identity_resolution": "exact-pwd-and-grp-names-outside-bind",
        "numeric_type": "type-is-int-and-nonnegative",
        "path": "canonical-fixed-/run/poker-knight-ng/service.sock",
        "path_argument": "forbidden",
        "production_identity": "opaque-resolved-poker-knight-ng-token",
        "raw_numeric_identity": "forbidden",
        "start_serving_before_postconditions": "forbidden",
        "test_namespace": "closed-syscall-harness-no-production-override",
    }
    filesystem = profile["filesystem"]
    assert set(filesystem) == {
        "ancestor_policy", "cleanup", "instance_lock", "namespace_threat_model",
        "parent_authority", "parent_dirfd", "path_mutations", "post_bind",
        "replacement_policy", "socket_mode", "stale_probe", "symlink_policy",
    }
    assert filesystem["ancestor_policy"] == "administrator-controlled-nonsymlink-directories"
    assert filesystem["parent_authority"] == "dedicated-service-identity-mode-0750"
    assert filesystem["namespace_threat_model"] == "no-hostile-process-with-dedicated-service-uid"
    assert filesystem["parent_dirfd"] == "O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC-held-for-lifecycle"
    assert filesystem["path_mutations"] == "descriptor-relative-except-asyncio-bind-under-lock"
    assert filesystem["replacement_policy"] == "fail-without-mutation"
    assert filesystem["socket_mode"] == "0660"
    assert filesystem["symlink_policy"] == "reject-never-follow"
    assert filesystem["instance_lock"] == {
        "acquisition": "flock-LOCK_EX|LOCK_NB-before-socket-inspection",
        "basename": "service.lock",
        "held_until": "listener-cleanup-complete",
        "mode": "0600",
        "open": "O_CREAT|O_CLOEXEC|O_NOFOLLOW-via-parent-dirfd",
        "type_owner_mode": "regular-expected-uid-gid-0600",
    }
    assert filesystem["stale_probe"] == {
        "address_family": "AF_UNIX",
        "cleanup": "close-probe-socket-always",
        "deadline_ms": 250,
        "only_stale_result": "ECONNREFUSED",
        "socket_type": "SOCK_STREAM|SOCK_NONBLOCK",
        "unexpected_result": "fail-closed-without-unlink",
    }
    assert filesystem["cleanup"] == "held-lock-same-device-inode-socket-only"
    adapter = profile["stream_adapter"]
    assert set(adapter) == {"public_types", "reader", "writer"}
    assert adapter["public_types"] == ["asyncio.StreamReader", "asyncio.StreamWriter"]
    assert adapter["reader"] == {
        "overflow_buffer": "adapter-owned",
        "private_stream_reader_state": "forbidden",
        "read_buffered": "synchronous-owned-overflow-only",
        "underlying_read_max": "requested-limit-plus-one",
        "returned_read_max": "requested-limit",
        "surplus_read_ahead_bytes": 1,
        "zero_byte_eof": "preserve",
    }
    assert adapter["writer"] == {
        "base_exception": "propagate",
        "close": "writer.close-once-repeated-noop",
        "peer_loss_errno": ["EPIPE", "ECONNRESET", "ECONNABORTED", "ENOTCONN"],
        "peer_loss_scope": ["write", "drain", "wait_closed"],
        "peer_loss_result": False,
        "send_response": "writer.write-once-then-await-drain",
        "unlisted_ordinary_failure": "propagate",
        "wait_closed": "await-writer.wait_closed-once-repeated-noop",
    }
    assert profile["l1_required_tests"] == {
        "identity": [
            "bool", "int-subclass", "IntEnum", "unknown-name",
            "primary-group-mismatch", "raw-numeric-construction",
        ],
        "replacement_boundaries": [
            "check-to-stale-probe", "stale-probe-to-reinspect",
            "reinspect-to-unlink", "bind-to-chmod", "chmod-to-reinspect",
            "reinspect-to-start-serving", "cleanup-reinspect-to-unlink",
        ],
        "stale_probe": [
            "connect-success", "ECONNREFUSED", "timeout",
            "unexpected-errno", "mandatory-close",
        ],
        "stream": {
            "idempotence": ["close", "wait_closed"],
            "reader": ["read-ahead", "private-state-forbidden"],
            "writer_other_faults": {
                "base_exception": "each-operation",
                "operations": ["write", "drain", "close", "wait_closed"],
                "ordinary": "each-operation",
            },
            "writer_peer_loss": {
                "coverage": "cartesian-product",
                "errnos": ["EPIPE", "ECONNRESET", "ECONNABORTED", "ENOTCONN"],
                "operations": ["write", "drain", "wait_closed"],
            },
        },
    }
    assert profile["l1_scope"] == {
        "accept_behavior": "close-without-session-scheduling",
        "automatic_routing": "forbidden",
        "connection_cap": "deferred-l2",
        "deployment": "forbidden",
        "logging": "forbidden",
        "shutdown_manager": "deferred-l3",
    }
    assert profile["l3_shutdown"] == {
        "admitted_noncancellable_solve_after_deadline": "wait-without-deadline",
        "connection_drain_deadline_ms": 5000,
        "forced_solve_termination": "forbidden",
        "new_admission": "stop-before-drain",
    }
    for phrase in (
        "Status: accepted",
        "unix-listener-construction.json",
        "opaque resolved service identity",
        "canonical production socket path",
        "wait without a deadline",
        "type(value) is int",
        "lifetime instance lock",
        "exclusive namespace",
        "descriptor-anchored",
        "250 ms",
        "writer.write(response)",
        "await writer.drain()",
        "await writer.wait_closed()",
        "adapter-owned one-byte read-ahead",
        "No listener code is authorized by this decision alone",
    ):
        assert phrase in adr
