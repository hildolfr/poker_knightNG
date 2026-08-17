"""Phase 7A private HTTP service contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
PROFILE = ROOT / "contracts" / "service" / "v1" / "http-service-profile.json"
MANIFEST = ROOT / "contracts" / "service" / "v1" / "http-service-profile.sha256"
ADR = ROOT / "docs" / "adr" / "0008-automatic-cuda-routing.md"
ROADMAP = ROOT / "docs" / "roadmap-status.md"
SPEC = ROOT / "validation" / "holdem" / "v1" / "SPEC.md"


def load_profile() -> dict[str, Any]:
    raw = PROFILE.read_bytes()
    value = json.loads(raw)
    assert raw == (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert MANIFEST.read_text("ascii") == f"{hashlib.sha256(raw).hexdigest()}  {PROFILE.name}\n"
    return value


def test_profile_is_closed_and_binds_frozen_v1_documents() -> None:
    profile = load_profile()
    assert set(profile) == {
        "authority", "execution", "forbidden", "format_version", "limits", "logging",
        "request", "responses", "routes", "socket", "transport",
    }
    assert profile["format_version"] == "poker-knight-ng-private-http-v1"
    assert profile["authority"] == {
        "adr": "docs/adr/0008-automatic-cuda-routing.md",
        "problem_schema": {
            "path": "contracts/v1/problem.schema.json",
            "sha256": "3d85b79f6089d6bbbef21891743270d8fbc87149df9ad84be07e119a5df43c18",
        },
        "request_schema": {
            "path": "contracts/v1/equity-request.schema.json",
            "sha256": "ca6c67a3cc65ecdcaa1dff50db929f9e76fdaf248ec9b4e94b469f823167491f",
        },
        "result_schema": {
            "path": "contracts/v1/equity-result.schema.json",
            "sha256": "2a7cbc159add6ab1d0b02f5b5534ca301392c372681a88886c588b87f115a1e3",
        },
    }


def test_transport_and_socket_are_private_local_only() -> None:
    profile = load_profile()
    assert profile["transport"] == {
        "connection": "close",
        "family": "AF_UNIX",
        "http_version": "HTTP/1.1",
        "requests_per_connection": 1,
    }
    assert profile["socket"] == {
        "default_path": "/run/poker-knight-ng/service.sock",
        "directory_mode": "0750",
        "service_group": "poker-knight-ng",
        "service_user": "poker-knight-ng",
        "socket_mode": "0660",
        "symlinks": "forbidden",
    }
    forbidden = set(profile["forbidden"])
    assert {"tcp", "reverse-proxy", "public-internet", "http-authentication", "cors", "websocket", "http2"} <= forbidden


def test_routes_are_explicit_and_auto_route_supported() -> None:
    profile = load_profile()
    assert profile["routes"] == [
        {
            "authorization": "socket-filesystem",
            "backend_probe": "forbidden",
            "body": "empty",
            "method": "GET",
            "path": "/healthz",
            "success_status": 204,
        },
        {
            "authorization": "socket-filesystem",
            "executor": "solve",
            "method": "POST",
            "path": "/v1/solve",
            "request_backend": "auto",
            "success_status": 200,
        },
        {
            "authorization": "socket-filesystem",
            "executor": "solve_cuda",
            "method": "POST",
            "path": "/v1/solve-cuda",
            "request_backend": "cuda",
            "success_status": 200,
        },
    ]
    assert profile["execution"] == {
        "client_disconnect": "continue-accepted-solve",
        "fallback": "forbidden",
        "partial_success": "forbidden",
        "queue": "forbidden",
        "retry": "forbidden",
        "shutdown": "stop-admission-and-drain",
        "trial_reduction": "forbidden",
    }


def test_request_and_capacity_bounds_are_exact() -> None:
    profile = load_profile()
    assert profile["limits"] == {
        "body_read_timeout_ms": 5000,
        "header_count": 32,
        "header_name_bytes": 128,
        "header_read_timeout_ms": 5000,
        "header_total_bytes": 8192,
        "header_value_bytes": 1024,
        "max_body_bytes": 16384,
        "max_connections": 16,
        "max_queue": 0,
        "max_requested_trials": 1000000,
        "max_solve_in_flight": 1,
    }
    assert profile["request"] == {
        "content_encoding": "identity-only",
        "content_length": "required-single-canonical-decimal",
        "content_types": ["application/json", "application/json; charset=utf-8"],
        "json": {
            "bom": "forbidden",
            "duplicate_members": "forbidden-at-every-depth",
            "non_finite": "forbidden",
            "root": "object",
            "trailing_document": "forbidden",
            "utf8": "required",
        },
        "transfer_encoding": "forbidden",
    }
    assert profile["logging"] == {
        "allowed_fields": ["correlation_id", "route", "status"],
        "forbidden_fields": [
            "backend_diagnostics", "board_cards", "canonical_case_hash", "exception_text",
            "hero_cards", "host_path", "request_body", "seed", "socket_peer",
        ],
    }


def test_response_classes_do_not_extend_v1_problem_schema() -> None:
    profile = load_profile()
    responses = profile["responses"]
    assert responses["success"] == "canonical-v1-result-json-line"
    assert responses["problem"] == "canonical-v1-problem-json-line"
    assert responses["problem_mapping"] == {
        "cuda-route-cpu-backend": "UNSUPPORTED_REQUEST",
        "engine-or-adapter-failure": "INTERNAL_ERROR",
        "service-trial-cap": "UNSUPPORTED_REQUEST",
        "solve-capacity-busy": "RESOURCE_EXHAUSTED",
        "solve-route-invalid-backend": "UNSUPPORTED_REQUEST",
    }
    assert responses["transport_failures"] == {
        "body-read-timeout": 408,
        "header-limit": 431,
        "header-or-framing": 400,
        "method": 405,
        "path": 404,
        "payload-too-large": 413,
        "unsupported-media": 415,
    }
    assert responses["transport_failure_body"] == "empty"
    assert responses["connection"] == "close"
    assert responses["cache_control"] == "no-store"
    assert responses["content_type"] == "application/json"
    assert responses["content_type_options"] == "nosniff"
    assert responses["request_id_header"] == "X-Poker-Knight-Request-ID"
    assert responses["request_id_format"] == "pk_ plus 32 lowercase hexadecimal digits"


def test_adr_freezes_security_and_deferred_implementation_boundary() -> None:
    text = ADR.read_text("utf-8")
    for required in (
        "Status: accepted", "AF_UNIX", "1,000,000", "one in-flight solve",
        "no execution timeout", "client disconnect", "drain", "TCP, UDP, HTTP/2",
        "reverse proxy exposure", "public internet exposure", "RESOURCE_EXHAUSTED",
        "UNSUPPORTED_REQUEST", "Phase 7B", "runtime selection",
    ):
        assert required in text
    assert "automatic CUDA routing" in text
    assert "does not modify" in text and "v1 request" in text


def test_spec_and_roadmap_reflect_phase7c_status() -> None:
    spec = SPEC.read_text("utf-8")
    roadmap = ROADMAP.read_text("utf-8")
    assert "automatic CUDA routing" in spec
    assert "L1 secure listener construction/adapters are implemented" in roadmap
    assert "ADR 0005" in roadmap
    phase7a = next(line for line in roadmap.splitlines() if line.startswith("| Phase 7A"))
    phase7b = next(line for line in roadmap.splitlines() if line.startswith("| Phase 7B"))
    phase7c = next(line for line in roadmap.splitlines() if line.startswith("| Phase 7C"))
    assert "**Complete**" in phase7a
    assert "`cd994d45516b257515c03f6225b4242babde0cc5`" in phase7a
    assert "https://github.com/hildolfr/poker_knightNG/actions/runs/31862456329" in phase7a
    assert "ADR 0005" in phase7a
    assert "**Active**" in phase7b
    assert "listener" in phase7b.lower()
    assert "**Active**" in phase7c
    assert "automatic" in phase7c.lower()
