"""Fixed production service-identity authority tests."""
from __future__ import annotations

from enum import IntEnum
from types import SimpleNamespace

import pytest


class _IntSubclass(int):
    pass


class _Number(IntEnum):
    ZERO = 0
    ONE = 1


def _records(*, uid=1201, primary_gid=1202, group_gid=1202):
    return (
        SimpleNamespace(pw_uid=uid, pw_gid=primary_gid),
        SimpleNamespace(gr_gid=group_gid),
    )


def _install(monkeypatch, passwd, group) -> None:
    from poker_knight_ng_service import identity

    monkeypatch.setattr(identity, "_getpwnam", lambda name: passwd)
    monkeypatch.setattr(identity, "_getgrnam", lambda name: group)


def test_resolver_uses_only_fixed_names_and_returns_opaque_token(monkeypatch) -> None:
    from poker_knight_ng_service import identity

    passwd, group = _records()
    names: list[tuple[str, str]] = []
    monkeypatch.setattr(
        identity,
        "_getpwnam",
        lambda name: names.append(("user", name)) or passwd,
    )
    monkeypatch.setattr(
        identity,
        "_getgrnam",
        lambda name: names.append(("group", name)) or group,
    )

    resolved = identity.resolve_production_identity()

    assert type(resolved) is identity.ResolvedServiceIdentity
    assert names == [
        ("user", "poker-knight-ng"),
        ("group", "poker-knight-ng"),
    ]
    assert not hasattr(resolved, "uid")
    assert not hasattr(resolved, "gid")
    with pytest.raises(TypeError):
        identity.ResolvedServiceIdentity()
    with pytest.raises(TypeError):
        identity.ResolvedServiceIdentity(1201, 1202)


@pytest.mark.parametrize("field", ["uid", "primary_gid", "group_gid"])
@pytest.mark.parametrize("invalid", [True, _IntSubclass(1), _Number.ONE, -1])
def test_resolver_rejects_nonexact_or_negative_numeric_authority(
    monkeypatch,
    field: str,
    invalid: object,
) -> None:
    from poker_knight_ng_service import identity

    values = {"uid": 1201, "primary_gid": 1202, "group_gid": 1202}
    values[field] = invalid
    passwd, group = _records(**values)
    _install(monkeypatch, passwd, group)

    with pytest.raises(identity.IdentityResolutionError):
        identity.resolve_production_identity()


@pytest.mark.parametrize("missing", ["user", "group"])
def test_resolver_fails_closed_when_fixed_name_is_missing(monkeypatch, missing: str) -> None:
    from poker_knight_ng_service import identity

    passwd, group = _records()

    def getpwnam(name):
        if missing == "user":
            raise KeyError(name)
        return passwd

    def getgrnam(name):
        if missing == "group":
            raise KeyError(name)
        return group

    monkeypatch.setattr(identity, "_getpwnam", getpwnam)
    monkeypatch.setattr(identity, "_getgrnam", getgrnam)

    with pytest.raises(identity.IdentityResolutionError):
        identity.resolve_production_identity()


def test_resolver_rejects_primary_group_mismatch(monkeypatch) -> None:
    from poker_knight_ng_service import identity

    passwd, group = _records(primary_gid=1202, group_gid=1203)
    _install(monkeypatch, passwd, group)

    with pytest.raises(identity.IdentityResolutionError):
        identity.resolve_production_identity()


def test_forged_opaque_token_has_no_identity_authority(monkeypatch) -> None:
    from poker_knight_ng_service import identity

    passwd, group = _records()
    _install(monkeypatch, passwd, group)
    issued = identity.resolve_production_identity()
    forged = object.__new__(identity.ResolvedServiceIdentity)

    assert identity._identity_values(issued) == (1201, 1202)
    with pytest.raises(identity.IdentityResolutionError):
        identity._identity_values(forged)
    with pytest.raises(identity.IdentityResolutionError):
        identity._identity_values(object())
