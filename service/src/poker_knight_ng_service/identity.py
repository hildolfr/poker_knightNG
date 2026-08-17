"""Opaque fixed-name production service identity authority."""
from __future__ import annotations

from grp import getgrnam as _getgrnam
from pwd import getpwnam as _getpwnam
from weakref import WeakKeyDictionary

_SERVICE_NAME = "poker-knight-ng"


class IdentityResolutionError(RuntimeError):
    """The fixed production service identity could not be resolved safely."""


class ResolvedServiceIdentity:
    """Opaque resolver-issued authority for the dedicated service principal."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *args, **kwargs):
        del cls, args, kwargs
        raise TypeError("service identity tokens are resolver-owned")


_issued: WeakKeyDictionary[ResolvedServiceIdentity, tuple[int, int]] = WeakKeyDictionary()


def _issue_identity(uid: int, gid: int) -> ResolvedServiceIdentity:
    token = object.__new__(ResolvedServiceIdentity)
    _issued[token] = (uid, gid)
    return token


def _identity_values(identity: object) -> tuple[int, int]:
    if type(identity) is not ResolvedServiceIdentity:
        raise IdentityResolutionError("invalid service identity authority")
    try:
        return _issued[identity]
    except KeyError:
        raise IdentityResolutionError("invalid service identity authority") from None


def _valid_id(value: object) -> bool:
    return type(value) is int and value >= 0


def resolve_production_identity() -> ResolvedServiceIdentity:
    """Resolve exactly the dedicated production account and primary group."""

    try:
        passwd = _getpwnam(_SERVICE_NAME)
        group = _getgrnam(_SERVICE_NAME)
    except Exception:
        raise IdentityResolutionError("fixed service identity resolution failed") from None

    uid = passwd.pw_uid
    primary_gid = passwd.pw_gid
    group_gid = group.gr_gid
    if not all(_valid_id(value) for value in (uid, primary_gid, group_gid)):
        raise IdentityResolutionError("fixed service identity is invalid")
    if primary_gid != group_gid:
        raise IdentityResolutionError("fixed service primary group mismatch")
    return _issue_identity(uid, group_gid)
