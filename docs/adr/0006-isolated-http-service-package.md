# ADR 0006: Isolate the HTTP service runtime package

- Status: accepted
- Date: 2026-08-15
- Deciders: project owner and maintainer

## Context

ADR 0005 requires the selected HTTP runtime to live behind an optional service dependency boundary and to be pinned before maintained implementation. The accepted h11 spike selected direct `h11==0.16.0` and bound the exact wheel at SHA-256 `63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86`.

The root `pyproject.toml` and `uv.lock` are inputs to already-published CPU/CUDA qualification authorities. Adding a root optional dependency invalidates those historical manifests even though the base library import would remain inert. Weakening or rotating those authorities solely to package the private service would erase useful release provenance.

## Decision

The private HTTP service is a separate Python distribution rooted at `service/`:

- package name `poker-knight-ng-service`;
- independent `service/pyproject.toml` and `service/uv.lock`;
- Python `>=3.13,<3.14`;
- h11 supplied by the exact selected `files.pythonhosted.org` wheel URL;
- service lock binding the selected wheel SHA-256;
- independent tests under `service/tests`; and
- frozen synchronization and testing in the existing single CPU CI job.

The root engine distribution and root lock remain byte-unchanged. Importing `poker_knight_ng` continues to require no service dependencies and does not import h11.

The service distribution is optional at the deployment boundary rather than an extra in the root engine distribution. A later adapter checkpoint may add an exact `poker-knight-ng` distribution dependency to the service package without changing root metadata. Development may resolve that dependency from the repository root; release and deployment must install the exact qualified engine wheel.

This ADR supersedes only ADR 0005's packaging phrase "optional service dependency group." It does not change ADR 0005's AF_UNIX-only transport, framing limits, route semantics, resource admission, lifecycle, privacy, authorization, or no-fallback requirements.

## Rejected alternatives

### Add a root `service` extra

Rejected because it changes qualification-authority inputs and made the historical verifiers fail closed during the runtime-selection checkpoint.

### Vendor h11

Rejected because it would enlarge the maintained security surface, obscure upstream provenance, and complicate updates.

### Install an unpinned same-version h11 at runtime

Rejected because a version string alone does not bind executed bytes. The service lock and CI bind the selected wheel artifact.

### Weaken or rotate historical qualification manifests

Rejected because the private service can be isolated without altering already-qualified engine inputs.

## Consequences

The repository carries two explicit Python project boundaries and two locks. CI must target `service/tests` explicitly because selecting a uv project does not change pytest's discovery root.

Service code cannot leak into the root wheel. Packaging, release and deployment checks must build and verify the service distribution independently, then pair it with an exact qualified engine wheel.

No listener, socket activation, engine invocation or deployment is authorized by this packaging decision.
