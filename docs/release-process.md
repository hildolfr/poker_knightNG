# Promotion and release process

This procedure promotes the qualified revival history without invalidating its
historical Git-object, qualification, or evidence bindings. It does not itself
authorize a release. Every promotion, tag, GitHub release, or package-index
publication requires the gates below.

## Invariants

- Preserve the complete reviewed commit graph. **Never squash** and **never
  rebase** qualified revival commits: public verifiers read exact historical Git
  objects and ancestry.
- Promote only by a verified fast-forward when the target branch is an ancestor
  of the reviewed source.
- Never rewrite or move a published release tag.
- Never publish private evidence, raw timing vectors, host paths, process
  inventories, unfiltered GPU details, or access-controlled artifacts. Publish
  only the existing privacy-safe public projections and their manifests.
- A green local run does not replace remote CI, and remote CI does not replace
  the independent specification, provenance, quality, security, and GPU
  qualification gates required by the relevant checkpoint.
- PyPI publication is a separate approval and credential boundary from GitHub
  promotion or a GitHub release.

## 0. Hosted-automation prerequisites (must be verified live)

The workflow files deliberately contain no long-lived package-index credential:
`publish-pypi.yml` requests only an OIDC ID token and uses `uv publish --trusted-publishing`.
That design is safe only after the following **out-of-repository** controls are
configured and recorded in the release evidence:

1. Create the repository `pypi` environment before any publication dispatch.
   Require the designated owner reviewers and restrict its deployment branches
   to the intended immutable release tags.
2. Register PyPI trusted publishing for project `poker-knight-ng` to this exact
   repository and the `publish-pypi.yml` workflow. Confirm the registration with
   the project owner; no repository secret, `PYPI_TOKEN`, or fallback credential
   may be added.
3. Confirm the Actions token policy permits the explicitly requested
   `contents: write` permission for `release-prerelease.yml`; its final tag push
   and `gh release create` otherwise fail after all release gates have run.
4. Protect manual prerelease dispatch with the repository `release` environment.
   `release-prerelease.yml` is bound to that environment before any tag push or
   GitHub release creation. Configure `release` with required owner reviewers
   and deployment-branch restrictions that admit only `main`; verify the live
   configuration and preserve that evidence before treating release automation
   as ready. The repository file can request this protection but cannot prove
   its GitHub-side reviewers or restrictions are configured.
5. Exercise both manual workflows against a disposable prerelease only after the
   above controls are in place. Preserve the run URLs and verify that the PyPI
   job waited for environment approval before its OIDC exchange.

A failed post-tag `gh release create` can leave an annotated remote tag without
its GitHub release. The workflow intentionally refuses to reuse or move that
name on rerun. Treat this as an incident: stop, preserve the tag and run logs,
and obtain an owner-approved recovery plan rather than deleting or moving the
tag.

## 1. Source-candidate gate

Before proposing promotion:

1. Start from a clean, remotely published candidate SHA.
2. Confirm the candidate descends from the current target branch.
3. Confirm the complete suite, deterministic authorities, public qualification
   verifiers, public benchmark projection, package build, and isolated wheel
   smoke test pass at that exact SHA.
4. Confirm the repository-wide tracked-file secret scan has no unresolved
   findings.
5. Obtain independent specification/provenance and quality/security reviews for
   the exact candidate range.
6. Require the pinned `CPU verification` GitHub Actions run to pass for that SHA.

A stale result from another SHA is not evidence for the candidate.

## 2. Promotion to `main`

Promotion requires **explicit owner approval** after the candidate SHA, reviews,
and remote CI result are reported. Then:

```sh
git fetch origin --prune
git merge-base --is-ancestor origin/main <candidate-sha>
git switch main
git pull --ff-only origin main
git merge --ff-only <candidate-sha>
uv sync --frozen --group dev
uv run pytest -q
git push origin main
```

After the push, verify `refs/heads/main` remotely resolves to the exact local
SHA and verify the new `main` CI run. Do not claim promotion complete before both
checks pass. Configure branch protection or a repository ruleset to require the
CPU verification check before subsequent changes merge.

## 3. Release-candidate gate

The package version, tag, release title, and artifact names must agree. For the
first candidate, use a SemVer prerelease rather than asserting stable production
readiness. Before tagging:

- build wheel and sdist from a clean tag candidate;
- inspect both archives for expected package code, schemas, CUDA sources,
  metadata, README, and license;
- install the exact wheel into a new Python 3.13 environment and exercise root
  import, `poker-knight-ng --help`, CPU `solve`, and canonical serialization;
- verify the optional CUDA dependency remains opt-in and root import remains
  CuPy-inert;
- run the complete public authority verifier set;
- generate SHA-256 checksums for release assets; and
- independently review the release notes and asset list.

The private service distribution is **deployment-bundle only**: it must never be
released or installed as a standalone wheel because its `Requires-Dist` pins the
root `poker-knight-ng` engine package, which is not published to a package index.
A service deployment bundle must contain the matching-version engine wheel and
service wheel (plus the pinned `h11` wheel); CI installs that complete set and
runs `pip check`. Service sdists exclude `tests/` and are inspected by the
packaging regression test before release use.

Tagging and creating a GitHub release each require explicit owner approval. A
release must attach only newly built public artifacts and checksums from the
approved tag. Private qualification or benchmark evidence is never a release
asset.

## 4. PyPI boundary

PyPI is not implied by a Git tag or GitHub release. It requires separate approval,
a confirmed project-name/ownership check, narrowly scoped credentials or trusted
publishing, and a final artifact digest comparison against the approved release
assets. TestPyPI may be used first, but its upload is still an external
publication and requires approval.

## 5. Rollback and incident handling

Rollback is non-destructive and evidence preserving:

- do not force-push `main`;
- do not delete or move an issued tag to conceal a bad release;
- mark a GitHub release as withdrawn or prerelease when appropriate and publish
  a corrective release;
- yank a package-index release only when justified, while preserving provenance
  and publishing the replacement version;
- revert faulty source with a new reviewed commit when `main` must change; and
- record the affected SHAs, artifact hashes, failure mode, and corrective gates
  without publishing private runtime evidence.

A rollback is complete only after the corrected remote ref, CI state, public
artifacts, and operator-facing status are verified.
