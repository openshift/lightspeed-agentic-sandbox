# lightspeed-agentic-sandbox

Multi-provider agentic sandbox library for OpenShift Lightspeed.

See [ARCHITECTURE.md](ARCHITECTURE.md) for architecture. Repository
workflows and dependency regeneration are documented in [AGENTS.md](AGENTS.md).

Local development uses `uv`. Run `make install` for dev dependencies,
`make install-all` for all providers plus e2e extras, and `make lock` to refresh
`uv.lock` after dependency changes.

## Bumping Dependencies

The container image is built hermetically in Konflux. After changing
dependencies in `pyproject.toml`, regenerate the Python and RPM lockfiles:

```bash
make bump-deps
make konflux-requirements
make rpm-lockfile
make verify
make test
```

`make bump-deps` updates `uv.lock`, the platform-specific requirements files,
and the top-level build requirements. `make konflux-requirements` separately
regenerates `.konflux/requirements.hashes.*.txt`,
`.konflux/requirements.hermetic.txt`, `.konflux/requirements-build.txt`, and
the Tekton prefetch package lists. `make rpm-lockfile` writes
`.konflux/rpms.lock.yaml` and requires Podman, `ACTIVATION_KEY`, `ORG_ID`, and
`.konflux/redhat.repo`.

See [AGENTS.md](AGENTS.md#konflux-hermetic-builds) for the complete workflow
and prerequisites.
