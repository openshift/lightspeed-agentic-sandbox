# lightspeed-agentic-sandbox

Multi-provider agentic sandbox library for OpenShift Lightspeed.

See [CLAUDE.md](CLAUDE.md) for architecture and usage.

Local development uses `uv`. Run `make install` for dev dependencies,
`make install-all` for all providers plus e2e extras, and `make lock` to refresh
`uv.lock` after dependency changes.

## Bumping Dependencies

The container image is built hermetially in Konflux. After changing
dependencies in `pyproject.toml`, regenerate the lockfiles:

```bash
make bump-deps          # upgrade uv.lock + regenerate requirements.{arch}.txt
make rpm-lockfile       # regenerate rpms.lock.yaml (needs podman)
```

See [CLAUDE.md](CLAUDE.md#konflux-hermetic-builds) for full details on the
hermetic build setup and how to add new dependencies.
