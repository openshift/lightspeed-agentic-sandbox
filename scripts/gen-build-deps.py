#!/usr/bin/env python3
"""
Generate requirements-build.txt for hermetic builds.

Scans the input requirements files for packages whose sdist hash is
present (meaning Cachi2 will prefetch the source tarball).  For each
such package, downloads the sdist from PyPI, reads build-system.requires
from pyproject.toml, and resolves the full build-dependency tree via
``uv pip compile`` to pinned ``name==version`` lines.

When all dependencies are fetched as source distributions (no binary
wheels), the build tools themselves must also be built from source.
This script recursively resolves build-system.requires of the build
dependencies until no new packages appear (fixed-point).

Usage:
    python scripts/gen-build-deps.py requirements-build.txt \\
        requirements.x86_64.txt requirements.aarch64.txt
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

PYPI_JSON = "https://pypi.org/pypi/{}/{}/json"
MAX_ROUNDS = 10

HEADER = (
    "#\n"
    "# Build dependencies for hermetic builds (auto-generated).\n"
    "# Needed when Cachi2 prefetches a source distribution instead of a wheel.\n"
    "# Regenerate: make requirements\n"
    "#\n"
)


def _norm(name: str) -> str:
    """PEP 503 name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_packages(*paths: str) -> dict[str, tuple[str, str]]:
    """Return {normalized_name: (raw_name, version)} from requirements files."""
    pkgs: dict[str, tuple[str, str]] = {}
    for p in paths:
        with open(p) as fh:
            for line in fh:
                m = re.match(r"^([A-Za-z0-9][\w.-]*)==([^\s\\;]+)", line.strip())
                if m:
                    pkgs.setdefault(_norm(m.group(1)), (m.group(1), m.group(2)))
    return pkgs


def parse_hashes(*paths: str) -> dict[str, set[str]]:
    """Return {normalized_name: {sha256_digest, …}} from requirements files."""
    hashes: dict[str, set[str]] = {}
    current: str | None = None
    for p in paths:
        with open(p) as fh:
            for line in fh:
                m = re.match(r"^([A-Za-z0-9][\w.-]*)==([^\s\\;]+)", line.strip())
                if m:
                    current = _norm(m.group(1))
                    hashes.setdefault(current, set())
                if current:
                    for h in re.findall(r"--hash=sha256:([0-9a-f]+)", line):
                        hashes[current].add(h)
    return hashes


def _pypi_urls(name: str, version: str) -> list[dict]:
    url = PYPI_JSON.format(name, version)
    if not url.startswith("https://"):
        return []
    req = urllib.request.Request(url, headers={"Accept": "application/json"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=15) as r:  # noqa: S310
        return json.loads(r.read()).get("urls", [])


def _has_sdist_hash(urls: list[dict], our_hashes: set[str]) -> bool:
    """True if any of our requirement hashes matches a source distribution."""
    for u in urls:
        fn = u["filename"]
        if fn.endswith((".tar.gz", ".zip")):
            digest = u.get("digests", {}).get("sha256", "")
            if digest in our_hashes:
                return True
    return False


def _read_build_requires_from_archive(urls: list[dict]) -> list[str]:
    """Download the first sdist and return its build-system.requires."""
    for u in urls:
        fn = u["filename"]
        dl_url = u["url"]
        if not dl_url.startswith("https://"):
            continue
        try:
            with urllib.request.urlopen(dl_url, timeout=60) as r:  # noqa: S310
                raw = r.read()
            if fn.endswith(".tar.gz"):
                with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
                    for m in tf.getmembers():
                        if re.match(r"^[^/]+/pyproject\.toml$", m.name):
                            fobj = tf.extractfile(m)
                            if fobj:
                                data = tomllib.loads(fobj.read().decode())
                                return data.get("build-system", {}).get("requires", ["setuptools"])
                    return ["setuptools"]
            elif fn.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    for n in zf.namelist():
                        if re.match(r"^[^/]+/pyproject\.toml$", n):
                            data = tomllib.loads(zf.read(n).decode())
                            return data.get("build-system", {}).get("requires", ["setuptools"])
                    return ["setuptools"]
        except Exception as exc:
            print(f"  warn: {fn}: {exc}", file=sys.stderr)
    return []


def sdist_build_requires(name: str, version: str) -> list[str]:
    """Fetch build-system.requires for a package version from PyPI."""
    urls = _pypi_urls(name, version)
    sdist_urls = [u for u in urls if u["filename"].endswith((".tar.gz", ".zip"))]
    if not sdist_urls:
        return []
    return _read_build_requires_from_archive(sdist_urls)


def _compile(specs: list[str]) -> str:
    """Run uv pip compile on the given specs and return the output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".in", delete=False) as tmp:
        for spec in sorted(specs):
            tmp.write(spec + "\n")
        tmp_path = tmp.name

    result = subprocess.run(
        ["uv", "pip", "compile", tmp_path, "--no-header", "--no-annotate"],
        capture_output=True,
        text=True,
    )
    Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    return result.stdout


def _parse_compiled(output: str) -> dict[str, str]:
    """Parse uv pip compile output into {normalized_name: version}."""
    pkgs: dict[str, str] = {}
    for line in output.splitlines():
        m = re.match(r"^([A-Za-z0-9][\w.-]*)==([^\s]+)", line.strip())
        if m:
            pkgs[_norm(m.group(1))] = m.group(2)
    return pkgs


def _extra_build_pins(
    exact_pins: dict[str, set[str]],
    resolved: dict[str, str],
) -> list[str]:
    """Return exact pins not covered by the unified build-dependency resolution."""
    extra_pins: list[str] = []
    for dep_norm, versions in sorted(exact_pins.items()):
        resolved_ver = resolved.get(dep_norm)
        for version in sorted(versions - {resolved_ver}):
            extra_pins.append(f"{dep_norm}=={version}")
    return extra_pins


def main() -> None:
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} OUTPUT REQ_FILE [REQ_FILE …]",
            file=sys.stderr,
        )
        sys.exit(1)

    output, *req_files = sys.argv[1:]
    runtime_pkgs = parse_packages(*req_files)
    all_hashes = parse_hashes(*req_files)

    print(
        f"Scanning {len(runtime_pkgs)} packages for prefetched sdists …",
        file=sys.stderr,
    )

    # -- Phase 1: collect build-system.requires from runtime packages ----------
    build_specs: list[str] = []
    # Track exact pins (e.g. hatchling==1.26.3) that must be prefetched even if
    # uv resolves the package to a different version.
    exact_pins: dict[str, set[str]] = {}  # {normalized_name: {version, …}}
    for norm_name, (name, ver) in sorted(runtime_pkgs.items()):
        our_hashes = all_hashes.get(norm_name, set())
        if not our_hashes:
            continue
        try:
            urls = _pypi_urls(name, ver)
        except Exception as exc:
            print(f"  warn: {name}=={ver}: {exc}", file=sys.stderr)
            continue
        if not _has_sdist_hash(urls, our_hashes):
            continue
        reqs = _read_build_requires_from_archive(
            [u for u in urls if u["filename"].endswith((".tar.gz", ".zip"))]
        )
        if reqs:
            print(f"  {name}=={ver} → {reqs}", file=sys.stderr)
            build_specs.extend(reqs)
            for req in reqs:
                pin = re.match(r"^([A-Za-z0-9][\w.-]*)==([^\s,;]+)$", req.strip())
                if pin:
                    exact_pins.setdefault(_norm(pin.group(1)), set()).add(pin.group(2))

    # Deduplicate by normalized name, keeping the first spec seen
    seen_specs: set[str] = set()
    unique: list[str] = []
    for spec in build_specs:
        m = re.match(r"([A-Za-z0-9][\w.-]*)", spec)
        if not m:
            continue
        n = _norm(m.group(1))
        if n not in seen_specs:
            seen_specs.add(n)
            unique.append(spec)

    if not unique:
        Path(output).write_text(HEADER)
        print("No build dependencies needed.", file=sys.stderr)
        return

    # -- Phase 2: compile and recursively resolve build deps of build deps -----
    compiled_output = _compile(unique)
    resolved = _parse_compiled(compiled_output)
    all_known = set(runtime_pkgs.keys()) | set(resolved.keys())
    # Track which resolved packages we already inspected for build deps.
    # Separate from seen_specs (which deduplicates requirement specs).
    checked: set[str] = set()

    for round_num in range(1, MAX_ROUNDS + 1):
        new_specs: list[str] = []
        for pkg_norm, pkg_ver in sorted(resolved.items()):
            if pkg_norm in checked:
                continue
            checked.add(pkg_norm)
            try:
                reqs = sdist_build_requires(pkg_norm, pkg_ver)
            except Exception as exc:
                print(f"  warn: {pkg_norm}=={pkg_ver}: {exc}", file=sys.stderr)
                continue
            for req in reqs:
                m = re.match(r"([A-Za-z0-9][\w.-]*)", req)
                if not m:
                    continue
                dep_norm = _norm(m.group(1))
                if dep_norm not in all_known:
                    new_specs.append(req)
                    all_known.add(dep_norm)

        if not new_specs:
            break

        print(
            f"  round {round_num}: found {len(new_specs)} new build dep(s): "
            f"{[s.split()[0] for s in new_specs]}",
            file=sys.stderr,
        )
        unique.extend(new_specs)
        compiled_output = _compile(unique)
        resolved = _parse_compiled(compiled_output)
        all_known = set(runtime_pkgs.keys()) | set(resolved.keys())

    # -- Phase 3: preserve exact pins required by individual sdists -------------
    # Different sdists can intentionally require different exact versions of the
    # same build backend.  Hermeto fetches every line and pip resolves each
    # PEP 517 build environment independently, so duplicate package names with
    # different versions are valid and required here.
    extra_pins = _extra_build_pins(exact_pins, resolved)
    for pin in extra_pins:
        dep_norm = pin.split("==", 1)[0]
        print(
            f"  extra pin: {pin} (resolved {resolved.get(dep_norm)}, required by an sdist)",
            file=sys.stderr,
        )

    final_output = compiled_output
    if extra_pins:
        if final_output and not final_output.endswith("\n"):
            final_output += "\n"
        final_output += "\n".join(extra_pins) + "\n"

    Path(output).write_text(HEADER + final_output)
    pkg_count = len(_parse_compiled(final_output))
    print(f"Wrote {output} ({pkg_count} packages)", file=sys.stderr)


if __name__ == "__main__":
    main()
