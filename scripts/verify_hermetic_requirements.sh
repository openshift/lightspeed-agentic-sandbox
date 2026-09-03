#!/bin/bash
# Verify that Konflux hermetic build hash files are in sync with uv.lock.
# Catches PRs that update pyproject.toml/uv.lock without updating hash files.
set -euo pipefail

SOURCE_HASH_FILE=".konflux/requirements.hashes.source.txt"
WHEEL_HASH_FILE=".konflux/requirements.hashes.wheel.txt"
WHEEL_PYPI_HASH_FILE=".konflux/requirements.hashes.wheel.pypi.txt"

# Packages in uv.lock runtime that are legitimately absent from hash files.
# Keep this list minimal — a growing allowlist is a red flag.
EXPECTED_MISSING=(
    # Windows-only — no linux wheels; omitted from Konflux prefetch hash files
    pywin32
)

log() { echo "==> $*"; }

# PEP 503 normalization: lowercase, collapse [-_.]+ to single underscore
normalize_names() {
    sed 's/ *==.*//' | tr '[:upper:]' '[:lower:]' | sed 's/[-_.]\{1,\}/_/g' | sort -u
}

# Extract normalized name\tversion pairs from pip-requirements-style input.
# Strips trailing continuations (\) and environment markers (; ...).
normalize_pinned() {
    grep -E '^[a-zA-Z0-9]' | sed 's/ *\\.*//; s/ *;.*//' | \
        awk -F'==' '{name=$1; ver=$2; gsub(/[-_.]+/, "_", name); gsub(/ /, "", ver); print tolower(name) "\t" ver}' | \
        sort -u
}

# --- Export runtime deps from uv.lock (with same extras as Konflux) ---
log "Exporting runtime dependencies from uv.lock"
UV_RAW=$(uv export --locked --no-dev --extra all --no-editable --no-header --no-annotate --format requirements.txt)
UV_PKGS=$(echo "$UV_RAW" | grep -E '^[a-zA-Z0-9]' | normalize_names)
UV_PINNED=$(echo "$UV_RAW" | normalize_pinned)
UV_COUNT=$(echo "$UV_PKGS" | wc -l | tr -d ' ')

# --- Validate hash files exist ---
for hash_file in "$SOURCE_HASH_FILE" "$WHEEL_HASH_FILE" "$WHEEL_PYPI_HASH_FILE"; do
    if [[ ! -r "$hash_file" ]]; then
        echo "ERROR: required hash file is missing or unreadable: $hash_file" >&2
        exit 1
    fi
done

# --- Extract packages from all three hash files ---
log "Parsing hash files"
HASH_RAW=$({ cat "$SOURCE_HASH_FILE"; cat "$WHEEL_HASH_FILE"; cat "$WHEEL_PYPI_HASH_FILE"; })
HASH_PKGS=$(echo "$HASH_RAW" | grep -E '^[a-zA-Z0-9]' | normalize_names)
HASH_PINNED=$(echo "$HASH_RAW" | normalize_pinned)
HASH_COUNT=$(echo "$HASH_PKGS" | grep -c . || true)

# Per-source pinned versions for tiered version checking:
# PyPI packages (source + wheel.pypi) must match uv.lock exactly.
# RHOAI packages (wheel) may intentionally pin different versions.
PYPI_PINNED=$({ cat "$SOURCE_HASH_FILE"; cat "$WHEEL_PYPI_HASH_FILE"; } | normalize_pinned)
RHOAI_PINNED=$(cat "$WHEEL_HASH_FILE" | normalize_pinned)

# --- Build allowlist set ---
if (( ${#EXPECTED_MISSING[@]} )); then
    ALLOW_SET=$(printf '%s\n' "${EXPECTED_MISSING[@]}" | normalize_names)
else
    ALLOW_SET=""
fi
ALLOW_COUNT=$(echo "$ALLOW_SET" | grep -c . || true)

# --- Compute missing = (uv_pkgs - hash_pkgs - allowlist) ---
MISSING=$(comm -23 <(echo "$UV_PKGS") <(echo "$HASH_PKGS") \
    | comm -23 - <(echo "$ALLOW_SET"))
MISSING_COUNT=$(echo "$MISSING" | grep -c . || true)

# --- Check for stale allowlist entries ---
STALE=$(
    {
        comm -23 <(echo "$ALLOW_SET") <(echo "$UV_PKGS")
        comm -12 <(echo "$ALLOW_SET") <(echo "$HASH_PKGS")
    } | sed '/^$/d' | sort -u
)
STALE_COUNT=$(echo "$STALE" | grep -c . || true)

# --- Check for orphans (in hash files but not in uv.lock) ---
# Build-only deps needed by source distributions (e.g. hatchling → pluggy)
# but not part of the runtime dependency tree.
EXPECTED_BUILD_ONLY=(pluggy)
BUILD_ALLOW=$(printf '%s\n' "${EXPECTED_BUILD_ONLY[@]}" | normalize_names)
ORPHANS=$(comm -23 <(echo "$HASH_PKGS") <(echo "$UV_PKGS") \
    | comm -23 - <(echo "$BUILD_ALLOW"))
ORPHAN_COUNT=$(echo "$ORPHANS" | grep -c . || true)

# --- Check for version mismatches (split by source) ---
# PyPI packages (source.txt + wheel.pypi.txt) must match uv.lock → ERROR
PYPI_MISMATCHED=""
while IFS=$'\t' read -r name uv_ver; do
    hash_versions=$(awk -F'\t' -v n="$name" '$1 == n {print $2}' <<< "$PYPI_PINNED" | paste -sd, -)
    if [[ -n "$hash_versions" ]] &&
        ! awk -F'\t' -v n="$name" -v v="$uv_ver" \
            '$1 == n && $2 == v { found=1 } END { exit !found }' <<< "$PYPI_PINNED"; then
        PYPI_MISMATCHED+="  $name: uv.lock=$uv_ver hash=$hash_versions"$'\n'
    fi
done <<< "$UV_PINNED"
PYPI_MISMATCHED_COUNT=$(echo "$PYPI_MISMATCHED" | grep -c . || true)

# RHOAI packages (wheel.txt) may intentionally differ → WARNING
RHOAI_MISMATCHED=""
while IFS=$'\t' read -r name uv_ver; do
    hash_versions=$(awk -F'\t' -v n="$name" '$1 == n {print $2}' <<< "$RHOAI_PINNED" | paste -sd, -)
    if [[ -n "$hash_versions" ]] &&
        ! awk -F'\t' -v n="$name" -v v="$uv_ver" \
            '$1 == n && $2 == v { found=1 } END { exit !found }' <<< "$RHOAI_PINNED"; then
        RHOAI_MISMATCHED+="  $name: uv.lock=$uv_ver hash=$hash_versions"$'\n'
    fi
done <<< "$UV_PINNED"
RHOAI_MISMATCHED_COUNT=$(echo "$RHOAI_MISMATCHED" | grep -c . || true)

# --- Report ---
log "$UV_COUNT runtime packages in uv.lock, $ALLOW_COUNT allowlisted, $HASH_COUNT in hash files"

EXIT_CODE=0

if [[ $MISSING_COUNT -gt 0 ]]; then
    echo ""
    echo "ERROR: $MISSING_COUNT package(s) in uv.lock but NOT in any hash file or allowlist:"
    echo "$MISSING" | sed 's/^/  - /'
    echo ""
    echo "Fix: run 'make konflux-requirements' or surgically add the missing packages."
    EXIT_CODE=1
fi

if [[ $PYPI_MISMATCHED_COUNT -gt 0 ]]; then
    echo ""
    echo "ERROR: $PYPI_MISMATCHED_COUNT PyPI package(s) have version mismatches between uv.lock and hash files:"
    echo "$PYPI_MISMATCHED"
    echo "Fix: run 'make konflux-requirements' to re-resolve PyPI packages."
    EXIT_CODE=1
fi

if [[ $RHOAI_MISMATCHED_COUNT -gt 0 ]]; then
    echo ""
    echo "WARNING: $RHOAI_MISMATCHED_COUNT RHOAI package(s) have version skew (expected — RHOAI pins curated versions):"
    echo "$RHOAI_MISMATCHED"
    echo "Investigate: update '.konflux/requirements.overrides.txt' if RHOAI is pinning an incompatible version."
fi

if [[ $STALE_COUNT -gt 0 ]]; then
    echo ""
    echo "ERROR: $STALE_COUNT allowlist entry/entries no longer in uv.lock (stale):"
    echo "$STALE" | sed 's/^/  - /'
    echo ""
    echo "Fix: remove stale entries from EXPECTED_MISSING in this script."
    EXIT_CODE=1
fi

if [[ $ORPHAN_COUNT -gt 0 ]]; then
    echo ""
    echo "ERROR: $ORPHAN_COUNT package(s) in hash files but NOT in uv.lock (orphans):"
    echo "$ORPHANS" | sed 's/^/  - /'
    echo ""
    echo "Fix: re-run 'make konflux-requirements' to drop stale entries."
    EXIT_CODE=1
fi

if [[ $EXIT_CODE -eq 0 ]]; then
    log "All runtime packages accounted for."
fi

exit $EXIT_CODE
