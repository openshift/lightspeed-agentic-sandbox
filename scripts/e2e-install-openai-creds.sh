#!/usr/bin/env bash
# Sync OpenAI credentials into a cluster Secret for sandbox batch e2e Jobs (OLS-3926).
#
# Batch Jobs run on the cluster and mount llm-creds-openai via envFrom; they cannot
# read Konflux workspace secrets directly. This script copies a key from the test
# runner (local shell or Tekton step) onto the target cluster with oc apply.
#
# Credential source (pick one):
#   Local:  OPENAI_API_KEY=sk-... bash scripts/e2e-install-openai-creds.sh
#   Konflux: OPENAI_PROVIDER_KEY_PATH=/var/run/credentials/token
#            (Konflux mounts the workspace secret "openai" there; e2e-containers.sh
#            calls this script automatically before pytest)
#
# Optional env:
#   E2E_NAMESPACE   default: openshift-lightspeed
#   SECRET_NAME     default: llm-creds-openai

set -euo pipefail

E2E_NAMESPACE="${E2E_NAMESPACE:-openshift-lightspeed}"
SECRET_NAME="${SECRET_NAME:-llm-creds-openai}"

if ! command -v oc >/dev/null 2>&1; then
    echo "error: oc not found" >&2
    exit 1
fi

if [ -n "${OPENAI_PROVIDER_KEY_PATH:-}" ]; then
    if [ ! -f "${OPENAI_PROVIDER_KEY_PATH}" ]; then
        echo "error: OPENAI_PROVIDER_KEY_PATH file not found: ${OPENAI_PROVIDER_KEY_PATH}" >&2
        exit 1
    fi
    KEY=$(tr -d '[:space:]' < "${OPENAI_PROVIDER_KEY_PATH}")
elif [ -n "${OPENAI_API_KEY:-}" ]; then
    KEY="${OPENAI_API_KEY}"
else
    echo "error: set OPENAI_API_KEY or OPENAI_PROVIDER_KEY_PATH" >&2
    exit 1
fi

if [ -z "${KEY}" ]; then
    echo "error: OpenAI API key is empty" >&2
    exit 1
fi

tmp_key=$(mktemp)
chmod 600 "${tmp_key}"
trap 'rm -f "${tmp_key}"' EXIT
printf '%s' "${KEY}" > "${tmp_key}"

# Build oc create secret command with OPENAI_API_KEY and optional OPENAI_MODEL and OPENAI_BASE_URL
cmd_args=(
    "create" "secret" "generic" "${SECRET_NAME}"
    "--namespace=${E2E_NAMESPACE}"
    "--from-file=OPENAI_API_KEY=${tmp_key}"
)

# Add OPENAI_MODEL if set
if [ -n "${OPENAI_MODEL:-}" ]; then
    tmp_model=$(mktemp)
    chmod 600 "${tmp_model}"
    printf '%s' "${OPENAI_MODEL}" > "${tmp_model}"
    trap 'rm -f "${tmp_key}" "${tmp_model}"' EXIT
    cmd_args+=("--from-file=OPENAI_MODEL=${tmp_model}")
fi

# Add OPENAI_BASE_URL if set
if [ -n "${OPENAI_BASE_URL:-}" ]; then
    tmp_url=$(mktemp)
    chmod 600 "${tmp_url}"
    printf '%s' "${OPENAI_BASE_URL}" > "${tmp_url}"
    if [ -n "${OPENAI_MODEL:-}" ]; then
        trap 'rm -f "${tmp_key}" "${tmp_model}" "${tmp_url}"' EXIT
    else
        trap 'rm -f "${tmp_key}" "${tmp_url}"' EXIT
    fi
    cmd_args+=("--from-file=OPENAI_BASE_URL=${tmp_url}")
fi

oc "${cmd_args[@]}" --dry-run=client -o yaml | oc apply -f -

echo "Created/updated Secret ${SECRET_NAME} in ${E2E_NAMESPACE}"
echo
echo "Mount in batch Job via envFrom:"
echo "  envFrom:"
echo "    - secretRef:"
echo "        name: ${SECRET_NAME}"
echo
echo "Provider env for OpenAI:"
echo "  LIGHTSPEED_PROVIDER=openai"
echo "  LIGHTSPEED_MODEL=gpt-5-mini    # or your model"
echo "  OPENAI_MODEL=gpt-5-mini       # kept in sync by config mapping"
