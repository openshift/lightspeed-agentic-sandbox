#!/usr/bin/env bash
# Sync direct Anthropic API credentials into a cluster Secret for sandbox batch e2e Jobs.
#
# Konflux workspace secret anthropic-apitoken mounts key apitoken under
# /var/run/credentials/. Batch Jobs expect ANTHROPIC_API_KEY via envFrom.
#
# Credential source (pick one):
#   Local:    ANTHROPIC_API_KEY=sk-ant-... bash scripts/e2e-install-anthropic-direct-creds.sh
#   Konflux:  secret anthropic-apitoken with key apitoken mounted at /var/run/credentials/apitoken
#
# Optional env:
#   E2E_NAMESPACE   default: openshift-lightspeed
#   SECRET_NAME     default: llm-creds-anthropic-direct

set -euo pipefail

E2E_NAMESPACE="${E2E_NAMESPACE:-openshift-lightspeed}"
SECRET_NAME="${SECRET_NAME:-llm-creds-anthropic-direct}"
CRED_DIR="/var/run/credentials"

if ! command -v oc >/dev/null 2>&1; then
    echo "error: oc not found" >&2
    exit 1
fi

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    KEY="${ANTHROPIC_API_KEY}"
elif [ -f "${CRED_DIR}/apitoken" ]; then
    KEY=$(tr -d '[:space:]' < "${CRED_DIR}/apitoken")
else
    echo "error: set ANTHROPIC_API_KEY or mount anthropic-apitoken secret with key apitoken" >&2
    exit 1
fi

if [ -z "${KEY}" ]; then
    echo "error: Anthropic API key is empty" >&2
    exit 1
fi

tmp_key=$(mktemp)
chmod 600 "${tmp_key}"
trap 'rm -f "${tmp_key}"' EXIT
printf '%s' "${KEY}" > "${tmp_key}"

oc create secret generic "${SECRET_NAME}" \
    --namespace="${E2E_NAMESPACE}" \
    --from-file=ANTHROPIC_API_KEY="${tmp_key}" \
    --dry-run=client -o yaml | oc apply -f -

echo "Created/updated Secret ${SECRET_NAME} in ${E2E_NAMESPACE}"
echo
echo "Mount in batch Job via envFrom:"
echo "  envFrom:"
echo "    - secretRef:"
echo "        name: ${SECRET_NAME}"
echo
echo "Provider env for direct Anthropic:"
echo "  LIGHTSPEED_PROVIDER=anthropic"
echo "  LIGHTSPEED_MODEL=claude-haiku-4-5   # or your model"
