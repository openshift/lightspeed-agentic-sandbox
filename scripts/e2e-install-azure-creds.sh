#!/usr/bin/env bash
# Sync Azure OpenAI credentials into a cluster Secret for sandbox batch e2e Jobs.
#
# Supports two auth modes based on which files are available:
#
#   API key mode (Konflux secret azureopenai-apitoken, key: token):
#     Creates llm-creds-azure-openai with key AZURE_OPENAI_API_KEY.
#     config.py _resolve_azure reads it from {llm-credentials}/apitoken,
#     so the secret key is named apitoken (not AZURE_OPENAI_API_KEY).
#
#   Entra ID mode (Konflux secret azureopenai-entra-id, keys: client_id,
#     client_secret, tenant_id):
#     Creates llm-creds-azure-openai with keys client_id, client_secret,
#     tenant_id. config.py _resolve_azure detects Entra ID from these files.
#
# The batch Job must also have LIGHTSPEED_PROVIDER_URL set to the Azure OpenAI
# endpoint (e.g., https://my-resource.openai.azure.com/) via job_env in
# suite_setup.py, sourced from AZURE_OPENAI_ENDPOINT on the test runner.
#
# Optional env:
#   E2E_NAMESPACE   default: openshift-lightspeed
#   SECRET_NAME     default: llm-creds-azure-openai

set -euo pipefail

E2E_NAMESPACE="${E2E_NAMESPACE:-openshift-lightspeed}"
SECRET_NAME="${SECRET_NAME:-llm-creds-azure-openai}"
CRED_DIR="/var/run/credentials"

if ! command -v oc >/dev/null 2>&1; then
    echo "error: oc not found" >&2
    exit 1
fi

# Detect auth mode
if [ -n "${AZURE_OPENAI_API_KEY:-}" ]; then
    MODE="api_key_env"
elif [ -f "${CRED_DIR}/token" ]; then
    MODE="api_key_file"
elif [ -n "${AZURE_CLIENT_ID:-}" ] && [ -n "${AZURE_CLIENT_SECRET:-}" ] && [ -n "${AZURE_TENANT_ID:-}" ]; then
    MODE="entra_id_env"
elif [ -f "${CRED_DIR}/client_id" ] && [ -f "${CRED_DIR}/client_secret" ] && [ -f "${CRED_DIR}/tenant_id" ]; then
    MODE="entra_id_files"
else
    echo "error: no Azure credentials found." >&2
    echo "  API key:  set AZURE_OPENAI_API_KEY or mount azureopenai-apitoken (key: token)" >&2
    echo "  Entra ID: set AZURE_CLIENT_ID/AZURE_CLIENT_SECRET/AZURE_TENANT_ID" >&2
    echo "            or mount azureopenai-entra-id (keys: client_id, client_secret, tenant_id)" >&2
    exit 1
fi

case "${MODE}" in
api_key_env | api_key_file)
    if [ "${MODE}" = "api_key_env" ]; then
        KEY="${AZURE_OPENAI_API_KEY}"
    else
        KEY=$(tr -d '[:space:]' < "${CRED_DIR}/token")
    fi
    if [ -z "${KEY}" ]; then
        echo "error: Azure OpenAI API key is empty" >&2
        exit 1
    fi
    tmp_key=$(mktemp)
    chmod 600 "${tmp_key}"
    trap 'rm -f "${tmp_key}"' EXIT
    printf '%s' "${KEY}" > "${tmp_key}"
    oc create secret generic "${SECRET_NAME}" \
        --namespace="${E2E_NAMESPACE}" \
        --from-file=apitoken="${tmp_key}" \
        --dry-run=client -o yaml | oc apply -f -
    echo "Created/updated Secret ${SECRET_NAME} (API key mode) in ${E2E_NAMESPACE}"
    ;;
entra_id_env | entra_id_files)
    if [ "${MODE}" = "entra_id_env" ]; then
        CLIENT_ID="${AZURE_CLIENT_ID}"
        CLIENT_SECRET="${AZURE_CLIENT_SECRET}"
        TENANT_ID="${AZURE_TENANT_ID}"
    else
        CLIENT_ID=$(tr -d '[:space:]' < "${CRED_DIR}/client_id")
        CLIENT_SECRET=$(tr -d '[:space:]' < "${CRED_DIR}/client_secret")
        TENANT_ID=$(tr -d '[:space:]' < "${CRED_DIR}/tenant_id")
    fi
    if [ -z "${CLIENT_ID}" ] || [ -z "${CLIENT_SECRET}" ] || [ -z "${TENANT_ID}" ]; then
        echo "error: one or more Entra ID fields are empty" >&2
        exit 1
    fi
    tmp_cid=$(mktemp); tmp_cs=$(mktemp); tmp_tid=$(mktemp)
    chmod 600 "${tmp_cid}" "${tmp_cs}" "${tmp_tid}"
    trap 'rm -f "${tmp_cid}" "${tmp_cs}" "${tmp_tid}"' EXIT
    printf '%s' "${CLIENT_ID}" > "${tmp_cid}"
    printf '%s' "${CLIENT_SECRET}" > "${tmp_cs}"
    printf '%s' "${TENANT_ID}" > "${tmp_tid}"
    oc create secret generic "${SECRET_NAME}" \
        --namespace="${E2E_NAMESPACE}" \
        --from-file=client_id="${tmp_cid}" \
        --from-file=client_secret="${tmp_cs}" \
        --from-file=tenant_id="${tmp_tid}" \
        --dry-run=client -o yaml | oc apply -f -
    echo "Created/updated Secret ${SECRET_NAME} (Entra ID mode) in ${E2E_NAMESPACE}"
    ;;
esac

echo
echo "Mount in batch Job via volume (not envFrom — values are file paths):"
echo "  volumes:"
echo "    - name: llm-credentials"
echo "      secret:"
echo "        secretName: ${SECRET_NAME}"
echo
echo "Provider env for Azure OpenAI:"
echo "  LIGHTSPEED_PROVIDER=azure"
echo "  LIGHTSPEED_MODEL=gpt-4o           # or your deployed model name"
echo "  LIGHTSPEED_PROVIDER_URL=https://my-resource.openai.azure.com/"
