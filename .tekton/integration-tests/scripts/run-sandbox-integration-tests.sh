#!/usr/bin/env bash
# Run sandbox batch BDD integration tests for a single LLM provider.
#
# Called by the Konflux sandbox-integration-test-pipeline after
# cloning the repo at the SNAPSHOT revision.
#
# Usage:
#   bash .tekton/integration-tests/scripts/run-sandbox-integration-tests.sh <provider>
#
# Provider ids:
#   anthropic-vertex-deepagents | anthropic-bedrock-deepagents |
#   gemini-vertex-adk | openai-agents
#
# Expects:
#   - Provider credentials mounted under /var/run/credentials/
#     (vertex/openai: token; bedrock: aws_access_key_id, aws_secret_access_key)
#   - KUBECONFIG (Konflux pipeline mounts the hosted cluster's kubeconfig Secret)
#   - OPERATOR_REPO (Konflux pipeline clones operator for Result CRDs)
#   - E2E_NAMESPACE (default openshift-lightspeed)
#   - ARTIFACT_DIR set (for junit XML output)
#   - SNAPSHOT set (for SANDBOX_IMAGE from tested container build)
#   - Working directory is the sandbox repo root

set -euo pipefail
trap 'echo "error: $0 line $LINENO: command \"$BASH_COMMAND\" exited with status $?" >&2' ERR

PROVIDER="${1:?Usage: $0 <provider> (anthropic-vertex-deepagents|anthropic-bedrock-deepagents|gemini-vertex-adk|openai-agents)}"
CRED_DIR="/var/run/credentials"
CRED_PATH="${CRED_DIR}/token"

# --- Set up provider credentials on the test runner host ---
case "${PROVIDER}" in
  anthropic-vertex-deepagents)
    if [ ! -f "${CRED_PATH}" ]; then
        echo "error: credential file not found at ${CRED_PATH}" >&2
        exit 1
    fi
    export GOOGLE_PROVIDER_CREDENTIALS_PATH="${CRED_PATH}"
    mkdir -p "${HOME}/.config/gcloud"
    cp "${CRED_PATH}" "${HOME}/.config/gcloud/application_default_credentials.json"
    export GOOGLE_APPLICATION_CREDENTIALS="${CRED_PATH}"
    export CLAUDE_CODE_USE_VERTEX=1
    ANTHROPIC_VERTEX_PROJECT_ID=$(python3 -c "import json; print(json.load(open('${CRED_PATH}'))['project_id'])")
    export ANTHROPIC_VERTEX_PROJECT_ID
    ;;
  anthropic-bedrock-deepagents)
    if [ ! -f "${CRED_DIR}/aws_access_key_id" ] || [ ! -f "${CRED_DIR}/aws_secret_access_key" ]; then
        echo "error: bedrock credentials not found under ${CRED_DIR}" >&2
        exit 1
    fi
    export AWS_ACCESS_KEY_ID
    AWS_ACCESS_KEY_ID="$(tr -d '[:space:]' < "${CRED_DIR}/aws_access_key_id")"
    export AWS_SECRET_ACCESS_KEY
    AWS_SECRET_ACCESS_KEY="$(tr -d '[:space:]' < "${CRED_DIR}/aws_secret_access_key")"
    export CLAUDE_CODE_USE_BEDROCK=1
    export AWS_REGION="${AWS_REGION:-us-east-1}"
    ;;
  gemini-vertex-adk)
    if [ ! -f "${CRED_PATH}" ]; then
        echo "error: credential file not found at ${CRED_PATH}" >&2
        exit 1
    fi
    export GOOGLE_PROVIDER_CREDENTIALS_PATH="${CRED_PATH}"
    mkdir -p "${HOME}/.config/gcloud"
    cp "${CRED_PATH}" "${HOME}/.config/gcloud/application_default_credentials.json"
    export GOOGLE_APPLICATION_CREDENTIALS="${CRED_PATH}"
    GOOGLE_CLOUD_PROJECT=$(python3 -c "import json; print(json.load(open('${CRED_PATH}'))['project_id'])")
    export GOOGLE_CLOUD_PROJECT
    ;;
  openai-agents)
    if [ ! -f "${CRED_PATH}" ]; then
        echo "error: credential file not found at ${CRED_PATH}" >&2
        exit 1
    fi
    export OPENAI_PROVIDER_KEY_PATH="${CRED_PATH}"
    ;;
  *)
    echo "error: unknown provider: ${PROVIDER}" >&2
    exit 1
    ;;
esac

# --- Resolve sandbox image from SNAPSHOT (tested build) ---
if [ -n "${SNAPSHOT:-}" ]; then
    SANDBOX_IMAGE=$(echo "${SNAPSHOT}" | python3 -c "
import json, sys
snap = json.load(sys.stdin)
matches = [c for c in snap['components'] if c['name'] == 'lightspeed-agentic-sandbox']
if matches:
    print(matches[0]['containerImage'])
")
    if [ -z "${SANDBOX_IMAGE}" ]; then
        echo "error: SNAPSHOT has no containerImage for lightspeed-agentic-sandbox" >&2
        exit 1
    fi
    export SANDBOX_IMAGE
    echo "runner: SANDBOX_IMAGE=${SANDBOX_IMAGE}"
fi

# --- Install uv and project deps (no make/root; Konflux runs as arbitrary UID) ---
UV_VERSION="0.11.19"
_ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi
    local pip_cmd=""
    if command -v pip3 >/dev/null 2>&1; then
        pip_cmd=pip3
    elif command -v pip >/dev/null 2>&1; then
        pip_cmd=pip
    fi
    if [ -n "${pip_cmd}" ]; then
        local max_attempts=3 delay=5 attempt
        for attempt in $(seq 1 "${max_attempts}"); do
            if "${pip_cmd}" install --quiet "uv==${UV_VERSION}"; then
                return 0
            fi
            if [ "${attempt}" -eq "${max_attempts}" ]; then
                echo "error: failed to install uv==${UV_VERSION} after ${max_attempts} attempts" >&2
                exit 1
            fi
            echo "warning: pip install attempt ${attempt}/${max_attempts} failed, retrying in ${delay}s..." >&2
            sleep "${delay}"
            delay=$((delay * 2))
        done
    fi
    echo "runner: installing uv ${UV_VERSION} via standalone installer"
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    command -v uv >/dev/null 2>&1 || {
        echo "error: uv not on PATH after install" >&2
        exit 1
    }
}
_ensure_uv
uv sync --all-extras
export E2E_SKIP_INSTALL=1

ARTIFACT_DIR="${ARTIFACT_DIR:-/workspace/artifacts}"
mkdir -p "${ARTIFACT_DIR}"
export E2E_ARGS="--junitxml=${ARTIFACT_DIR}/junit_e2e.xml --tb=short"
export E2E_NAMESPACE="${E2E_NAMESPACE:-openshift-lightspeed}"
if [ -n "${OPERATOR_REPO:-}" ]; then
    export OPERATOR_REPO
    echo "runner: OPERATOR_REPO=${OPERATOR_REPO}"
fi
echo "runner: credentials configured for ${PROVIDER}, starting batch e2e (namespace=${E2E_NAMESPACE})"
bash scripts/e2e-containers.sh "${PROVIDER}"
