#!/usr/bin/env bash
# Run sandbox batch BDD tests, including the classifier corpus, for one provider.
#
# Called by the Konflux sandbox-integration-test-pipeline after
# cloning the repo at the SNAPSHOT revision.
#
# Usage:
#   bash .tekton/integration-tests/scripts/run-sandbox-integration-tests.sh <provider>
#
# Provider ids:
#   anthropic-vertex-deepagents | anthropic-bedrock-deepagents |
#   gemini-vertex-adk | openai-agents | anthropic-deepagents | azure-openai-agents
#
# Expects:
#   - Provider credentials mounted under /var/run/credentials/
#     (vertex/openai/anthropic-direct: token; bedrock: aws_access_key_id + aws_secret_access_key;
#      azure-openai secret: client_id + client_secret + tenant_id + endpoint + model + api_version)
#   - azure-openai-agents: AZURE_OPENAI_ENDPOINT or secret key 'endpoint' required;
#     secret keys 'model' and 'api_version' override AZURE_OPENAI_MODEL/LIGHTSPEED_PROVIDER_API_VERSION
#   - KUBECONFIG (Konflux pipeline mounts the hosted cluster's kubeconfig Secret)
#   - OPERATOR_REPO (Konflux pipeline clones operator for Result CRDs)
#   - E2E_NAMESPACE (default openshift-lightspeed)
#   - ARTIFACT_DIR set (for junit XML output)
#   - SNAPSHOT set (for SANDBOX_IMAGE from tested container build)
#   - Working directory is the sandbox repo root

set -euo pipefail
trap 'exit 130' INT
trap 'echo "error: $0 line $LINENO: command \"$BASH_COMMAND\" exited with status $?" >&2' ERR

PROVIDER="${1:?Usage: $0 <provider> (anthropic-vertex-deepagents|anthropic-bedrock-deepagents|gemini-vertex-adk|openai-agents|anthropic-deepagents|azure-openai-agents)}"
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
  anthropic-deepagents)
    if [ ! -f "${CRED_DIR}/apitoken" ]; then
        echo "error: anthropic-apitoken not found at ${CRED_DIR}/apitoken" >&2
        exit 1
    fi
    export ANTHROPIC_API_KEY
    ANTHROPIC_API_KEY="$(tr -d '[:space:]' < "${CRED_DIR}/apitoken")"
    ;;
  azure-openai-agents)
    if [ -f "${CRED_PATH}" ]; then
        # azure-openai secret (key: token) — API key mode
        export AZURE_OPENAI_API_KEY
        AZURE_OPENAI_API_KEY="$(tr -d '[:space:]' < "${CRED_PATH}")"
    elif [ -f "${CRED_DIR}/client_id" ] && [ -f "${CRED_DIR}/client_secret" ] && [ -f "${CRED_DIR}/tenant_id" ]; then
        # azure-openai secret (keys: client_id, client_secret, tenant_id) — Entra ID mode
        export AZURE_CLIENT_ID AZURE_CLIENT_SECRET AZURE_TENANT_ID
        AZURE_CLIENT_ID="$(tr -d '[:space:]' < "${CRED_DIR}/client_id")"
        AZURE_CLIENT_SECRET="$(tr -d '[:space:]' < "${CRED_DIR}/client_secret")"
        AZURE_TENANT_ID="$(tr -d '[:space:]' < "${CRED_DIR}/tenant_id")"
    else
        echo "error: no Azure credentials found under ${CRED_DIR}" >&2
        echo "  Mount azure-openai secret with key 'token' (API key) or keys 'client_id', 'client_secret', 'tenant_id' (Entra ID)" >&2
        exit 1
    fi
    # Read endpoint/model/api_version from secret files when present (azure-openai secret),
    # otherwise fall back to env vars or the hardcoded defaults for azureopenai-apitoken
    # and azureopenai-entra-id secrets which contain only auth credentials.
    if [ -z "${AZURE_OPENAI_ENDPOINT:-}" ] && [ -f "${CRED_DIR}/endpoint" ]; then
        export AZURE_OPENAI_ENDPOINT
        AZURE_OPENAI_ENDPOINT="$(tr -d '[:space:]' < "${CRED_DIR}/endpoint")"
    fi
    export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://ols-test.openai.azure.com/}"
    if [ -z "${AZURE_OPENAI_MODEL:-}" ] && [ -f "${CRED_DIR}/model" ]; then
        export AZURE_OPENAI_MODEL
        AZURE_OPENAI_MODEL="$(tr -d '[:space:]' < "${CRED_DIR}/model")"
    fi
    export AZURE_OPENAI_MODEL="${AZURE_OPENAI_MODEL:-gpt-6-luna}"
    if [ -z "${LIGHTSPEED_PROVIDER_API_VERSION:-}" ] && [ -f "${CRED_DIR}/api_version" ]; then
        export LIGHTSPEED_PROVIDER_API_VERSION
        LIGHTSPEED_PROVIDER_API_VERSION="$(tr -d '[:space:]' < "${CRED_DIR}/api_version")"
    fi
    export LIGHTSPEED_PROVIDER_API_VERSION="${LIGHTSPEED_PROVIDER_API_VERSION:-2025-04-01-preview}"
    export LIGHTSPEED_PROVIDER_URL="${AZURE_OPENAI_ENDPOINT}"
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
native_checks_exit=0
if [[ "${PROVIDER}" == "openai-agents" ]]; then
    echo "runner: running native OpenAI request-capture and agent integration tests"
    uv run pytest -q \
        tests/test_openai_models.py::test_openai_model_sends_responses_request_with_configured_settings \
        tests/test_openai_models.py::test_openai_model_omits_reasoning_from_request_when_unconfigured \
        tests/test_openai_models.py::test_native_openai_structured_output_sends_strict_schema_in_responses_request \
        tests/test_openai_models.py::test_native_openai_deepagent_reads_file_and_shapes_structured_result || native_checks_exit=$?
fi
case "${native_checks_exit}" in 130|143) exit "${native_checks_exit}" ;; esac
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
test_exit=0
bash scripts/e2e-containers.sh "${PROVIDER}" || test_exit=$?
echo "runner: native checks exit=${native_checks_exit}; BDD exit=${test_exit}"
if [ "${native_checks_exit}" -ne 0 ]; then
    exit "${native_checks_exit}"
fi
exit "${test_exit}"
