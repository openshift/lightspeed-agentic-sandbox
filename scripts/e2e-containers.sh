#!/usr/bin/env bash
# Run batch E2E BDD tests against an OpenShift cluster (OLS-3926).
#
# Installs cluster fixtures (unless E2E_SKIP_FIXTURES=1), syncs LLM credential
# Secrets, then runs pytest which creates batch Jobs per scenario.
#
# Usage (from lightspeed-agentic-sandbox/):
#   bash scripts/e2e-containers.sh                  # all three providers (sequential)
#   bash scripts/e2e-containers.sh openai-agents                    # all e2e tests
#   bash scripts/e2e-containers.sh openai-agents -k mcp             # pytest args (no --)
#   OPENAI_MODEL=gpt-4o bash scripts/e2e-containers.sh openai-agents
#
# Model resolution (priority order):
#   1. OPENAI_MODEL env var (for openai-agents)
#   2. GEMINI_MODEL env var (for gemini-vertex-adk)
#   3. ANTHROPIC_MODEL env var (for anthropic-vertex-deepagents)
#   4. ANTHROPIC_BEDROCK_MODEL env var (for anthropic-bedrock-deepagents)
#   5. Defaults from config.env (see _load_config_env_defaults)
#
# Custom LLM endpoint support (e.g., vLLM):
#   export OPENAI_API_KEY="token"
#   export OPENAI_MODEL="model-name"
#   export OPENAI_BASE_URL="https://vllm.example.com/v1"
#   bash scripts/e2e-containers.sh openai-agents -k mcp
#   See docs/e2e-custom-llm-endpoints.md for details.
#
# Prerequisites:
#   - oc + KUBECONFIG with permissions in E2E_NAMESPACE (default openshift-lightspeed)
#   - OPERATOR_REPO pointing at lightspeed-agentic-operator (for Result CRDs)
#   - Provider credentials on the host (see tests/e2e/credentials.py)
#
# Optional env:
#   E2E_SKIP_INSTALL=1       skip make install-all
#   E2E_SKIP_FIXTURES=1      skip scripts/e2e-install-fixtures.sh (fixtures already on cluster)
#   E2E_BATCH_VERIFY_FIXTURES  default 1 here — full SA/OTEL fixture check in pytest
#   SANDBOX_IMAGE            batch Job image (default: IMAGE or Konflux main tag)
#   E2E_NAMESPACE, SANDBOX_SA, LLM_SECRET, OPERATOR_REPO, E2E_ARGS, ARTIFACT_DIR

set -euo pipefail
trap 'echo "error: $0 line $LINENO: command \"$BASH_COMMAND\" exited with status $?" >&2' ERR

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${ROOT}"

UV="${UV:-uv}"
CONFIG_ENV="${ROOT}/tests/e2e/config.env"
E2E_NAMESPACE="${E2E_NAMESPACE:-openshift-lightspeed}"

_e2e_trim() {
    local s="${1:-}"
    s="${s//$'\r'/}"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "${s}"
}

IMAGE="$(_e2e_trim "${IMAGE:-}")"
if [[ -z "${IMAGE}" || -z "${IMAGE// }" ]]; then
    IMAGE="quay.io/redhat-user-workloads/crt-nshift-lightspeed-tenant/lightspeed-agentic-sandbox:main"
fi
export SANDBOX_IMAGE="$(_e2e_trim "${SANDBOX_IMAGE:-${IMAGE}}")"

if [ ! -f "${CONFIG_ENV}" ]; then
    echo "e2e: missing ${CONFIG_ENV}" >&2
    exit 1
fi

if ! command -v oc >/dev/null 2>&1; then
    echo "e2e: oc not found (batch e2e requires OpenShift cluster access)" >&2
    exit 1
fi

if ! oc whoami >/dev/null 2>&1; then
    echo "e2e: not logged in to a cluster (set KUBECONFIG or run oc login)" >&2
    exit 1
fi

export E2E_BATCH_VERIFY_FIXTURES="${E2E_BATCH_VERIFY_FIXTURES:-1}"
export E2E_NAMESPACE

PROVIDERS=(anthropic-vertex-deepagents anthropic-bedrock-deepagents gemini-vertex-adk openai-agents)

source_e2e_config() {
    local anthropic bedrock gemini openai
    read -r anthropic bedrock gemini openai < <(
        env -i bash -c "set -a; source \"${CONFIG_ENV}\"; set +a; printf '%s %s %s %s\n' \"\$ANTHROPIC_MODEL\" \"\$ANTHROPIC_BEDROCK_MODEL\" \"\$GEMINI_MODEL\" \"\$OPENAI_MODEL\""
    )
    export ANTHROPIC_MODEL="${anthropic}"
    export ANTHROPIC_BEDROCK_MODEL="${bedrock}"
    export GEMINI_MODEL="${gemini}"
    export OPENAI_MODEL="${openai}"
}

apply_model_override() {
    local provider="$1"
    local model="$2"
    case "${provider}" in
        anthropic-vertex-deepagents) export ANTHROPIC_MODEL="${model}" ;;
        anthropic-bedrock-deepagents) export ANTHROPIC_BEDROCK_MODEL="${model}" ;;
        gemini-vertex-adk) export GEMINI_MODEL="${model}" ;;
        openai-agents) export OPENAI_MODEL="${model}" ;;
        *)
            echo "e2e: unknown provider for model override: ${provider}" >&2
            exit 1
            ;;
    esac
}

sync_provider_model() {
    local provider="$1"
    unset LIGHTSPEED_MODEL
    case "${provider}" in
        anthropic-vertex-deepagents)
            export LIGHTSPEED_MODEL="${ANTHROPIC_MODEL:-}"
            export ANTHROPIC_MODEL="${LIGHTSPEED_MODEL}"
            ;;
        anthropic-bedrock-deepagents)
            export LIGHTSPEED_MODEL="${ANTHROPIC_BEDROCK_MODEL:-}"
            export ANTHROPIC_MODEL="${LIGHTSPEED_MODEL}"
            ;;
        gemini-vertex-adk)
            export LIGHTSPEED_MODEL="${GEMINI_MODEL:-}"
            export GEMINI_MODEL="${LIGHTSPEED_MODEL}"
            ;;
        openai-agents)
            export LIGHTSPEED_MODEL="${OPENAI_MODEL:-}"
            export OPENAI_MODEL="${LIGHTSPEED_MODEL}"
            ;;
        *)
            echo "e2e: unknown provider: ${provider}" >&2
            exit 1
            ;;
    esac
}

prepare_host_llm_creds() {
    local gcloud_adc="${HOME}/.config/gcloud/application_default_credentials.json"
    if [ ! -f "${gcloud_adc}" ]; then
        return 0
    fi
    export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-${gcloud_adc}}"
}

_install_e2e_fixtures() {
    if [[ "${E2E_SKIP_FIXTURES:-}" == "1" ]]; then
        echo "e2e: skipping fixture install (E2E_SKIP_FIXTURES=1)"
        return 0
    fi
    echo "e2e: installing cluster fixtures..."
    bash "${ROOT}/scripts/e2e-install-fixtures.sh"
}

_install_vertex_cluster_creds() {
    local secret_name="$1"
    local cred_file="${GOOGLE_APPLICATION_CREDENTIALS:-}"
    if [ -z "${cred_file}" ] || [ ! -f "${cred_file}" ]; then
        echo "e2e: GOOGLE_APPLICATION_CREDENTIALS must point at a service-account JSON file" >&2
        exit 1
    fi
    echo "e2e: syncing Secret ${secret_name} in ${E2E_NAMESPACE}..."
    oc create secret generic "${secret_name}" \
        --namespace="${E2E_NAMESPACE}" \
        --from-file=GOOGLE_APPLICATION_CREDENTIALS="${cred_file}" \
        --dry-run=client -o yaml | oc apply -f -
}

_install_provider_cluster_creds() {
    local provider="$1"
    case "${provider}" in
        openai-agents)
            if [ -n "${OPENAI_PROVIDER_KEY_PATH:-}" ]; then
                OPENAI_PROVIDER_KEY_PATH="${OPENAI_PROVIDER_KEY_PATH}" \
                    bash "${ROOT}/scripts/e2e-install-openai-creds.sh"
            elif [ -n "${OPENAI_API_KEY:-}" ]; then
                OPENAI_API_KEY="${OPENAI_API_KEY}" \
                    bash "${ROOT}/scripts/e2e-install-openai-creds.sh"
            else
                echo "e2e: OPENAI_API_KEY or OPENAI_PROVIDER_KEY_PATH required for openai-agents" >&2
                exit 1
            fi
            ;;
        gemini-vertex-adk)
            _install_vertex_cluster_creds "llm-creds-vertex"
            ;;
        anthropic-vertex-deepagents)
            _install_vertex_cluster_creds "llm-creds-anthropic"
            ;;
        anthropic-bedrock-deepagents)
            bash "${ROOT}/scripts/e2e-install-bedrock-creds.sh"
            ;;
        *)
            echo "e2e: unknown provider: ${provider}" >&2
            exit 1
            ;;
    esac
}

_configure_batch_job_env() {
    local provider="$1"
    local mcp_url="http://lightspeed-mock-mcp.${E2E_NAMESPACE}.svc:19090/mcp"
    export LIGHTSPEED_MCP_SERVERS="[{\"name\":\"mock-ocp-mcp\",\"url\":\"${mcp_url}\"}]"

    if [ -z "${LIGHTSPEED_REASONING_CONFIG:-}" ]; then
        case "${provider}" in
            openai-agents)
                export LIGHTSPEED_REASONING_CONFIG='{"effort":"low"}'
                ;;
            gemini-vertex-adk)
                export LIGHTSPEED_REASONING_CONFIG='{"thinking_budget":1024}'
                ;;
            anthropic-vertex-deepagents)
                export LIGHTSPEED_REASONING_CONFIG='{"thinking":{"type":"enabled","budget_tokens":1024}}'
                ;;
            anthropic-bedrock-deepagents)
                export LIGHTSPEED_REASONING_CONFIG='{"thinking":{"type":"enabled","budget_tokens":1024}}'
                ;;
        esac
    fi

    if [ -n "${OPENAI_BASE_URL:-}" ]; then
        export OPENAI_BASE_URL="${OPENAI_BASE_URL}"
    fi
}

_run_e2e_pytest() {
    local provider="$1"
    echo "e2e: running pytest for ${provider} (SANDBOX_IMAGE=${SANDBOX_IMAGE})..."
    local pytest_exit=0
    if [ -n "${ARTIFACT_DIR:-}" ]; then
        mkdir -p "${ARTIFACT_DIR}"
        local log_file="${ARTIFACT_DIR}/e2e-${provider}-pytest.log"
        # shellcheck disable=SC2086
        set +e
        "${UV}" run --extra e2e pytest -c tests/e2e/pytest.ini tests/e2e -v ${E2E_ARGS:-} 2>&1 | tee "${log_file}"
        pytest_exit=${PIPESTATUS[0]}
        set -e
        cat >"${ARTIFACT_DIR}/e2e-${provider}-summary.txt" <<EOF
provider: ${provider}
SANDBOX_IMAGE: ${SANDBOX_IMAGE}
LIGHTSPEED_MODEL: ${LIGHTSPEED_MODEL:-}
OPENAI_MODEL: ${OPENAI_MODEL:-}
ANTHROPIC_MODEL: ${ANTHROPIC_MODEL:-}
GEMINI_MODEL: ${GEMINI_MODEL:-}
pytest exit: ${pytest_exit}
pytest log: e2e-${provider}-pytest.log
EOF
    else
        # shellcheck disable=SC2086
        set +e
        "${UV}" run --extra e2e pytest -c tests/e2e/pytest.ini tests/e2e -v ${E2E_ARGS:-}
        pytest_exit=$?
        set -e
    fi
    return "${pytest_exit}"
}

run_one() {
    local provider="$1"
    local model_override="${2:-}"

    source_e2e_config

    if [ -n "${model_override}" ]; then
        apply_model_override "${provider}" "${model_override}"
    fi

    if [[ -z "${E2E_SKIP_INSTALL:-}" ]]; then
        make install-all
    fi

    prepare_host_llm_creds

    "${UV}" run --extra e2e python tests/e2e/credentials.py check "${provider}"

    _install_provider_cluster_creds "${provider}"

    sync_provider_model "${provider}"
    _configure_batch_job_env "${provider}"
    echo "e2e: model LIGHTSPEED_MODEL=${LIGHTSPEED_MODEL:-} OPENAI_MODEL=${OPENAI_MODEL:-} ANTHROPIC_MODEL=${ANTHROPIC_MODEL:-} GEMINI_MODEL=${GEMINI_MODEL:-}"
    echo "e2e: batch job env LIGHTSPEED_MCP_SERVERS=${LIGHTSPEED_MCP_SERVERS:-}"
    echo "e2e: batch job env LIGHTSPEED_REASONING_CONFIG=${LIGHTSPEED_REASONING_CONFIG:-}"
    echo "e2e: batch job env OPENAI_BASE_URL=${OPENAI_BASE_URL:-}"

    export E2E_PROVIDER="${provider}"
    export CLAUDE_CODE_USE_VERTEX="${CLAUDE_CODE_USE_VERTEX:-}"
    export ANTHROPIC_VERTEX_PROJECT_ID="${ANTHROPIC_VERTEX_PROJECT_ID:-}"
    export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-}"

    _run_e2e_pytest "${provider}"
}

_install_e2e_fixtures

if [ $# -eq 0 ]; then
    exit_code=0
    for p in "${PROVIDERS[@]}"; do
        if ! run_one "${p}" ""; then
            exit_code=1
        fi
    done
    exit "${exit_code}"
fi

provider="$1"
shift || true

# Resolve model from environment variables only
# Each provider has its own env var: OPENAI_MODEL, GEMINI_MODEL, ANTHROPIC_MODEL, ANTHROPIC_BEDROCK_MODEL
model_override=""
case "${provider}" in
    openai-agents)
        model_override="${OPENAI_MODEL:-}"
        ;;
    gemini-vertex-adk)
        model_override="${GEMINI_MODEL:-}"
        ;;
    anthropic-vertex-deepagents)
        model_override="${ANTHROPIC_MODEL:-}"
        ;;
    anthropic-bedrock-deepagents)
        model_override="${ANTHROPIC_BEDROCK_MODEL:-}"
        ;;
esac

# Parse remaining arguments as pytest args (optionally after --)
if [ "${1:-}" = "--" ]; then
    shift || true
    export E2E_ARGS="$*"
else
    export E2E_ARGS="$*"
fi

run_one "${provider}" "${model_override}"
