#!/usr/bin/env bash
# Sync AWS Bedrock credentials into a cluster Secret for sandbox batch e2e Jobs.
#
# Konflux workspace secret bedrock-apitoken mounts keys aws_access_key_id and
# aws_secret_access_key under /var/run/credentials/. Batch Jobs mount the
# generated Secret at /var/run/secrets/llm-credentials, so expose the lowercase
# file keys resolved by the sandbox. Uppercase keys are also written for
# compatibility with older/manual envFrom jobs.
#
# Optional env:
#   E2E_NAMESPACE    default: openshift-lightspeed
#   SECRET_NAME      default: llm-creds-bedrock
#   CRED_DIR         default: /var/run/credentials
#   AWS_SESSION_TOKEN or CRED_DIR/aws_session_token
#   BEDROCK_ROLE_ARN, AWS_ROLE_ARN, or CRED_DIR/role_arn

set -euo pipefail

E2E_NAMESPACE="${E2E_NAMESPACE:-openshift-lightspeed}"
SECRET_NAME="${SECRET_NAME:-llm-creds-bedrock}"
CRED_DIR="${CRED_DIR:-/var/run/credentials}"

if ! command -v oc >/dev/null 2>&1; then
    echo "error: oc not found" >&2
    exit 1
fi

_read_key() {
    local env_val="$1"
    local file_name="$2"
    if [ -n "${env_val}" ]; then
        printf '%s' "${env_val}"
        return 0
    fi
    if [ -f "${CRED_DIR}/${file_name}" ]; then
        tr -d '[:space:]' < "${CRED_DIR}/${file_name}"
        return 0
    fi
    return 1
}

_read_optional_key() {
    local env_val="$1"
    local file_name="$2"
    _read_key "${env_val}" "${file_name}" || true
}

if ! AWS_ACCESS_KEY_ID="$(_read_key "${AWS_ACCESS_KEY_ID:-}" "aws_access_key_id")"; then
    echo "error: set AWS_ACCESS_KEY_ID or mount Konflux bedrock-apitoken" >&2
    exit 1
fi
if ! AWS_SECRET_ACCESS_KEY="$(_read_key "${AWS_SECRET_ACCESS_KEY:-}" "aws_secret_access_key")"; then
    echo "error: set AWS_SECRET_ACCESS_KEY or mount Konflux bedrock-apitoken" >&2
    exit 1
fi

if [ -z "${AWS_ACCESS_KEY_ID}" ] || [ -z "${AWS_SECRET_ACCESS_KEY}" ]; then
    echo "error: AWS credentials are empty" >&2
    exit 1
fi

AWS_SESSION_TOKEN="$(_read_optional_key "${AWS_SESSION_TOKEN:-}" "aws_session_token")"
BEDROCK_ROLE_ARN="$(_read_optional_key "${BEDROCK_ROLE_ARN:-${AWS_ROLE_ARN:-}}" "role_arn")"

tmp_access=$(mktemp)
tmp_secret=$(mktemp)
tmp_session=""
tmp_role=""
chmod 600 "${tmp_access}" "${tmp_secret}"
cleanup_files=("${tmp_access}" "${tmp_secret}")
trap 'rm -f "${cleanup_files[@]}"' EXIT
printf '%s' "${AWS_ACCESS_KEY_ID}" > "${tmp_access}"
printf '%s' "${AWS_SECRET_ACCESS_KEY}" > "${tmp_secret}"

secret_args=(
    --from-file=aws_access_key_id="${tmp_access}"
    --from-file=aws_secret_access_key="${tmp_secret}"
    --from-file=AWS_ACCESS_KEY_ID="${tmp_access}"
    --from-file=AWS_SECRET_ACCESS_KEY="${tmp_secret}"
)

if [ -n "${AWS_SESSION_TOKEN}" ]; then
    tmp_session=$(mktemp)
    chmod 600 "${tmp_session}"
    cleanup_files+=("${tmp_session}")
    printf '%s' "${AWS_SESSION_TOKEN}" > "${tmp_session}"
    secret_args+=(--from-file=aws_session_token="${tmp_session}")
    secret_args+=(--from-file=AWS_SESSION_TOKEN="${tmp_session}")
fi

if [ -n "${BEDROCK_ROLE_ARN}" ]; then
    tmp_role=$(mktemp)
    chmod 600 "${tmp_role}"
    cleanup_files+=("${tmp_role}")
    printf '%s' "${BEDROCK_ROLE_ARN}" > "${tmp_role}"
    secret_args+=(--from-file=role_arn="${tmp_role}")
fi

oc create secret generic "${SECRET_NAME}" \
    --namespace="${E2E_NAMESPACE}" \
    "${secret_args[@]}" \
    --dry-run=client -o yaml | oc apply -f -

echo "Created/updated Secret ${SECRET_NAME} in ${E2E_NAMESPACE}"
echo
echo "Mount in batch Job at /var/run/secrets/llm-credentials:"
echo "  volumes:"
echo "    - name: llm-credentials"
echo "      secret:"
echo "        secretName: ${SECRET_NAME}"
echo
if [ -n "${BEDROCK_ROLE_ARN}" ]; then
    echo "Configured optional role_arn for STS AssumeRole."
    echo
fi
echo "Provider env for Bedrock:"
echo "  LIGHTSPEED_PROVIDER=bedrock"
echo "  LIGHTSPEED_MODEL=global.anthropic.claude-haiku-5-5"
echo "  AWS_REGION=us-east-1                 # Bedrock region"
