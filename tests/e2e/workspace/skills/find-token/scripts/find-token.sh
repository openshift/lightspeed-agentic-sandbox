#!/bin/bash
# Generates analysis output aligned with operator AnalysisResult + e2e components schema.
set -euo pipefail

OUTDIR="${E2E_OUTPUT_DIR:-/tmp/lightspeed-e2e-output}"
mkdir -p "${OUTDIR}"

DIAG_TOKEN=$(head -c 12 /dev/urandom | base64 | tr -d '=/+')
VERIFY_TOKEN=$(head -c 12 /dev/urandom | base64 | tr -d '=/+')
TIMESTAMP=$(date +%s)

cat > "${OUTDIR}/.hidden_token" <<EOF
DIAG_${DIAG_TOKEN}
VERIFY_${VERIFY_TOKEN}
EOF

cat <<EOF
{
  "actionRequired": "True",
  "options": [
    {
      "title": "Token retrieval analysis",
      "diagnosis": {
        "summary": "Successfully retrieved verification tokens",
        "rootCause": "Verification requested by e2e harness"
      },
      "remediationPlan": {
        "description": "Return the generated tokens for verification",
        "actions": [
          {
            "command": "bash scripts/find-token.sh",
            "type": "verify",
            "description": "Generate cryptographic tokens"
          },
          {
            "command": "cat ${OUTDIR}/.hidden_token",
            "type": "report",
            "description": "Return tokens in structured format"
          }
        ],
        "reversible": "Reversible"
      },
      "verification": {
        "description": "Token integrity check"
      },
      "components": [
        {
          "type": "token_verification",
          "source": {
            "generator": "find-token.sh",
            "timestamp": "${TIMESTAMP}"
          },
          "tokens": {
            "primary": {
              "value": "DIAG_${DIAG_TOKEN}",
              "valid": true
            },
            "secondary": {
              "value": "VERIFY_${VERIFY_TOKEN}",
              "valid": true
            }
          },
          "audit": {
            "outcome": "pass",
            "checks_performed": ["generation", "file_write", "integrity"],
            "findings": [
              {"check": "generation", "result": "pass", "severity": "info"},
              {"check": "file_write", "result": "pass", "severity": "info"},
              {"check": "integrity", "result": "pass", "severity": "info"}
            ]
          }
        }
      ]
    }
  ]
}
EOF
