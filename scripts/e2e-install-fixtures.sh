#!/usr/bin/env bash
# Manual install of sandbox e2e cluster fixtures (OLS-3926).
#
# Installs on the current OpenShift cluster (oc/kubectl context):
#   - Namespace
#   - Agentic Result CRDs (from lightspeed-agentic-operator)
#   - Sandbox batch ServiceAccount + RBAC (Result CR create + status update)
#   - OTEL collector (TLS OTLP gRPC + structured OTLP/HTTP test evidence)
#   - OTEL CA secret for sandbox batch Jobs
#
# Does NOT deploy the agentic operator controller.
#
# Usage:
#   OPERATOR_REPO=/path/to/lightspeed-agentic-operator \
#     bash scripts/e2e-install-fixtures.sh
#
# Optional env:
#   E2E_NAMESPACE          default: openshift-lightspeed
#   OPERATOR_REPO          default: ../lightspeed-agentic-operator
#   OTEL_COLLECTOR_IMAGE   default: quay.io/redhat-user-workloads/crt-nshift-lightspeed-tenant/lightspeed-otel-collector:main

set -euo pipefail
trap 'echo "error: $0 line $LINENO: command \"$BASH_COMMAND\" exited with status $?" >&2' ERR

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

E2E_NAMESPACE="${E2E_NAMESPACE:-openshift-lightspeed}"
OPERATOR_REPO="${OPERATOR_REPO:-${ROOT}/../lightspeed-agentic-operator}"
OTEL_COLLECTOR_IMAGE="${OTEL_COLLECTOR_IMAGE:-quay.io/redhat-user-workloads/crt-nshift-lightspeed-tenant/lightspeed-otel-collector:main}"
SANDBOX_SA="${SANDBOX_SA:-lightspeed-sandbox-e2e}"

if ! command -v oc >/dev/null 2>&1; then
    echo "error: oc not found" >&2
    exit 1
fi

if [ ! -d "${OPERATOR_REPO}/config/crd/bases" ]; then
    echo "error: OPERATOR_REPO must point at lightspeed-agentic-operator (missing config/crd/bases)" >&2
    echo "  set OPERATOR_REPO=/path/to/lightspeed-agentic-operator" >&2
    exit 1
fi

echo "=== Sandbox e2e fixture install ==="
echo "  cluster:    $(oc config current-context)"
echo "  namespace:  ${E2E_NAMESPACE}"
echo "  operator:   ${OPERATOR_REPO}"
echo "  otel image: ${OTEL_COLLECTOR_IMAGE}"
echo "==================================="

echo "Creating namespace ${E2E_NAMESPACE}..."
oc create namespace "${E2E_NAMESPACE}" --dry-run=client -o yaml | oc apply -f -

echo "Installing Result CRDs..."
for crd in \
    agentic.openshift.io_analysisresults.yaml \
    agentic.openshift.io_executionresults.yaml \
    agentic.openshift.io_verificationresults.yaml \
    agentic.openshift.io_escalationresults.yaml; do
    oc apply -f "${OPERATOR_REPO}/config/crd/bases/${crd}"
done

echo "Waiting for Result CRDs to become Established..."
for crd in \
    analysisresults.agentic.openshift.io \
    executionresults.agentic.openshift.io \
    verificationresults.agentic.openshift.io \
    escalationresults.agentic.openshift.io; do
    oc wait --for=condition=Established "crd/${crd}" --timeout=120s
done

echo "Creating sandbox batch ServiceAccount ${SANDBOX_SA}..."
oc apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${SANDBOX_SA}
  namespace: ${E2E_NAMESPACE}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ${SANDBOX_SA}-result-publisher
  namespace: ${E2E_NAMESPACE}
rules:
  - apiGroups: ["agentic.openshift.io"]
    resources:
      - analysisresults
      - executionresults
      - verificationresults
      - escalationresults
    verbs: ["create", "get", "list", "watch", "update", "patch"]
  - apiGroups: ["agentic.openshift.io"]
    resources:
      - analysisresults/status
      - executionresults/status
      - verificationresults/status
      - escalationresults/status
    verbs: ["get", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ${SANDBOX_SA}-result-publisher
  namespace: ${E2E_NAMESPACE}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: ${SANDBOX_SA}-result-publisher
subjects:
  - kind: ServiceAccount
    name: ${SANDBOX_SA}
    namespace: ${E2E_NAMESPACE}
EOF

echo "Deploying OTEL collector..."

oc apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: lightspeed-otel-collector-config
  namespace: ${E2E_NAMESPACE}
data:
  config.yaml: |
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: "0.0.0.0:4317"
            tls:
              cert_file: /var/run/secrets/serving-cert/tls.crt
              key_file: /var/run/secrets/serving-cert/tls.key
    exporters:
      otlphttp/e2e:
        endpoint: "http://lightspeed-mock-mcp.${E2E_NAMESPACE}.svc:${MOCK_MCP_PORT:-19090}"
        encoding: json
        compression: none
    extensions:
      health_check:
        endpoint: "0.0.0.0:13133"
    service:
      extensions: [health_check]
      pipelines:
        traces:
          receivers: [otlp]
          exporters: [otlphttp/e2e]
        logs:
          receivers: [otlp]
          exporters: [otlphttp/e2e]
---
apiVersion: v1
kind: Service
metadata:
  name: lightspeed-otel-collector
  namespace: ${E2E_NAMESPACE}
  annotations:
    service.beta.openshift.io/serving-cert-secret-name: lightspeed-otel-collector-cert
spec:
  selector:
    app: lightspeed-otel-collector
  type: ClusterIP
  ports:
    - name: otlp-grpc
      port: 4317
      protocol: TCP
      targetPort: otlp-grpc
    - name: admin
      port: 8443
      protocol: TCP
      targetPort: admin
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: lightspeed-otel-collector-cabundle
  namespace: ${E2E_NAMESPACE}
  annotations:
    service.beta.openshift.io/inject-cabundle: "true"
data: {}
EOF

echo "Waiting for OTEL collector TLS secret..."
for _ in $(seq 1 30); do
    if oc get secret lightspeed-otel-collector-cert -n "${E2E_NAMESPACE}" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
oc get secret lightspeed-otel-collector-cert -n "${E2E_NAMESPACE}"

echo "Waiting for CA bundle injection..."
CA_BUNDLE=""
for _ in $(seq 1 30); do
    CA_BUNDLE=$(oc get configmap lightspeed-otel-collector-cabundle -n "${E2E_NAMESPACE}" \
        -o jsonpath='{.data.service-ca\.crt}' 2>/dev/null || true)
    if [ -n "${CA_BUNDLE}" ]; then
        break
    fi
    sleep 2
done
if [ -z "${CA_BUNDLE}" ]; then
    echo "error: CA bundle not injected after 60s" >&2
    exit 1
fi

TMPCA=$(mktemp)
echo "${CA_BUNDLE}" > "${TMPCA}"
oc create secret generic lightspeed-otel-ca -n "${E2E_NAMESPACE}" \
    --from-file=otel-ca.crt="${TMPCA}" --dry-run=client -o yaml | oc apply -f -
rm -f "${TMPCA}"

oc apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lightspeed-otel-collector
  namespace: ${E2E_NAMESPACE}
  labels:
    app: lightspeed-otel-collector
spec:
  replicas: 1
  selector:
    matchLabels:
      app: lightspeed-otel-collector
  template:
    metadata:
      labels:
        app: lightspeed-otel-collector
    spec:
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
      containers:
        - name: otel-collector
          image: ${OTEL_COLLECTOR_IMAGE}
          args: ["--config=/etc/otel/config.yaml"]
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            seccompProfile:
              type: RuntimeDefault
          ports:
            - name: otlp-grpc
              containerPort: 4317
              protocol: TCP
            - name: health
              containerPort: 13133
              protocol: TCP
            - name: admin
              containerPort: 8443
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /
              port: health
            initialDelaySeconds: 10
            periodSeconds: 15
          readinessProbe:
            httpGet:
              path: /
              port: health
            initialDelaySeconds: 5
            periodSeconds: 10
          volumeMounts:
            - name: config
              mountPath: /etc/otel
              readOnly: true
            - name: serving-cert
              mountPath: /var/run/secrets/serving-cert
              readOnly: true
      volumes:
        - name: config
          configMap:
            name: lightspeed-otel-collector-config
        - name: serving-cert
          secret:
            secretName: lightspeed-otel-collector-cert
EOF

echo "Waiting for OTEL collector rollout..."
# Existing pods do not automatically reload the mounted Collector configuration.
oc rollout restart deployment/lightspeed-otel-collector -n "${E2E_NAMESPACE}"
oc rollout status deployment/lightspeed-otel-collector -n "${E2E_NAMESPACE}" --timeout=180s

MOCK_MCP_IMAGE="${MOCK_MCP_IMAGE:-quay.io/redhat-user-workloads/crt-nshift-lightspeed-tenant/lightspeed-agentic-sandbox:main}"
MOCK_MCP_PORT="${MOCK_MCP_PORT:-19090}"

echo "Deploying mock MCP server (image: ${MOCK_MCP_IMAGE})..."
oc create configmap lightspeed-mock-mcp-script \
    --namespace="${E2E_NAMESPACE}" \
    --from-file=mock_mcp_server.py="${ROOT}/tests/e2e/mock_mcp_server.py" \
    --dry-run=client -o yaml | oc apply -f -

oc apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: lightspeed-mock-mcp
  namespace: ${E2E_NAMESPACE}
  labels:
    app: lightspeed-mock-mcp
spec:
  selector:
    app: lightspeed-mock-mcp
  type: ClusterIP
  ports:
    - name: mcp
      port: ${MOCK_MCP_PORT}
      protocol: TCP
      targetPort: mcp
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lightspeed-mock-mcp
  namespace: ${E2E_NAMESPACE}
  labels:
    app: lightspeed-mock-mcp
spec:
  replicas: 1
  selector:
    matchLabels:
      app: lightspeed-mock-mcp
  template:
    metadata:
      labels:
        app: lightspeed-mock-mcp
    spec:
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: mock-mcp
          image: ${MOCK_MCP_IMAGE}
          imagePullPolicy: IfNotPresent
          args:
            - python3.12
            - /scripts/mock_mcp_server.py
            - --port
            - "${MOCK_MCP_PORT}"
          securityContext:
            allowPrivilegeEscalation: false
            runAsNonRoot: true
            capabilities:
              drop: ["ALL"]
            seccompProfile:
              type: RuntimeDefault
          ports:
            - name: mcp
              containerPort: ${MOCK_MCP_PORT}
              protocol: TCP
          readinessProbe:
            tcpSocket:
              port: mcp
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            tcpSocket:
              port: mcp
            initialDelaySeconds: 10
            periodSeconds: 15
          volumeMounts:
            - name: script
              mountPath: /scripts
              readOnly: true
      volumes:
        - name: script
          configMap:
            name: lightspeed-mock-mcp-script
            defaultMode: 0555
EOF

echo "Waiting for mock MCP server rollout..."
# Reload mounted fixture code even when the image/pod template has not changed.
oc rollout restart deployment/lightspeed-mock-mcp -n "${E2E_NAMESPACE}"
oc rollout status deployment/lightspeed-mock-mcp -n "${E2E_NAMESPACE}" --timeout=180s

MCP_URL="http://lightspeed-mock-mcp.${E2E_NAMESPACE}.svc:${MOCK_MCP_PORT}/mcp"
OTEL_ENDPOINT="lightspeed-otel-collector.${E2E_NAMESPACE}.svc:4317"
echo
echo "=== Install complete ==="
echo "Namespace:              ${E2E_NAMESPACE}"
echo "Sandbox SA:             ${SANDBOX_SA}"
echo "OTEL gRPC endpoint:     ${OTEL_ENDPOINT}"
echo "OTEL CA secret:         lightspeed-otel-ca (key: otel-ca.crt)"
echo "Mock MCP URL:           ${MCP_URL}"
echo "LLM credentials:      bash scripts/e2e-install-openai-creds.sh"
echo
echo "Batch Job env (for e2e):"
echo "  LIGHTSPEED_AUDIT_ENABLED=true"
echo "  LIGHTSPEED_MCP_SERVERS=[{\"name\":\"mock-ocp-mcp\",\"url\":\"${MCP_URL}\"}]"
echo "  OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_ENDPOINT}"
echo "  OTEL_EXPORTER_OTLP_PROTOCOL=grpc"
echo "  OTEL_EXPORTER_OTLP_CERTIFICATE=/etc/otel-ca/otel-ca.crt"
echo "  LIGHTSPEED_AGENTICRUN_UID=<test-run-uid>"
echo "  LIGHTSPEED_AGENTICRUN_STEP=analysis"
echo
echo "Verify OTEL collector logs:"
echo "  oc logs -n ${E2E_NAMESPACE} deployment/lightspeed-otel-collector -f"
echo
oc get crd | grep agentic.openshift.io || true
oc get all -n "${E2E_NAMESPACE}"
