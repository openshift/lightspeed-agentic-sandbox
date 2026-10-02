"""Batch Job runner for E2E — cluster Job lifecycle (OLS-3926).

Maps each scenario to a batch Job + Result CR, then builds a response envelope
for BDD steps. When the Result CR status is generic, enriches from the batch
pod log line documented in ``batch_log_contract`` (see ``lightspeed_agentic.logging``).
"""

from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from kubernetes.client import (  # type: ignore[import-untyped]
    ApiException,
    BatchV1Api,
    CoreV1Api,
    CustomObjectsApi,
    V1ConfigMap,
    V1Job,
    V1ObjectMeta,
    V1OwnerReference,
)

from tests.e2e.batch_log_contract import (
    PROVIDER_OUTPUT_LOG_PREFIX,
    _GENERIC_CR_SUMMARIES,
)
from tests.e2e.k8s_constants import CRD_GROUP, CRD_VERSION
from tests.e2e.skills_fixtures import (
    E2E_POD_OUTPUT_DIR,
    E2E_POD_SKILLS_DIR,
    E2E_POD_SKILLS_SRC_DIR,
    E2E_POD_SKILLS_WORKDIR,
    SKILLS_SOURCE,
    configmap_items_for_skill,
    ensure_skill_configmaps,
    skill_materialize_script,
)
from tests.e2e.suite_setup import (
    BatchE2EConfig,
    E2E_COMPONENT_LABEL,
    E2E_COMPONENT_VALUE,
    E2E_RUN_LABEL,
)

_KIND_TO_PLURAL: dict[str, str] = {
    "AnalysisResult": "analysisresults",
    "ExecutionResult": "executionresults",
    "VerificationResult": "verificationresults",
    "EscalationResult": "escalationresults",
}

_STEP_TO_KIND: dict[str, str] = {
    "analysis": "AnalysisResult",
    "execution": "ExecutionResult",
    "verification": "VerificationResult",
    "escalation": "EscalationResult",
}

E2E_DEFAULT_AGENT_TIMEOUT_SECONDS = 600
E2E_DEFAULT_AGENT_MAX_TURNS = "200"


@dataclass
class RunBatchResult:
    """Outcome of a single batch Job run."""

    job_succeeded: bool = False
    result_cr: dict[str, Any] | None = None
    pod_logs: str = ""
    termination_message: str | None = None
    error: str | None = None
    latency_seconds: float = 0.0
    job_name: str = ""
    result_name: str | None = None
    run_uid: str = ""
    step: str = "analysis"
    body: dict[str, Any] = field(default_factory=dict)
    token_file: str = ""

    @property
    def agent_succeeded(self) -> bool:
        """True when the agent step succeeded (Completed.reason=Succeeded)."""
        return bool(self.body.get("success"))


def build_result_template(
    *,
    namespace: str,
    result_name: str,
    run_uid: str,
    step: str,
    agentic_run_name: str,
    session_id: str,
) -> dict[str, Any]:
    """Build a Result CR template for ``/input/result-template``."""
    kind = _STEP_TO_KIND.get(step)
    if kind is None:
        msg = f"unknown batch step: {step!r}"
        raise ValueError(msg)
    return {
        "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
        "kind": kind,
        "metadata": {
            "name": result_name,
            "namespace": namespace,
            "labels": {
                "agentic.openshift.io/run": run_uid,
                "agentic.openshift.io/step": step,
                E2E_COMPONENT_LABEL: E2E_COMPONENT_VALUE,
                E2E_RUN_LABEL: session_id,
            },
        },
        "spec": {"agenticRunName": agentic_run_name},
    }


def _delete_config_map_ignore_not_found(
    core_api: CoreV1Api,
    namespace: str,
    name: str,
) -> None:
    """Delete an input ConfigMap; ignore 404 when create partially failed."""
    try:
        core_api.delete_namespaced_config_map(name, namespace)
    except ApiException as exc:
        if exc.status != 404:
            raise


def _set_config_map_job_owner(
    core_api: CoreV1Api,
    namespace: str,
    config_map_name: str,
    job_name: str,
    job_uid: str,
) -> None:
    """Tie input ConfigMap lifecycle to the batch Job (GC when Job TTL deletes it)."""
    cm = core_api.read_namespaced_config_map(config_map_name, namespace)
    cm.metadata.owner_references = [  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        V1OwnerReference(
            api_version="batch/v1",
            kind="Job",
            name=job_name,
            uid=job_uid,
            controller=True,
            block_owner_deletion=False,
        )
    ]
    core_api.replace_namespaced_config_map(config_map_name, namespace, cm)


def run_batch_query(
    config: BatchE2EConfig,
    core_api: CoreV1Api,
    batch_api: BatchV1Api,
    custom_api: CustomObjectsApi,
    query: str,
    *,
    system_prompt: str | None = None,
    output_schema: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    step: str = "analysis",
    wait_timeout_seconds: float = 600.0,
    job_name_prefix: str | None = None,
    timeout_ms: int | None = None,
    mount_skills: bool = False,
    job_env_overrides: Mapping[str, str] | None = None,
) -> RunBatchResult:
    """Create input ConfigMap + batch Job, wait for completion, read Result CR."""
    start = time.monotonic()
    run_uid = secrets.token_hex(16)
    stamp = int(time.time())
    job_name = _sanitize_k8s_name(f"{job_name_prefix or 'e2e'}-{stamp}-{run_uid[:8]}")
    input_cm_name = f"{job_name}-input"
    result_name = _sanitize_k8s_name(f"{job_name}-{step}-1")

    if context is None:
        context = {"targetNamespaces": ["default"]}

    skill_configmaps: dict[str, str] | None = None
    if mount_skills:
        skill_configmaps = ensure_skill_configmaps(
            core_api,
            config.namespace,
            config.session_id,
        )

    result_template = build_result_template(
        namespace=config.namespace,
        result_name=result_name,
        run_uid=run_uid,
        step=step,
        agentic_run_name=job_name,
        session_id=config.session_id,
    )

    labels = {
        E2E_RUN_LABEL: config.session_id,
        E2E_COMPONENT_LABEL: E2E_COMPONENT_VALUE,
        "agentic.openshift.io/run": run_uid,
        "agentic.openshift.io/step": step,
    }

    cm_data: dict[str, str] = {
        "query": query,
        "output-schema": json.dumps({} if output_schema is None else output_schema),
        "context": json.dumps(context),
        "result-template": json.dumps(result_template),
    }
    if system_prompt:
        cm_data["system-prompt"] = system_prompt

    result = RunBatchResult(job_name=job_name, result_name=result_name, run_uid=run_uid, step=step)

    cm_created = False
    try:
        core_api.create_namespaced_config_map(
            config.namespace,
            V1ConfigMap(
                metadata=V1ObjectMeta(name=input_cm_name, labels=labels),
                data=cm_data,
            ),
        )
        cm_created = True
        created_job = batch_api.create_namespaced_job(
            config.namespace,
            _build_job_spec(
                config,
                job_name,
                input_cm_name,
                labels,
                run_uid,
                step,
                timeout_ms=timeout_ms,
                skill_configmaps=skill_configmaps,
                job_env_overrides=job_env_overrides,
            ),
        )
        job_uid = created_job.metadata.uid  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        if job_uid:
            _set_config_map_job_owner(
                core_api,
                config.namespace,
                input_cm_name,
                job_name,
                job_uid,
            )
    except ApiException as exc:
        if cm_created:
            _delete_config_map_ignore_not_found(core_api, config.namespace, input_cm_name)
        result.error = f"create batch resources: {exc.reason}"
        result.latency_seconds = time.monotonic() - start
        return result

    job_ok, wait_err = _wait_for_job(
        batch_api,
        core_api,
        config.namespace,
        job_name,
        wait_timeout_seconds,
    )
    result.job_succeeded = job_ok
    result.pod_logs = _fetch_job_pod_logs(core_api, config.namespace, job_name)
    result.termination_message = _fetch_termination_message(core_api, config.namespace, job_name)
    if mount_skills:
        result.token_file = _parse_echo_token_from_pod_logs(result.pod_logs)

    if not job_ok:
        result.error = wait_err or "batch job did not succeed"
        result.latency_seconds = time.monotonic() - start
        return result

    kind = result_template["kind"]
    try:
        result.result_cr = custom_api.get_namespaced_custom_object(  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=config.namespace,
            plural=_KIND_TO_PLURAL[kind],
            name=result_name,
        )
    except ApiException as exc:
        if exc.status == 404:
            result.error = f"Result CR {kind}/{result_name} not found after successful Job"
        else:
            result.error = f"read Result CR: {exc.reason}"
        result.latency_seconds = time.monotonic() - start
        return result

    result.body = _body_from_result_cr(result.result_cr)  # pyright: ignore[reportArgumentType]
    if _needs_pod_log_enrichment(result.body):
        result.body = _enrich_body_from_pod_logs(result.body, result.pod_logs)
    result.latency_seconds = time.monotonic() - start
    return result


def _needs_pod_log_enrichment(body: dict[str, Any]) -> bool:
    """Return True when Result CR status lacks agent echo fields BDD expects."""
    if body.get("failureReason"):
        return False
    summary = str(body.get("summary", "")).strip().lower()
    if summary in _GENERIC_CR_SUMMARIES or not summary:
        return True
    echo_keys = (
        "namespaces",
        "firstFailureReason",
        "approvedTitle",
        "firstCommand",
        "ticketId",
        "token",
        "items",
        "onlyFieldAlpha",
    )
    return not any(key in body for key in echo_keys)


_PROVIDER_OUTPUT_LOG_PREFIX = PROVIDER_OUTPUT_LOG_PREFIX


def _normalize_pod_logs(pod_logs: str) -> str:
    """Decode escaped newlines returned by some Kubernetes log API responses."""
    if pod_logs.count("\n") <= 1 and "\\n" in pod_logs:
        return pod_logs.replace("\\n", "\n")
    return pod_logs


# echo-token.sh stdout: {"token": "<32 hex>", "status": "ok"}
_ECHO_TOKEN_SCRIPT_JSON_RE = re.compile(r'\{"token":\s*"([0-9a-f]{32})"\s*,\s*"status":\s*"ok"\}')
_ECHO_TOKEN_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


def _parse_echo_token_from_pod_logs(pod_logs: str) -> str:
    """Return token from echo-token.sh stdout embedded in batch pod logs."""
    normalized = _normalize_pod_logs(pod_logs)
    matches = _ECHO_TOKEN_SCRIPT_JSON_RE.findall(normalized)
    if matches:
        return matches[-1]  # type: ignore[no-any-return]
    # Some providers (e.g. Gemini ADK) omit raw tool stdout from pod logs but include
    # the script token in the final structured agent output line.
    parsed = _parse_provider_output_json(pod_logs)
    if parsed:
        token = str(parsed.get("token", "")).strip()
        status = str(parsed.get("status", "")).strip()
        if _ECHO_TOKEN_HEX_RE.match(token) and status == "ok":
            return token
    return ""


def _parse_provider_output_json(pod_logs: str) -> dict[str, Any] | None:
    """Extract the agent JSON object logged after ``[provider:run] output:``."""
    normalized = _normalize_pod_logs(pod_logs)
    marker = _PROVIDER_OUTPUT_LOG_PREFIX
    idx = normalized.find(marker)
    if idx == -1:
        return None
    tail = normalized[idx + len(marker) :].lstrip()
    try:
        parsed, _end = json.JSONDecoder().raw_decode(tail)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _enrich_body_from_pod_logs(body: dict[str, Any], pod_logs: str) -> dict[str, Any]:
    """Merge agent output from batch pod logs when Result CR status lacks echo fields."""
    parsed = _parse_provider_output_json(pod_logs)
    if parsed is not None:
        merged = dict(body)
        merged.update(parsed)
        return merged

    normalized = _normalize_pod_logs(pod_logs)
    marker = _PROVIDER_OUTPUT_LOG_PREFIX
    idx = normalized.find(marker)
    if idx == -1:
        return body
    text = normalized[idx + len(marker) :].split("\n", 1)[0].strip()
    if not text or not body.get("success"):
        return body
    merged = dict(body)
    if not merged.get("summary") or merged.get("summary") == "Step completed":
        merged["summary"] = text
    merged.setdefault("content", text)
    return merged


def _body_from_result_cr(result_cr: dict[str, Any]) -> dict[str, Any]:
    """Map Result CR status to a response body for BDD assertions."""
    status = result_cr.get("status") or {}
    body: dict[str, Any] = {}

    for key in (
        "options",
        "actionRequired",
        "diagnosis",
        "actionsTaken",
        "checks",
        "summary",
        "content",
    ):
        if key in status:
            body[key] = status[key]

    failure_reason = status.get("failureReason")
    if failure_reason:
        body["success"] = False
        body["summary"] = failure_reason
        return body

    completed = _condition(status.get("conditions") or [], "Completed")
    if completed and completed.get("reason") == "Failed":
        body["success"] = False
        body["summary"] = completed.get("message") or "step failed"
        return body

    body["success"] = True
    if "summary" not in body:
        diagnosis = body.get("diagnosis")
        if isinstance(diagnosis, dict) and diagnosis.get("summary"):
            body["summary"] = diagnosis["summary"]
        else:
            body["summary"] = completed.get("message") if completed else "step completed"

    diagnosis = body.get("diagnosis")
    if isinstance(diagnosis, dict):
        for key, value in diagnosis.items():
            body.setdefault(key, value)

    return body


def _condition(conditions: list[dict[str, Any]], cond_type: str) -> dict[str, Any] | None:
    for cond in conditions:
        if cond.get("type") == cond_type:
            return cond
    return None


def _job_container_security_context() -> dict[str, Any]:
    return {
        "allowPrivilegeEscalation": False,
        "runAsNonRoot": True,
        "capabilities": {"drop": ["ALL"]},
        "seccompProfile": {"type": "RuntimeDefault"},
    }


def _build_job_spec(
    config: BatchE2EConfig,
    job_name: str,
    input_cm_name: str,
    labels: dict[str, str],
    run_uid: str,
    step: str,
    timeout_ms: int | None = None,
    skill_configmaps: dict[str, str] | None = None,
    job_env_overrides: Mapping[str, str] | None = None,
) -> V1Job:
    otel_enabled = bool(config.otel_endpoint)
    env = [
        {"name": "LIGHTSPEED_PROVIDER", "value": config.lightspeed_provider},
        {"name": "LIGHTSPEED_MODEL", "value": config.model},
        {"name": "LIGHTSPEED_AUDIT_ENABLED", "value": "true"},
        {"name": "LIGHTSPEED_AGENTICRUN_UID", "value": run_uid},
        {"name": "LIGHTSPEED_AGENTICRUN_STEP", "value": step},
    ]
    for key, value in config.extra_env.items():
        env.append({"name": key, "value": value})
    job_env = {**config.job_env, **(job_env_overrides or {})}
    for key, value in job_env.items():
        env.append({"name": key, "value": value})

    env_names = {item["name"] for item in env}
    if "LIGHTSPEED_AGENT_TIMEOUT_SECONDS" not in env_names:
        timeout_seconds = E2E_DEFAULT_AGENT_TIMEOUT_SECONDS
        if timeout_ms is not None:
            timeout_seconds = max(1, (timeout_ms + 999) // 1000)
        env.append({"name": "LIGHTSPEED_AGENT_TIMEOUT_SECONDS", "value": str(timeout_seconds)})
    if "LIGHTSPEED_AGENT_MAX_TURNS" not in env_names:
        env.append({"name": "LIGHTSPEED_AGENT_MAX_TURNS", "value": E2E_DEFAULT_AGENT_MAX_TURNS})
    if "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED" not in env_names:
        env.append({"name": "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED", "value": "false"})
    if otel_enabled:
        env.extend(
            [
                {"name": "OTEL_EXPORTER_OTLP_ENDPOINT", "value": config.otel_endpoint},
                {"name": "OTEL_EXPORTER_OTLP_PROTOCOL", "value": "grpc"},
            ]
        )

    volumes: list[dict[str, Any]] = [
        {"name": "input", "configMap": {"name": input_cm_name}},
        {"name": "llm-credentials", "secret": {"secretName": config.llm_secret}},
    ]
    volume_mounts: list[dict[str, Any]] = [
        {"name": "input", "mountPath": "/input", "readOnly": True},
        {
            "name": "llm-credentials",
            "mountPath": "/var/run/secrets/llm-credentials",
            "readOnly": True,
        },
    ]
    if otel_enabled and config.otel_ca_secret:
        volumes.append({"name": "otel-ca", "secret": {"secretName": config.otel_ca_secret}})
        volume_mounts.append(
            {
                "name": "otel-ca",
                "mountPath": "/var/run/secrets/lightspeed/tls/otel-ca",
                "readOnly": True,
            }
        )
    init_containers: list[dict[str, Any]] = []
    init_volume_mounts: list[dict[str, Any]] = []
    if skill_configmaps:
        volumes.append({"name": "skills-root", "emptyDir": {}})
        volume_mounts.append({"name": "skills-root", "mountPath": E2E_POD_SKILLS_DIR})
        for skill_name, cm_name in skill_configmaps.items():
            vol_name = _sanitize_k8s_name(f"skill-{skill_name}")
            volumes.append(
                {
                    "name": vol_name,
                    "configMap": {
                        "name": cm_name,
                        "items": configmap_items_for_skill(SKILLS_SOURCE / skill_name),
                        "defaultMode": 0o555,
                    },
                }
            )
            init_volume_mounts.append(
                {
                    "name": vol_name,
                    "mountPath": f"{E2E_POD_SKILLS_SRC_DIR}/{skill_name}",
                    "readOnly": True,
                }
            )
        init_containers.append(
            {
                "name": "materialize-skills",
                "image": config.sandbox_image,
                "imagePullPolicy": "IfNotPresent",
                "command": ["bash", "-c"],
                "args": [skill_materialize_script()],
                "volumeMounts": [
                    *init_volume_mounts,
                    {"name": "skills-root", "mountPath": E2E_POD_SKILLS_DIR},
                ],
                "securityContext": _job_container_security_context(),
            }
        )
        volumes.append({"name": "e2e-output", "emptyDir": {}})
        volume_mounts.append({"name": "e2e-output", "mountPath": E2E_POD_OUTPUT_DIR})
        env.extend(
            [
                {"name": "LIGHTSPEED_SKILLS_DIR", "value": E2E_POD_SKILLS_DIR},
                {"name": "E2E_OUTPUT_DIR", "value": E2E_POD_OUTPUT_DIR},
            ]
        )

    pod_spec: dict[str, Any] = {
        "serviceAccountName": config.service_account,
        "automountServiceAccountToken": True,
        "restartPolicy": "Never",
        "securityContext": {
            "runAsNonRoot": True,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "volumes": volumes,
        "containers": [
            {
                "name": "agent",
                "image": config.sandbox_image,
                "imagePullPolicy": "IfNotPresent",
                "terminationMessagePolicy": "FallbackToLogsOnError",
                "securityContext": _job_container_security_context(),
                "envFrom": [{"secretRef": {"name": config.llm_secret}}],
                "env": env,
                "volumeMounts": volume_mounts,
            }
        ],
    }
    if init_containers:
        pod_spec["initContainers"] = init_containers

    return V1Job(
        metadata=V1ObjectMeta(name=job_name, labels=labels),
        spec={
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 3600,
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    )


def _wait_for_job(
    batch_api: BatchV1Api,
    core_api: CoreV1Api,
    namespace: str,
    job_name: str,
    timeout_seconds: float,
) -> tuple[bool, str | None]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        job = batch_api.read_namespaced_job(job_name, namespace)
        status = job.status  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        if status and status.succeeded:
            return True, None
        if status and status.failed:
            msg = _fetch_termination_message(core_api, namespace, job_name)
            return False, msg or "job failed"
        time.sleep(2.0)
    return False, f"timeout after {timeout_seconds}s waiting for job/{job_name}"


def _fetch_job_pod_logs(core_api: CoreV1Api, namespace: str, job_name: str) -> str:
    pods = core_api.list_namespaced_pod(namespace=namespace, label_selector=f"job-name={job_name}")
    if not pods.items:  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        return ""
    pod_name = pods.items[0].metadata.name  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess, reportIndexIssue]
    try:
        raw = core_api.read_namespaced_pod_log(name=pod_name, namespace=namespace, tail_lines=200)  # pyright: ignore[reportArgumentType]
    except ApiException:
        return ""
    return _normalize_pod_logs(raw)  # pyright: ignore[reportArgumentType]


def _fetch_termination_message(core_api: CoreV1Api, namespace: str, job_name: str) -> str | None:
    pods = core_api.list_namespaced_pod(namespace=namespace, label_selector=f"job-name={job_name}")
    if not pods.items:  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        return None
    statuses = pods.items[0].status.container_statuses or []  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess, reportIndexIssue]
    if not statuses:
        return None
    terminated = statuses[0].state.terminated
    if terminated is None:
        return None
    return terminated.message  # type: ignore[no-any-return]


def _sanitize_k8s_name(value: str) -> str:
    cleaned = value.lower()
    for ch in ". _":
        cleaned = cleaned.replace(ch, "-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned[:63].rstrip("-")
