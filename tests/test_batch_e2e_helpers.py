"""Unit tests for batch E2E helpers (no live cluster)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from kubernetes.client import ApiException  # type: ignore[import-untyped]

from tests.e2e.batch_log_contract import PROVIDER_OUTPUT_LOG_PREFIX
from tests.e2e.batch_runner import (
    _body_from_result_cr,
    _build_job_spec,
    _delete_config_map_ignore_not_found,
    _enrich_body_from_pod_logs,
    _needs_pod_log_enrichment,
    _parse_echo_token_from_pod_logs,
    _parse_provider_output_json,
    _set_config_map_job_owner,
    build_result_template,
    run_batch_query,
)
from tests.e2e.otel_verify import logs_contain_audit_logs_for_run, logs_contain_traces_for_run
from tests.e2e.skills_fixtures import (
    E2E_POD_SKILLS_DIR,
    E2E_POD_SKILLS_SRC_DIR,
    _cm_key_from_rel,
    _rel_from_cm_key,
    skill_materialize_script,
)
from tests.e2e.suite_setup import (
    BatchE2EConfig,
    load_batch_e2e_config,
    resolve_llm_secret,
    resolve_model,
)


class TestBuildResultTemplate:
    def test_analysis_template(self) -> None:
        tmpl = build_result_template(
            namespace="openshift-lightspeed",
            result_name="run-analysis-1",
            run_uid="abc123",
            step="analysis",
            agentic_run_name="run",
            session_id="sess1",
        )
        assert tmpl["kind"] == "AnalysisResult"
        assert tmpl["metadata"]["name"] == "run-analysis-1"
        assert tmpl["metadata"]["labels"]["agentic.openshift.io/run"] == "abc123"
        assert tmpl["spec"]["agenticRunName"] == "run"


class TestBodyFromResultCr:
    def test_agent_success(self) -> None:
        cr: dict[str, Any] = {
            "status": {
                "actionRequired": "False",
                "diagnosis": {"summary": "all good", "rootCause": "none"},
                "conditions": [
                    {
                        "type": "Completed",
                        "status": "True",
                        "reason": "Succeeded",
                        "message": "Step completed",
                    }
                ],
            }
        }
        body = _body_from_result_cr(cr)
        assert body["success"] is True
        assert body["summary"] == "all good"

    def test_agent_failure_reason(self) -> None:
        cr = {
            "status": {
                "failureReason": "agent timed out",
                "conditions": [
                    {
                        "type": "Completed",
                        "status": "True",
                        "reason": "Failed",
                        "message": "Step failed",
                    }
                ],
            }
        }
        body = _body_from_result_cr(cr)
        assert body["success"] is False
        assert body["summary"] == "agent timed out"

    def test_flattens_diagnosis_echo_fields(self) -> None:
        cr = {
            "status": {
                "diagnosis": {
                    "summary": "echo ok",
                    "namespaces": "fleet-alpha,fleet-beta",
                    "ticketId": "E2E-STRUCT-001",
                },
                "conditions": [
                    {
                        "type": "Completed",
                        "status": "True",
                        "reason": "Succeeded",
                        "message": "Step completed",
                    }
                ],
            }
        }
        body = _body_from_result_cr(cr)
        assert body["success"] is True
        assert body["namespaces"] == "fleet-alpha,fleet-beta"
        assert body["ticketId"] == "E2E-STRUCT-001"
        assert body["summary"] == "echo ok"


class TestSkillMaterializeScript:
    def test_copies_from_src_to_skills_root(self) -> None:
        script = skill_materialize_script()
        assert "cp -aL" in script
        assert "! -name '..*'" in script
        assert "/mnt/e2e-skills-src" in script
        assert "/app/skills" in script
        assert "/app/skills/.agents" in script

    def test_skips_kubelet_configmap_directories(self, tmp_path: Path) -> None:
        src_root = tmp_path / "src"
        dest_root = tmp_path / "skills"
        skill = src_root / "find-token"
        payload = skill / "..2026_10_01_12_00_00.123456789"
        (payload / "scripts").mkdir(parents=True)
        (payload / "SKILL.md").write_text("skill body\n", encoding="utf-8")
        (payload / "scripts" / "find-token.sh").write_text("echo token\n", encoding="utf-8")
        (skill / "..data").symlink_to(payload.name)
        (skill / "SKILL.md").symlink_to(Path("..data") / "SKILL.md")
        (skill / "scripts").symlink_to(Path("..data") / "scripts")

        script = skill_materialize_script().replace(E2E_POD_SKILLS_SRC_DIR, str(src_root))
        script = script.replace(E2E_POD_SKILLS_DIR, str(dest_root))
        subprocess.run(  # noqa: S603
            ["bash", "-c", script],  # noqa: S607
            check=True,
        )

        copied = dest_root / "find-token"
        assert (copied / "SKILL.md").read_text(encoding="utf-8") == "skill body\n"
        assert (copied / "scripts" / "find-token.sh").read_text(encoding="utf-8") == "echo token\n"
        assert not (copied / "SKILL.md").is_symlink()
        assert not (copied / "scripts").is_symlink()
        assert [path.name for path in copied.iterdir() if path.name.startswith("..")] == []


class TestSkillConfigMapKeys:
    def test_round_trip_simple_path(self) -> None:
        rel = "scripts/echo-token.sh"
        key = _cm_key_from_rel(rel)
        assert key == "scripts__echo-token.sh"
        assert _rel_from_cm_key(key) == rel

    def test_rejects_double_underscore_in_path(self) -> None:
        with pytest.raises(ValueError, match="must not contain '__'"):
            _cm_key_from_rel("docs/a__b.md")


class TestParseEchoTokenFromPodLogs:
    def test_extracts_script_stdout_json(self) -> None:
        token = "a" * 32
        logs = f'tool output: {{"token": "{token}", "status": "ok"}}\n'
        assert _parse_echo_token_from_pod_logs(logs) == token

    def test_uses_last_match_when_script_ran_multiple_times(self) -> None:
        first = "b" * 32
        second = "c" * 32
        logs = (
            f'shell: {{"token": "{first}", "status": "ok"}}\n'
            f'shell: {{"token": "{second}", "status": "ok"}}\n'
        )
        assert _parse_echo_token_from_pod_logs(logs) == second

    def test_returns_empty_when_script_output_missing(self) -> None:
        logs = '[provider:run] output: {"success": true, "token": "deadbeef", "summary": "ok"}'
        assert _parse_echo_token_from_pod_logs(logs) == ""

    def test_falls_back_to_provider_output_json(self) -> None:
        token = "d" * 32
        logs = (
            "INFO lightspeed_agentic: [provider:run] output: "
            f'{{"success": true, "summary": "token ok", "token": "{token}", "status": "ok"}}'
        )
        assert _parse_echo_token_from_pod_logs(logs) == token


class TestEnrichBodyFromPodLogs:
    def test_log_prefix_matches_sandbox_event_logger(self) -> None:
        assert PROVIDER_OUTPUT_LOG_PREFIX == "[provider:run] output: "

    def test_merges_provider_output_line(self) -> None:
        body = {"success": True, "summary": "Step completed"}
        logs = (
            "INFO lightspeed_agentic: [provider:run] output: "
            '{"success": true, "summary": "e2e-flat-ok", "ticketId": "E2E-STRUCT-001"}'
        )
        enriched = _enrich_body_from_pod_logs(body, logs)
        assert enriched["ticketId"] == "E2E-STRUCT-001"
        assert enriched["summary"] == "e2e-flat-ok"

    def test_merges_plain_text_reasoning_output(self) -> None:
        body = {"success": True, "summary": "Step completed"}
        logs = "INFO lightspeed_agentic: [provider:run] output: 391\n"
        enriched = _enrich_body_from_pod_logs(body, logs)
        assert enriched["summary"] == "391"
        assert enriched["content"] == "391"

    def test_skips_enrichment_when_cr_already_has_echo_fields(self) -> None:
        body = {
            "success": True,
            "summary": "context-echo-ok",
            "namespaces": "ns-a, ns-b",
        }
        assert _needs_pod_log_enrichment(body) is False

    def test_enriches_when_cr_summary_is_generic(self) -> None:
        body = {"success": True, "summary": "Step completed"}
        assert _needs_pod_log_enrichment(body) is True

    def test_skips_enrichment_when_cr_has_failure_reason(self) -> None:
        body = {"success": False, "summary": "agent error", "failureReason": "load_skill failed"}
        assert _needs_pod_log_enrichment(body) is False

    def test_parses_escaped_newline_logs(self) -> None:
        body = {"success": True, "summary": "Step completed"}
        logs = (
            "INFO lightspeed_agentic: [provider:run] output: "
            '{"success": true, "summary": "e2e-flat-ok", "ticketId": "E2E-STRUCT-001"}'
            "\\nINFO lightspeed_agentic: [agent] query complete"
        )
        parsed = _parse_provider_output_json(logs)
        assert parsed is not None
        assert parsed["ticketId"] == "E2E-STRUCT-001"
        enriched = _enrich_body_from_pod_logs(body, logs)
        assert enriched["ticketId"] == "E2E-STRUCT-001"


class TestLoadBatchE2EConfig:
    def test_openai_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("E2E_PROVIDER", "openai-agents")
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("LIGHTSPEED_MCP_SERVERS", raising=False)
        monkeypatch.delenv("LIGHTSPEED_REASONING_CONFIG", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        expected_secret = resolve_llm_secret("openai-agents")
        config = load_batch_e2e_config()
        assert config.lightspeed_provider == "openai"
        assert config.llm_secret == expected_secret
        assert config.model == "gpt-6-luna"
        assert config.verify_full_fixtures is False
        assert config.job_env == {
            "LIGHTSPEED_TLS_PROFILE": "IntermediateType",
            "LIGHTSPEED_TLS_MIN_VERSION": "VersionTLS12",
            "LIGHTSPEED_TLS_CIPHER_SUITES": '["ECDHE-RSA-AES128-GCM-SHA256"]',
        }

    def test_anthropic_vertex_maps_to_vertex_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("E2E_PROVIDER", "anthropic-vertex-deepagents")
        monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "my-gcp-project")
        monkeypatch.setenv("CLOUD_ML_REGION", "us-central1")

        config = load_batch_e2e_config()
        assert config.lightspeed_provider == "vertex"
        assert config.extra_env == {
            "LIGHTSPEED_MODEL_PROVIDER": "anthropic",
            "LIGHTSPEED_PROVIDER_PROJECT": "my-gcp-project",
            "LIGHTSPEED_PROVIDER_REGION": "us-central1",
        }
        assert config.llm_secret == resolve_llm_secret("anthropic-vertex-deepagents")

    def test_anthropic_bedrock_maps_to_bedrock_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("E2E_PROVIDER", "anthropic-bedrock-deepagents")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        config = load_batch_e2e_config()
        assert config.lightspeed_provider == "bedrock"
        assert config.extra_env == {"LIGHTSPEED_PROVIDER_REGION": "us-east-1"}
        assert config.llm_secret == resolve_llm_secret("anthropic-bedrock-deepagents")


class TestResolveHelpers:
    def test_resolve_model_env_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_MODEL", "custom-model")
        assert resolve_model("openai-agents") == "custom-model"

    def test_resolve_llm_secret_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_SECRET", "my-secret")
        assert resolve_llm_secret("openai-agents") == "my-secret"


class TestOtelVerify:
    RUN_UID = "a" * 32
    EXPECTED_OPERATION = "chat"
    EXPECTED_PROVIDER = "openai"

    def _span_block(
        self,
        *,
        span_uid: str | None,
        resource_uid: str | None = None,
        resource_phase: str | None = None,
        operation: str = "chat",
        provider: str = "openai",
        trace_id: str = "1" * 32,
        span_id: str = "2" * 16,
        service_name: str = "lightspeed-agentic-sandbox",
    ) -> str:
        resource_attributes = [f"     -> service.name: Str({service_name})"]
        if resource_uid is not None:
            resource_attributes.append(f"     -> agenticrun.uid: Str({resource_uid})")
        if resource_phase is not None:
            resource_attributes.append(f"     -> agenticrun.phase: Str({resource_phase})")
        span_attributes = [
            f"     -> gen_ai.operation.name: Str({operation})",
            f"     -> gen_ai.provider.name: Str({provider})",
        ]
        if span_uid is not None:
            span_attributes.insert(0, f"     -> agenticrun.uid: Str({span_uid})")
        lines = [
            "ResourceSpans #0",
            "Resource Schema URL: https://opentelemetry.io/schemas/1.41.0",
            "Resource attributes:",
            *resource_attributes,
            "ScopeSpans #0",
            "ScopeSpans Schema URL: https://opentelemetry.io/schemas/1.41.0",
            "InstrumentationScope lightspeed_agentic.audit",
            "Span #0",
            f"    Trace ID       : {trace_id}",
            f"    ID             : {span_id}",
            f"    Name           : {operation} model",
            "    Kind           : Client",
            "    Status code    : Unset",
            "Attributes:",
            *span_attributes,
        ]
        return "\n".join(lines) + "\n"

    def _log_block(
        self,
        *,
        uid: str | None = None,
        phase: str | None = "analysis",
        event: str | None = "chat",
        body: str = '{"gen_ai.operation.name":"chat","gen_ai.provider.name":"openai"}',
        trace_id: str = "1" * 32,
        span_id: str = "2" * 16,
        resource_uid: str | None = None,
        resource_phase: str | None = None,
        service_name: str = "lightspeed-agentic-sandbox",
    ) -> str:
        resource_attributes = [f"     -> service.name: Str({service_name})"]
        if resource_uid is not None:
            resource_attributes.append(f"     -> agenticrun.uid: Str({resource_uid})")
        if resource_phase is not None:
            resource_attributes.append(f"     -> agenticrun.phase: Str({resource_phase})")
        attributes = []
        if uid is not None:
            attributes.append(f"     -> agenticrun.uid: Str({uid})")
        if phase is not None:
            attributes.append(f"     -> agenticrun.phase: Str({phase})")
        if event is not None:
            attributes.append(f"     -> event: Str({event})")
        lines = [
            "ResourceLogs #0",
            "Resource Schema URL: https://opentelemetry.io/schemas/1.41.0",
            "Resource attributes:",
            *resource_attributes,
            "ScopeLogs #0",
            "ScopeLogs Schema URL: https://opentelemetry.io/schemas/1.41.0",
            "InstrumentationScope lightspeed_agentic.audit",
            "LogRecord #0",
            f"    Body: Str({body})",
            "Attributes:",
            *attributes,
            f"    Trace ID       : {trace_id}",
            f"    Span ID        : {span_id}",
        ]
        return "\n".join(lines) + "\n"

    @pytest.mark.parametrize(
        ("span_options", "log_options", "expected_traces", "expected_audit"),
        [
            pytest.param({}, None, True, False, id="trace-only"),
            pytest.param({}, {}, True, True, id="correlated-trace-and-audit"),
            pytest.param(None, None, False, False, id="no-span-markers"),
            pytest.param(
                {"span_uid": None, "resource_uid": RUN_UID},
                {},
                False,
                False,
                id="resource-uid-is-not-span-uid",
            ),
            pytest.param({"trace_id": "0" * 32}, {}, False, False, id="zero-trace-id"),
            pytest.param({"span_id": "0" * 16}, {}, False, False, id="zero-span-id"),
            pytest.param(
                {"operation": "generate_content"},
                {},
                False,
                False,
                id="wrong-operation",
            ),
            pytest.param(
                {"provider": "gcp.vertex_ai"},
                {},
                False,
                False,
                id="wrong-provider",
            ),
            pytest.param({}, {"phase": "execution"}, True, False, id="wrong-phase"),
            pytest.param({}, {"trace_id": "3" * 32}, True, False, id="unmatched-context"),
            pytest.param(None, {}, False, False, id="audit-without-source-span"),
            pytest.param({"span_uid": "b" * 32}, {}, False, False, id="different-run"),
            pytest.param({}, {"event": "invoke_agent"}, True, False, id="event-body-mismatch"),
            pytest.param(
                {},
                {"body": '{"gen_ai.operation.name":"chat","gen_ai.provider.name":"gcp.vertex_ai"}'},
                True,
                False,
                id="wrong-body-provider",
            ),
            pytest.param({}, {"trace_id": "malformed"}, True, False, id="invalid-log-trace-id"),
            pytest.param({}, {"span_id": "0" * 16}, True, False, id="zero-log-span-id"),
            pytest.param(
                {},
                {"body": '{"gen_ai.operation.name":'},
                True,
                False,
                id="malformed-body",
            ),
            pytest.param(
                {"resource_uid": RUN_UID},
                {
                    "uid": None,
                    "resource_uid": RUN_UID,
                    "body": '{"gen_ai.operation.name":"chat","gen_ai.provider.name":"openai",'
                    f'"agenticrun.uid":"{RUN_UID}"}}',
                },
                True,
                False,
                id="uid-required-on-record",
            ),
            pytest.param(
                {"resource_phase": "analysis"},
                {
                    "phase": None,
                    "resource_phase": "analysis",
                    "body": '{"gen_ai.operation.name":"chat","gen_ai.provider.name":"openai",'
                    '"agenticrun.phase":"analysis"}',
                },
                True,
                False,
                id="phase-required-on-record",
            ),
            pytest.param({}, {"event": None}, True, False, id="event-required-on-record"),
        ],
    )
    def test_correlated_telemetry_verifier(
        self,
        span_options: dict[str, Any] | None,
        log_options: dict[str, Any] | None,
        expected_traces: bool,
        expected_audit: bool,
    ) -> None:
        logs = f"agenticrun.uid={self.RUN_UID}\n"
        if span_options is not None:
            logs += self._span_block(**{"span_uid": self.RUN_UID, **span_options})
        if log_options is not None:
            logs += self._log_block(**{"uid": self.RUN_UID, **log_options})
        assert (
            logs_contain_traces_for_run(
                logs,
                self.RUN_UID,
                expected_operation=self.EXPECTED_OPERATION,
                expected_provider=self.EXPECTED_PROVIDER,
            )
            is expected_traces
        )
        assert (
            logs_contain_audit_logs_for_run(
                logs,
                self.RUN_UID,
                phase="analysis",
                expected_operation=self.EXPECTED_OPERATION,
                expected_provider=self.EXPECTED_PROVIDER,
            )
            is expected_audit
        )


class TestBuildJobSpec:
    def _config(self, *, job_env: dict[str, str] | None = None) -> BatchE2EConfig:
        llm_secret = resolve_llm_secret("openai-agents")
        return BatchE2EConfig(
            namespace="ns",
            sandbox_image="img:tag",
            service_account="sa",
            llm_secret=llm_secret,
            lightspeed_provider="openai",
            model="gpt-5-mini",
            provider_name="openai-agents",
            session_id="sess",
            otel_endpoint="",
            otel_ca_secret="",
            verify_full_fixtures=False,
            job_env=job_env or {},
        )

    def _env(self, job: Any) -> dict[str, str]:
        env = job.spec["template"]["spec"]["containers"][0]["env"]
        return {item["name"]: item["value"] for item in env}

    def test_sets_required_execution_limit_env_defaults(self) -> None:
        job = _build_job_spec(
            self._config(),
            "job-name",
            "input-cm",
            {"app": "test"},
            "run-uid",
            "analysis",
        )

        env = self._env(job)

        assert env["LIGHTSPEED_AGENT_TIMEOUT_SECONDS"] == "600"
        assert env["LIGHTSPEED_AGENT_MAX_TURNS"] == "200"

    def test_timeout_ms_override_rounds_up_to_seconds(self) -> None:
        job = _build_job_spec(
            self._config(),
            "job-name",
            "input-cm",
            {"app": "test"},
            "run-uid",
            "analysis",
            timeout_ms=1500,
        )

        assert self._env(job)["LIGHTSPEED_AGENT_TIMEOUT_SECONDS"] == "2"

    def test_passes_tls_policy_to_job(self) -> None:
        job = _build_job_spec(
            self._config(
                job_env={
                    "LIGHTSPEED_TLS_PROFILE": "IntermediateType",
                    "LIGHTSPEED_TLS_MIN_VERSION": "VersionTLS12",
                    "LIGHTSPEED_TLS_CIPHER_SUITES": '["ECDHE-RSA-AES128-GCM-SHA256"]',
                }
            ),
            "job-name",
            "input-cm",
            {"app": "test"},
            "run-uid",
            "analysis",
        )

        env = self._env(job)

        assert env["LIGHTSPEED_TLS_PROFILE"] == "IntermediateType"
        assert env["LIGHTSPEED_TLS_MIN_VERSION"] == "VersionTLS12"
        assert env["LIGHTSPEED_TLS_CIPHER_SUITES"] == '["ECDHE-RSA-AES128-GCM-SHA256"]'

    def test_job_env_overrides_execution_limit_defaults(self) -> None:
        job = _build_job_spec(
            self._config(
                job_env={
                    "LIGHTSPEED_AGENT_TIMEOUT_SECONDS": "42",
                    "LIGHTSPEED_AGENT_MAX_TURNS": "7",
                }
            ),
            "job-name",
            "input-cm",
            {"app": "test"},
            "run-uid",
            "analysis",
        )

        env = self._env(job)

        assert env["LIGHTSPEED_AGENT_TIMEOUT_SECONDS"] == "42"
        assert env["LIGHTSPEED_AGENT_MAX_TURNS"] == "7"


class TestRunBatchQuery:
    def test_job_create_failure_deletes_input_config_map(self) -> None:
        llm_secret = resolve_llm_secret("openai-agents")
        config = BatchE2EConfig(
            namespace="ns",
            sandbox_image="img:tag",
            service_account="sa",
            llm_secret=llm_secret,
            lightspeed_provider="openai",
            model="gpt-5-mini",
            provider_name="openai-agents",
            session_id="sess",
            otel_endpoint="",
            otel_ca_secret="",
            verify_full_fixtures=False,
        )
        core_api = MagicMock()
        batch_api = MagicMock()
        custom_api = MagicMock()
        batch_api.create_namespaced_job.side_effect = ApiException(status=403, reason="Forbidden")

        result = run_batch_query(
            config,
            core_api,
            batch_api,
            custom_api,
            "hello",
        )

        assert result.error is not None
        assert "Forbidden" in result.error
        core_api.delete_namespaced_config_map.assert_called_once()
        delete_args = core_api.delete_namespaced_config_map.call_args[0]
        assert delete_args[0].endswith("-input")
        assert delete_args[1] == "ns"

    def test_missing_result_cr_after_successful_job(self) -> None:
        llm_secret = resolve_llm_secret("openai-agents")
        config = BatchE2EConfig(
            namespace="ns",
            sandbox_image="img:tag",
            service_account="sa",
            llm_secret=llm_secret,
            lightspeed_provider="openai",
            model="gpt-5-mini",
            provider_name="openai-agents",
            session_id="sess",
            otel_endpoint="",
            otel_ca_secret="",
            verify_full_fixtures=False,
        )
        core_api = MagicMock()
        batch_api = MagicMock()
        custom_api = MagicMock()

        job_status = MagicMock()
        job_status.succeeded = 1
        job_status.failed = None
        batch_api.create_namespaced_job.return_value = MagicMock(
            metadata=MagicMock(uid="job-uid-abc"),
        )
        batch_api.read_namespaced_job.return_value = MagicMock(status=job_status)
        core_api.list_namespaced_pod.return_value = MagicMock(items=[])
        custom_api.get_namespaced_custom_object.side_effect = ApiException(status=404)

        result = run_batch_query(
            config,
            core_api,
            batch_api,
            custom_api,
            "hello",
        )

        assert result.job_succeeded is True
        assert result.result_cr is None
        assert result.error is not None
        assert "not found" in result.error
        core_api.replace_namespaced_config_map.assert_called_once()


class TestConfigMapJobOwner:
    def test_set_config_map_job_owner(self) -> None:
        core_api = MagicMock()
        cm = MagicMock()
        cm.metadata.owner_references = None
        core_api.read_namespaced_config_map.return_value = cm

        _set_config_map_job_owner(core_api, "ns", "e2e-input", "e2e-job", "uid-123")

        core_api.replace_namespaced_config_map.assert_called_once_with(
            "e2e-input",
            "ns",
            cm,
        )
        assert cm.metadata.owner_references is not None
        owner = cm.metadata.owner_references[0]
        assert owner.api_version == "batch/v1"
        assert owner.kind == "Job"
        assert owner.name == "e2e-job"
        assert owner.uid == "uid-123"
        assert owner.controller is True

    def test_delete_config_map_ignore_not_found(self) -> None:
        core_api = MagicMock()
        _delete_config_map_ignore_not_found(core_api, "ns", "orphan-cm")
        core_api.delete_namespaced_config_map.assert_called_once_with("orphan-cm", "ns")

    def test_delete_config_map_swallows_404(self) -> None:
        core_api = MagicMock()
        core_api.delete_namespaced_config_map.side_effect = ApiException(status=404)
        _delete_config_map_ignore_not_found(core_api, "ns", "missing-cm")
