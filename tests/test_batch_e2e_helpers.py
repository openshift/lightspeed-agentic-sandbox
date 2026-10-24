"""Unit tests for batch E2E helpers (no live cluster)."""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import urlsplit

import jsonschema
import pytest
from kubernetes.client import (  # type: ignore[import-untyped]
    ApiClient,
    ApiException,
    Configuration,
    CoreV1Api,
)
from urllib3.response import HTTPResponse

from lightspeed_agentic.logging import EventLogger
from lightspeed_agentic.types import ResultEvent
from tests.e2e.batch_log_contract import PROVIDER_OUTPUT_LOG_PREFIX
from tests.e2e.batch_runner import (
    _body_from_result_cr,
    _build_job_spec,
    _delete_config_map_ignore_not_found,
    _enrich_body_from_pod_logs,
    _fetch_job_pod_logs,
    _needs_pod_log_enrichment,
    _parse_echo_token_from_pod_logs,
    _parse_provider_output_json,
    _set_config_map_job_owner,
    build_result_template,
    run_batch_query,
)
from tests.e2e.k8s_helpers import fetch_pod_logs
from tests.e2e.otel_verify import fetch_otel_collector_logs
from tests.e2e.schemas_contract import ECHO_TOKEN_SCHEMA
from tests.e2e.skills_fixtures import (
    E2E_POD_SKILLS_DIR,
    E2E_POD_SKILLS_SRC_DIR,
    _cm_key_from_rel,
    _rel_from_cm_key,
    skill_materialize_script,
)
from tests.e2e.suite_setup import (
    BatchE2EConfig,
    _session_job_env,
    load_batch_e2e_config,
    resolve_llm_secret,
    resolve_model,
)


@pytest.mark.parametrize(
    ("provider_name", "supported"),
    [
        ("anthropic-vertex-deepagents", True),
        ("anthropic-bedrock-deepagents", True),
        ("anthropic-deepagents", True),
        ("openai-agents", True),
        ("azure-openai-agents", True),
        ("unsupported", False),
    ],
)
def test_inspection_provider_guard_admits_migrated_providers(
    monkeypatch: pytest.MonkeyPatch, provider_name: str, supported: bool
) -> None:
    pytest.importorskip("pytest_bdd")
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "e2e"))
    from tests.e2e.steps.given import provider_supports_tool_result_inspection

    try:
        provider_supports_tool_result_inspection(provider_name)
    except pytest.skip.Exception:
        assert not supported, f"DeepAgents provider {provider_name} was incorrectly skipped"
    else:
        assert supported, f"Legacy provider {provider_name} was incorrectly admitted"


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


class TestEchoTokenSchema:
    def test_echo_token_schema_accepts_32_character_lowercase_hex(self) -> None:
        jsonschema.validate(
            {
                "summary": "argument:literal-test with spaces",
                "token": "0123456789abcdef0123456789abcdef",
                "status": "ok",
            },
            ECHO_TOKEN_SCHEMA,
        )

    @pytest.mark.parametrize(
        "token",
        [
            "a" * 31,
            "a" * 33,
            "1aefa1389b39d57d57d32a05b13b0f5cae6",
            "A" * 32,
            "g" * 32,
        ],
        ids=["too-short", "too-long", "ci-regression", "uppercase", "not-hex"],
    )
    def test_echo_token_schema_rejects_malformed_tokens(self, token: str) -> None:
        with pytest.raises(jsonschema.ValidationError) as exc_info:
            jsonschema.validate(
                {"summary": "argument:literal-test with spaces", "token": token, "status": "ok"},
                ECHO_TOKEN_SCHEMA,
            )
        assert list(exc_info.value.path) == ["token"]

    def test_anthropic_conversion_enforces_e2e_token_length(self) -> None:
        from pydantic import ValidationError

        from lightspeed_agentic.providers.deepagents import _json_schema_to_pydantic

        model = _json_schema_to_pydantic(ECHO_TOKEN_SCHEMA)
        body = {
            "summary": "argument:literal-test with spaces",
            "token": "0123456789abcdef0123456789abcdef",
            "status": "ok",
        }
        assert model.model_validate(body).model_dump() == body
        invalid_body = {**body, "token": "1aefa1389b39d57d57d32a05b13b0f5cae6"}
        with pytest.raises(ValidationError) as exc_info:
            model.model_validate(invalid_body)
        assert exc_info.value.errors()[0]["loc"] == ("token",)


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

    @pytest.mark.parametrize(
        ("output_json", "expected"),
        [
            ('{"summary": "first\\nsecond"}', "first\nsecond"),
            (r'{"summary": "literal \\n and C:\\namespaces"}', r"literal \n and C:\namespaces"),
        ],
    )
    def test_preserves_json_escapes_in_single_output_line(
        self, output_json: str, expected: str
    ) -> None:
        logs = PROVIDER_OUTPUT_LOG_PREFIX + output_json
        parsed = _parse_provider_output_json(logs)
        assert parsed is not None
        assert parsed["summary"] == expected
        enriched = _enrich_body_from_pod_logs({"success": True, "summary": "Step completed"}, logs)
        assert enriched["summary"] == expected

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

    @pytest.mark.parametrize(
        ("google_location", "cloud_ml_region", "expected_region"),
        [
            ("us-central1", "us-east5", "us-east5"),
            (None, "us-east5", "us-east5"),
            ("us-central1", None, "us-central1"),
            (None, None, None),
        ],
    )
    def test_gemini_vertex_forwards_configured_region_without_inventing_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        google_location: str | None,
        cloud_ml_region: str | None,
        expected_region: str | None,
    ) -> None:
        monkeypatch.setenv("E2E_PROVIDER", "gemini-vertex-adk")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-gcp-project")
        if google_location is None:
            monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
        else:
            monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", google_location)
        if cloud_ml_region is None:
            monkeypatch.delenv("CLOUD_ML_REGION", raising=False)
        else:
            monkeypatch.setenv("CLOUD_ML_REGION", cloud_ml_region)

        config = load_batch_e2e_config()

        expected_env = {
            "LIGHTSPEED_MODEL_PROVIDER": "google",
            "LIGHTSPEED_PROVIDER_PROJECT": "my-gcp-project",
        }
        if expected_region is not None:
            expected_env["LIGHTSPEED_PROVIDER_REGION"] = expected_region
        assert config.extra_env == expected_env

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

    def test_session_job_env_does_not_enable_tool_output_inspection(self) -> None:
        assert "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED" not in _session_job_env()

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
        assert env["LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED"] == "false"

    def test_applies_job_env_override_without_mutating_config(self) -> None:
        config = self._config()
        job = _build_job_spec(
            config,
            "job-name",
            "input-cm",
            {"app": "test"},
            "run-uid",
            "analysis",
            job_env_overrides={"LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED": "true"},
        )

        assert self._env(job)["LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED"] == "true"
        assert "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED" not in config.job_env

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
                    "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED": "true",
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
        assert env["LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED"] == "true"


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


@pytest.fixture
def log_http(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    class ObservedResponse(HTTPResponse):
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            super().close()

    class UnreadableBody(BytesIO):
        def read(self, *_args: Any, **_kwargs: Any) -> bytes:
            raise RuntimeError("offline log read failed")

    state = SimpleNamespace(body=b"", read_error=False, responses=[], queries=[])

    def respond(method: str, url: str, **kwargs: Any) -> HTTPResponse:
        assert method == "GET"
        path = urlsplit(url).path
        if path.endswith("/log"):
            state.queries.append(dict(kwargs.get("fields") or []))
            body = UnreadableBody() if state.read_error else BytesIO(state.body)
            response = ObservedResponse(
                body=body, status=200, headers={"Content-Type": "text/plain"}, preload_content=False
            )
            state.responses.append(response)
            return response
        assert path == "/api/v1/namespaces/ns/pods"
        return HTTPResponse(
            body=json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "PodList",
                    "metadata": {},
                    "items": [
                        {"metadata": {"name": "pod-a", "namespace": "ns", "uid": "offline-uid"}}
                    ],
                }
            ).encode(),
            status=200,
            headers={"Content-Type": "application/json"},
        )

    with ApiClient(Configuration(host="https://offline-kubernetes.invalid")) as api:
        monkeypatch.setattr(api.rest_client.pool_manager, "request", respond)
        state.core = CoreV1Api(api)
        yield state


def _read_logs(reader: str, core: CoreV1Api) -> str:
    if reader == "batch":
        return _fetch_job_pod_logs(core, "ns", "offline-job")
    if reader == "pods":
        return fetch_pod_logs(core, namespace="ns", label_selector="app=test", tail_lines=7)[
            "pod-a"
        ]
    return fetch_otel_collector_logs(core, "ns", tail_lines=None)


@pytest.mark.parametrize("reader", ["batch", "pods", "collector"])
@pytest.mark.parametrize(
    "log_text",
    [
        (
            '[provider:run] output: {"summary": "<tool_data source=\\"list_namespaces\\">'
            '\\ne2e-sentinel-ns-7f3a9\\n</tool_data>"}\n'
        ),
        "Starting\nUnicode π 🍵; apostrophe's; literal \\n and C:\\namespaces\\test\nFinished\n",
        "",
    ],
    ids=["json-escapes", "unicode-and-literal-backslashes", "empty"],
)
def test_native_log_readers_preserve_text(log_http: Any, reader: str, log_text: str) -> None:
    log_http.body = log_text.encode("utf-8")
    result = _read_logs(reader, log_http.core)
    prefix = "=== collector pod/pod-a uid=offline-uid ===\n" if reader == "collector" else ""
    assert result == prefix + log_text
    assert len(log_http.responses) == 1
    # Observe our ownership boundary before fixture teardown. urllib3 may close
    # its body automatically after reading; the raw response still needs closing.
    assert log_http.responses[0].close_calls == 1
    expected_query: dict[str, Any] = {"timestamps": reader == "collector"}
    if reader != "collector":
        expected_query["tailLines"] = 200 if reader == "batch" else 7
    assert log_http.queries == [expected_query]


@pytest.mark.parametrize("reader", ["batch", "pods", "collector"])
def test_native_log_readers_close_response_on_read_failure(log_http: Any, reader: str) -> None:
    log_http.read_error = True
    with pytest.raises(RuntimeError, match="offline log read failed"):
        _read_logs(reader, log_http.core)
    assert len(log_http.responses) == 1
    assert log_http.responses[0].close_calls == 1


@pytest.mark.parametrize("reader", ["batch", "pods", "collector"])
def test_native_log_readers_replace_invalid_utf8(log_http: Any, reader: str) -> None:
    log_http.body = b"Malformed byte: \xff\n"
    result = _read_logs(reader, log_http.core)
    prefix = "=== collector pod/pod-a uid=offline-uid ===\n" if reader == "collector" else ""
    assert result == prefix + "Malformed byte: \ufffd\n"
    assert log_http.responses[0].close_calls == 1


def test_native_batch_logs_preserve_mcp_sentinel(
    log_http: Any, caplog: pytest.LogCaptureFixture
) -> None:
    summary = (
        '<tool_data source="list_namespaces">\nNamespaces: e2e-sentinel-ns-7f3a9\n</tool_data>'
    )
    with caplog.at_level(logging.INFO, logger="lightspeed_agentic"):
        EventLogger("run").log(ResultEvent(text=json.dumps({"success": True, "summary": summary})))
    log_http.body = caplog.text.encode("utf-8")
    logs = _fetch_job_pod_logs(log_http.core, "ns", "offline-job")
    body = _enrich_body_from_pod_logs({"success": True, "summary": "Step completed"}, logs)
    assert body["success"] is True
    assert body["summary"] == summary
    assert "e2e-sentinel-ns-7f3a9" in body["summary"]


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
