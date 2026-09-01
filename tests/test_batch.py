"""Tests for the batch entrypoint."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

from lightspeed_agentic.batch import BatchInput, InputReadError
from lightspeed_agentic.config import ResolvedSDK
from lightspeed_agentic.mcp import MCPConfigError
from lightspeed_agentic.run_agent import AgentResult

_TEMPLATE = {
    "apiVersion": "agentic.openshift.io/v1alpha1",
    "kind": "AnalysisResult",
    "metadata": {"name": "run-analysis-1", "namespace": "openshift-lightspeed"},
    "spec": {"agenticRunName": "run"},
}

_INPUTS = BatchInput(
    query="diagnose the issue",
    output_schema={"type": "object"},
    context={"targetNamespaces": ["default"]},
    result_template=_TEMPLATE,
)

_MOCK_SDK = ResolvedSDK("deepagents", ("ANTHROPIC_API_KEY",))


class TestBatchMain:
    def test_input_read_failure_writes_termination_log_and_exits(self) -> None:
        with (
            patch(
                "lightspeed_agentic.batch.read_batch_inputs",
                side_effect=InputReadError("read /input/query: missing"),
            ),
            patch("lightspeed_agentic.batch.write_termination_log") as write_log,
            patch("lightspeed_agentic.batch.sys.exit") as exit_mock,
        ):
            from lightspeed_agentic.batch import main

            main()
            write_log.assert_called_once()
            exit_mock.assert_called_once_with(1)

    def test_success_publishes_and_exits_zero(self) -> None:
        agent_result = AgentResult(
            output={
                "success": True,
                "summary": "done",
                "options": [{"title": "fix"}],
                "actionRequired": True,
                "diagnosis": {"summary": "s", "rootCause": "r"},
            },
            input_tokens=500,
            output_tokens=200,
        )

        with (
            patch("lightspeed_agentic.batch.read_batch_inputs", return_value=_INPUTS),
            patch("lightspeed_agentic.batch.resolve_sdk", return_value=_MOCK_SDK),
            patch(
                "lightspeed_agentic.batch.run_readiness_checks",
                return_value=(True, {"provider_env": "ok"}),
            ),
            patch("lightspeed_agentic.batch.parse_reasoning_config", return_value=None),
            patch("lightspeed_agentic.batch.create_provider") as create_provider,
            patch("lightspeed_agentic.batch.resolve_router_model", return_value="test-model"),
            patch("lightspeed_agentic.batch.parse_mcp_servers", return_value=[]),
            patch("lightspeed_agentic.batch.run_agent_query", new_callable=AsyncMock) as run_query,
            patch("lightspeed_agentic.batch.publish_agent_result") as publish,
            patch("lightspeed_agentic.batch.otel_runtime_enabled", return_value=True),
            patch("lightspeed_agentic.batch.init_tracer") as init_tracer,
            patch("lightspeed_agentic.batch.shutdown_tracer"),
            patch("lightspeed_agentic.batch.sys.exit") as exit_mock,
        ):
            provider = create_provider.return_value
            provider.name = "deepagents"
            run_query.return_value = agent_result

            from lightspeed_agentic.batch import main

            main()

            publish.assert_called_once()
            assert publish.call_args.args[1] == agent_result.output
            publish_kwargs = publish.call_args.kwargs
            assert publish_kwargs["started_at"] is not None
            assert publish_kwargs["completed_at"] is not None
            assert publish_kwargs["input_tokens"] == 500
            assert publish_kwargs["output_tokens"] == 200
            init_tracer.assert_called_once_with(agenticrun_phase="analysis")
            assert run_query.call_args.kwargs["step"] == "analysis"
            exit_mock.assert_not_called()

    def test_capture_content_defaults_on_when_audit_enabled(self) -> None:
        """Unset LIGHTSPEED_CAPTURE_CONTENT captures content when audit is on."""
        with (
            patch.dict("os.environ", {"LIGHTSPEED_AUDIT_ENABLED": "true"}, clear=False),
            patch("lightspeed_agentic.batch.read_batch_inputs", return_value=_INPUTS),
            patch("lightspeed_agentic.batch.resolve_sdk", return_value=_MOCK_SDK),
            patch(
                "lightspeed_agentic.batch.run_readiness_checks",
                return_value=(True, {"provider_env": "ok"}),
            ),
            patch("lightspeed_agentic.batch.parse_reasoning_config", return_value=None),
            patch("lightspeed_agentic.batch.create_provider") as create_provider,
            patch("lightspeed_agentic.batch.resolve_router_model", return_value="test-model"),
            patch("lightspeed_agentic.batch.parse_mcp_servers", return_value=[]),
            patch("lightspeed_agentic.batch.run_agent_query", new_callable=AsyncMock) as run_query,
            patch("lightspeed_agentic.batch.publish_agent_result"),
            patch("lightspeed_agentic.batch.otel_runtime_enabled", return_value=False),
        ):
            provider = create_provider.return_value
            provider.name = "deepagents"
            run_query.return_value = AgentResult(
                output={"success": True, "summary": "done", "options": [], "actionRequired": False},
            )

            from lightspeed_agentic.batch import main

            main()

            assert run_query.call_args.kwargs["capture_content"] is True

    def test_capture_content_opt_out_when_audit_enabled(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"LIGHTSPEED_AUDIT_ENABLED": "true", "LIGHTSPEED_CAPTURE_CONTENT": "false"},
                clear=False,
            ),
            patch("lightspeed_agentic.batch.read_batch_inputs", return_value=_INPUTS),
            patch("lightspeed_agentic.batch.resolve_sdk", return_value=_MOCK_SDK),
            patch(
                "lightspeed_agentic.batch.run_readiness_checks",
                return_value=(True, {"provider_env": "ok"}),
            ),
            patch("lightspeed_agentic.batch.parse_reasoning_config", return_value=None),
            patch("lightspeed_agentic.batch.create_provider") as create_provider,
            patch("lightspeed_agentic.batch.resolve_router_model", return_value="test-model"),
            patch("lightspeed_agentic.batch.parse_mcp_servers", return_value=[]),
            patch("lightspeed_agentic.batch.run_agent_query", new_callable=AsyncMock) as run_query,
            patch("lightspeed_agentic.batch.publish_agent_result"),
            patch("lightspeed_agentic.batch.otel_runtime_enabled", return_value=False),
        ):
            provider = create_provider.return_value
            provider.name = "deepagents"
            run_query.return_value = AgentResult(
                output={"success": True, "summary": "done"},
            )

            from lightspeed_agentic.batch import main

            main()

            assert run_query.call_args.kwargs["capture_content"] is False

    def test_passes_traceparent_env_to_run_agent_query(self) -> None:
        """TRACEPARENT env from operator pod spec is forwarded to run_agent_query."""
        traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

        with (
            patch.dict("os.environ", {"TRACEPARENT": traceparent}, clear=False),
            patch("lightspeed_agentic.batch.read_batch_inputs", return_value=_INPUTS),
            patch("lightspeed_agentic.batch.resolve_sdk", return_value=_MOCK_SDK),
            patch(
                "lightspeed_agentic.batch.run_readiness_checks",
                return_value=(True, {"provider_env": "ok"}),
            ),
            patch("lightspeed_agentic.batch.parse_reasoning_config", return_value=None),
            patch("lightspeed_agentic.batch.create_provider") as create_provider,
            patch("lightspeed_agentic.batch.resolve_router_model", return_value="test-model"),
            patch("lightspeed_agentic.batch.parse_mcp_servers", return_value=[]),
            patch("lightspeed_agentic.batch.run_agent_query", new_callable=AsyncMock) as run_query,
            patch("lightspeed_agentic.batch.publish_agent_result"),
            patch("lightspeed_agentic.batch.otel_runtime_enabled", return_value=False),
            patch("lightspeed_agentic.batch.sys.exit") as exit_mock,
        ):
            provider = create_provider.return_value
            provider.name = "deepagents"
            run_query.return_value = AgentResult(
                output={
                    "success": True,
                    "summary": "done",
                    "options": [{"title": "fix"}],
                    "actionRequired": True,
                    "diagnosis": {"summary": "s", "rootCause": "r"},
                },
            )

            from lightspeed_agentic.batch import main

            main()

            assert run_query.call_args.kwargs["traceparent"] == traceparent
            exit_mock.assert_not_called()

    def test_readiness_failure_writes_termination_log_and_exits(self) -> None:
        checks = {
            "provider_env": "error: missing ANTHROPIC_API_KEY",
        }
        with (
            patch("lightspeed_agentic.batch.read_batch_inputs", return_value=_INPUTS),
            patch("lightspeed_agentic.batch.resolve_sdk", return_value=_MOCK_SDK),
            patch("lightspeed_agentic.batch.parse_reasoning_config", return_value=None),
            patch("lightspeed_agentic.batch.run_readiness_checks", return_value=(False, checks)),
            patch("lightspeed_agentic.batch.write_termination_log") as write_log,
            patch("lightspeed_agentic.batch.init_tracer") as init_tracer,
            patch("lightspeed_agentic.batch.shutdown_tracer"),
            patch("lightspeed_agentic.batch.sys.exit") as exit_mock,
        ):
            from lightspeed_agentic.batch import main

            main()

            init_tracer.assert_not_called()
            write_log.assert_called_once_with(
                "readiness failed: provider_env=error: missing ANTHROPIC_API_KEY"
            )
            exit_mock.assert_called_once_with(1)

    def test_reasoning_config_failure_before_tracer_writes_termination_log(self) -> None:
        with (
            patch("lightspeed_agentic.batch.read_batch_inputs", return_value=_INPUTS),
            patch("lightspeed_agentic.batch.resolve_sdk", return_value=_MOCK_SDK),
            patch(
                "lightspeed_agentic.batch.parse_reasoning_config",
                side_effect=ValueError("LIGHTSPEED_REASONING_CONFIG contains invalid JSON"),
            ),
            patch("lightspeed_agentic.batch.write_termination_log") as write_log,
            patch("lightspeed_agentic.batch.init_tracer") as init_tracer,
            patch("lightspeed_agentic.batch.shutdown_tracer"),
            patch("lightspeed_agentic.batch.sys.exit") as exit_mock,
        ):
            from lightspeed_agentic.batch import main

            main()

            init_tracer.assert_not_called()
            write_log.assert_called_once_with("LIGHTSPEED_REASONING_CONFIG contains invalid JSON")
            exit_mock.assert_called_once_with(1)

    def test_skips_otel_when_unconfigured(self) -> None:
        with (
            patch("lightspeed_agentic.batch.read_batch_inputs", return_value=_INPUTS),
            patch("lightspeed_agentic.batch.resolve_sdk", return_value=_MOCK_SDK),
            patch(
                "lightspeed_agentic.batch.run_readiness_checks",
                return_value=(True, {"provider_env": "ok"}),
            ),
            patch("lightspeed_agentic.batch.parse_reasoning_config", return_value=None),
            patch("lightspeed_agentic.batch.parse_mcp_servers", return_value=[]),
            patch("lightspeed_agentic.batch.otel_runtime_enabled", return_value=False),
            patch("lightspeed_agentic.batch.create_provider") as create_provider,
            patch("lightspeed_agentic.batch.resolve_router_model", return_value="test-model"),
            patch("lightspeed_agentic.batch.run_agent_query", new_callable=AsyncMock) as run_query,
            patch("lightspeed_agentic.batch.publish_agent_result"),
            patch("lightspeed_agentic.batch.init_tracer") as init_tracer,
            patch("lightspeed_agentic.batch.shutdown_tracer") as shutdown_tracer,
        ):
            provider = create_provider.return_value
            provider.name = "deepagents"
            run_query.return_value = AgentResult(
                output={
                    "success": True,
                    "summary": "done",
                    "options": [{"title": "fix"}],
                    "actionRequired": True,
                    "diagnosis": {"summary": "s", "rootCause": "r"},
                },
            )

            from lightspeed_agentic.batch import main

            main()

            init_tracer.assert_not_called()
            shutdown_tracer.assert_not_called()

    def test_mcp_config_failure_before_tracer_writes_termination_log(self) -> None:
        with (
            patch("lightspeed_agentic.batch.read_batch_inputs", return_value=_INPUTS),
            patch("lightspeed_agentic.batch.resolve_sdk", return_value=_MOCK_SDK),
            patch("lightspeed_agentic.batch.parse_reasoning_config", return_value=None),
            patch(
                "lightspeed_agentic.batch.parse_mcp_servers",
                side_effect=MCPConfigError("LIGHTSPEED_MCP_SERVERS must be a JSON array"),
            ),
            patch("lightspeed_agentic.batch.write_termination_log") as write_log,
            patch("lightspeed_agentic.batch.init_tracer") as init_tracer,
            patch("lightspeed_agentic.batch.shutdown_tracer"),
            patch("lightspeed_agentic.batch.sys.exit") as exit_mock,
        ):
            from lightspeed_agentic.batch import main

            main()

            init_tracer.assert_not_called()
            write_log.assert_called_once_with("LIGHTSPEED_MCP_SERVERS must be a JSON array")
            exit_mock.assert_called_once_with(1)

    def test_invalid_mcp_entry_writes_termination_log(self) -> None:
        servers_json = json.dumps([{"url": "http://missing-name:8080/mcp"}])
        with (
            patch.dict(os.environ, {"LIGHTSPEED_MCP_SERVERS": servers_json}, clear=False),
            patch("lightspeed_agentic.batch.read_batch_inputs", return_value=_INPUTS),
            patch("lightspeed_agentic.batch.resolve_sdk", return_value=_MOCK_SDK),
            patch("lightspeed_agentic.batch.parse_reasoning_config", return_value=None),
            patch("lightspeed_agentic.batch.write_termination_log") as write_log,
            patch("lightspeed_agentic.batch.init_tracer") as init_tracer,
            patch("lightspeed_agentic.batch.shutdown_tracer"),
            patch("lightspeed_agentic.batch.sys.exit") as exit_mock,
        ):
            from lightspeed_agentic.batch import main

            main()

            init_tracer.assert_not_called()
            write_log.assert_called_once()
            assert "missing or invalid name" in write_log.call_args.args[0]
            exit_mock.assert_called_once_with(1)

    def test_publish_failure_writes_termination_log(self) -> None:
        from lightspeed_agentic.publish_results.publish import PublishError

        with (
            patch("lightspeed_agentic.batch.read_batch_inputs", return_value=_INPUTS),
            patch("lightspeed_agentic.batch.resolve_sdk", return_value=_MOCK_SDK),
            patch(
                "lightspeed_agentic.batch.run_readiness_checks",
                return_value=(True, {"provider_env": "ok"}),
            ),
            patch("lightspeed_agentic.batch.parse_reasoning_config", return_value=None),
            patch("lightspeed_agentic.batch.create_provider") as create_provider,
            patch("lightspeed_agentic.batch.resolve_router_model", return_value="test-model"),
            patch("lightspeed_agentic.batch.parse_mcp_servers", return_value=[]),
            patch(
                "lightspeed_agentic.batch.run_agent_query",
                new_callable=AsyncMock,
                return_value=AgentResult(output={"success": True, "summary": "ok"}),
            ),
            patch(
                "lightspeed_agentic.batch.publish_agent_result",
                side_effect=PublishError("create failed"),
            ),
            patch("lightspeed_agentic.batch.write_termination_log") as write_log,
            patch("lightspeed_agentic.batch.otel_runtime_enabled", return_value=True),
            patch("lightspeed_agentic.batch.init_tracer"),
            patch("lightspeed_agentic.batch.shutdown_tracer"),
            patch("lightspeed_agentic.batch.sys.exit") as exit_mock,
        ):
            create_provider.return_value.name = "deepagents"

            from lightspeed_agentic.batch import main

            main()

            write_log.assert_called_once_with("create failed")
            exit_mock.assert_called_once_with(1)
