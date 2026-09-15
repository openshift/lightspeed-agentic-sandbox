"""OpenAI provider — wraps openai-agents SDK.

Uses SandboxAgent with native Shell, Filesystem, and Skills capabilities.
The SDK handles tool registration, skill discovery, and command execution.
"""
# mypy: disable-error-code=unused-ignore

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:

    class AgentOutputSchemaBase:
        """Type-checker stub for the optional openai-agents base class."""

        pass

else:
    try:
        from agents.agent_output import AgentOutputSchemaBase
    except ImportError:

        class AgentOutputSchemaBase:  # pragma: no cover - optional SDK fallback
            """Fallback base so the module imports without the openai extra."""

            pass


from lightspeed_agentic.skills import has_skills
from lightspeed_agentic.types import (
    AgentProvider,
    ContentBlockStopEvent,
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    stringify,
)

logger = logging.getLogger(__name__)


_OPENAI_HOSTS = ("api.openai.com",)

# Models that support json_schema response format (structured output with strict mode)
_MODELS_WITH_JSON_SCHEMA = {
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
}


def _is_native_openai() -> bool:
    """True when talking to api.openai.com (explicitly or by default)."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        return True
    try:
        from urllib.parse import urlparse

        return urlparse(base_url).hostname in _OPENAI_HOSTS
    except Exception:
        return False


def _model_supports_json_schema(model: str) -> bool:
    """Check if a model supports json_schema response format (structured output).

    Args:
        model: Model identifier (e.g., 'gpt-4o', 'gpt-4-turbo')

    Returns:
        True if model supports json_schema, False otherwise.
        For custom endpoints (non-OpenAI), returns True (assume compatibility).
    """
    if not _is_native_openai():
        # Custom endpoints (vLLM, etc.) - assume they support json_schema
        # if user is requesting it. Failures will be caught at API call time.
        return True
    # For native OpenAI, check against known compatible models
    return any(model.startswith(m) for m in _MODELS_WITH_JSON_SCHEMA)


_openai_initialized = False


class _RawJsonSchema(AgentOutputSchemaBase):
    """Wraps JSON schema for OpenAI model output type.

    Args:
        schema: The JSON schema dict for structured output.
        is_native: True if using native OpenAI Responses API (strict mode),
                   False if using Chat Completions (non-strict mode).
    """

    def __init__(self, schema: dict[str, Any], is_native: bool) -> None:
        self._schema = schema
        self._is_native = is_native

    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return "raw_json_schema"

    def json_schema(self) -> dict[str, Any]:
        return self._schema

    def is_strict_json_schema(self) -> bool:
        return self._is_native

    def validate_json(self, json_str: str) -> Any:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in structured output: {e}") from e


def _patch_exec_command_args() -> None:
    """Sanitize ExecCommandArgs input before Pydantic validation.

    The openai-agents SDK registers exec_command with strict_json_schema=False,
    so the model can send "shell": true (boolean) instead of a string path.
    Pydantic rejects the bool, crashing the execution step. OLS-3257.
    Remove when openai-agents fixes exec_command schema validation.
    """
    from agents.sandbox.capabilities.tools.shell_tool import ExecCommandTool

    _original_invoke = ExecCommandTool._invoke

    async def _sanitized_invoke(self: ExecCommandTool, ctx: object, raw_input: str) -> str:
        try:
            parsed = json.loads(raw_input)
        except (json.JSONDecodeError, ValueError):
            # If parsing fails, pass through to original handler
            return await _original_invoke(self, ctx, raw_input)
        if isinstance(parsed, dict) and isinstance(parsed.get("shell"), bool):
            logger.debug("Coercing exec_command shell=%s to None (OLS-3257)", parsed["shell"])
            parsed["shell"] = None
            raw_input = json.dumps(parsed)
        return await _original_invoke(self, ctx, raw_input)

    ExecCommandTool._invoke = _sanitized_invoke  # type: ignore[assignment]


def _ensure_openai_init() -> None:
    global _openai_initialized
    if _openai_initialized:
        return
    from agents import enable_verbose_stdout_logging
    from agents.tracing import set_tracing_disabled

    set_tracing_disabled(True)
    enable_verbose_stdout_logging()  # type: ignore[no-untyped-call]
    _patch_exec_command_args()
    _openai_initialized = True


def _validated_e2e_output_dir() -> str | None:
    """Return E2E_OUTPUT_DIR when it resolves under the system temp directory."""
    raw = os.environ.get("E2E_OUTPUT_DIR", "").strip()
    if not raw:
        return None
    try:
        resolved = Path(raw).resolve()
    except OSError:
        logger.warning("E2E_OUTPUT_DIR is not a valid path: %s", raw)
        return None
    temp_root = Path(tempfile.gettempdir()).resolve()
    if resolved != temp_root and not str(resolved).startswith(str(temp_root) + os.sep):
        logger.warning(
            "E2E_OUTPUT_DIR outside temp root %s: %s",
            temp_root,
            resolved,
        )
        return None
    return str(resolved)


def _build_manifest(cwd: str) -> Any:
    """Build sandbox manifest, optionally granting write access to E2E_OUTPUT_DIR."""
    from agents.sandbox.manifest import Manifest, SandboxPathGrant  # type: ignore[attr-defined]

    kwargs: dict[str, Any] = {"root": cwd}
    output_dir = _validated_e2e_output_dir()
    if output_dir:
        kwargs["extra_path_grants"] = (
            SandboxPathGrant(
                path=output_dir,
                read_only=False,
                description="e2e skill token output",
            ),
        )
    return Manifest(**kwargs)


class OpenAIProvider(AgentProvider):
    _client: Any = None

    @property
    def name(self) -> str:
        return "openai"

    def _build_model_settings(self, reasoning_config: dict[str, Any]) -> Any:
        """Build ModelSettings from reasoning config.

        Helper to avoid code duplication between single and two-phase paths.
        """
        from agents.model_settings import ModelSettings
        from openai.types.shared import Reasoning

        rc = dict(reasoning_config)
        model_settings_kwargs: dict[str, Any] = {}
        if "verbosity" in rc:
            model_settings_kwargs["verbosity"] = rc.pop("verbosity")
        if rc:
            model_settings_kwargs["reasoning"] = Reasoning(**rc)
        if model_settings_kwargs:
            return ModelSettings(**model_settings_kwargs)
        return None

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        """Execute agent query using OpenAI Responses or Chat Completions model.

        For native OpenAI: Uses OpenAIResponsesModel with full capabilities.
        For vLLM/custom endpoints: Uses OpenAIChatCompletionsModel with manually
        implemented filesystem function tools (ChatCompletions compatible).
        """
        _ensure_openai_init()

        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=os.environ.get("OPENAI_BASE_URL"),
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            )

        from agents import (
            RawResponsesStreamEvent,
            RunItemStreamEvent,
            Runner,
        )
        from agents.items import ToolCallItem, ToolCallOutputItem
        from agents.run_config import RunConfig, SandboxRunConfig
        from agents.sandbox import SandboxAgent
        from agents.sandbox.capabilities import Filesystem, Shell, Skills
        from agents.sandbox.capabilities.skills import LocalDirLazySkillSource
        from agents.sandbox.entries import LocalDir
        from agents.sandbox.sandboxes.unix_local import (
            UnixLocalSandboxClient,
        )
        from openai.types.responses import (
            ResponseReasoningSummaryTextDeltaEvent,
            ResponseReasoningTextDeltaEvent,
            ResponseTextDeltaEvent,
        )

        # Setup model and capabilities based on endpoint
        is_native = _is_native_openai()
        capabilities: list[Any] = [Shell()]
        function_tools_list: list[Any] | None = None

        if is_native:
            from agents.models.openai_responses import OpenAIResponsesModel

            model: Any = OpenAIResponsesModel(model=options.model, openai_client=self._client)
            # Native OpenAI: use full Filesystem() capability
            capabilities.append(Filesystem())
        else:
            from agents.models.openai_chatcompletions import (
                OpenAIChatCompletionsModel,
            )

            from lightspeed_agentic.function_tools import (
                apply_patch,
                list_directory,
                read_file,
                write_file,
            )

            model = OpenAIChatCompletionsModel(
                model=options.model,
                openai_client=self._client,
                buffer_streamed_tool_calls=True,
            )
            # vLLM/custom: use manually implemented filesystem function tools
            # (avoids incompatible CustomTool apply_patch in Filesystem)
            function_tools_list = [read_file, write_file, list_directory, apply_patch]

        # Add Skills if present
        if has_skills(options.cwd):
            capabilities.append(
                Skills(
                    lazy_from=LocalDirLazySkillSource(
                        source=LocalDir(src=Path(options.cwd)),
                    ),
                    skills_path="skills/.agents",
                ),
            )

        # Manifest root is cwd's parent (/app) so shell commands can reach workspace
        manifest = _build_manifest(str(Path(options.cwd).parent))

        # Setup MCP servers (gracefully degrade if unavailable)
        mcp_manager = None
        if options.mcp_servers:
            try:
                from agents.mcp import MCPServerManager

                from lightspeed_agentic.mcp import to_openai_mcp_servers

                mcp_servers_list = to_openai_mcp_servers(options.mcp_servers)
                if not mcp_servers_list:
                    logger.warning("MCP servers configured but conversion produced no servers")
                else:
                    mcp_manager = MCPServerManager(mcp_servers_list)
                    await mcp_manager.__aenter__()
                    # Validate manager initialized properly
                    if not hasattr(mcp_manager, "active_servers"):
                        logger.warning("MCPServerManager missing active_servers attribute")
                        mcp_manager = None
                    else:
                        active_servers = mcp_manager.active_servers
                        active_count = len(active_servers) if active_servers else 0
                        if active_count == 0:
                            logger.warning("MCPServerManager initialized but no active servers")
                        else:
                            # Add MCP servers to capabilities
                            capabilities.extend(mcp_manager.active_servers)
                            logger.debug(f"Initialized {active_count} MCP servers")
            except Exception as e:
                logger.warning(
                    "Failed to initialize MCP servers, continuing without them: "
                    f"{type(e).__name__}: {e}"
                )
                mcp_manager = None

        try:
            agent_kwargs: dict[str, Any] = {
                "name": "lightspeed",
                "instructions": options.system_prompt,
                "model": model,
                "capabilities": capabilities,
                "default_manifest": manifest,
            }

            # Add function tools for vLLM/custom endpoints
            if function_tools_list:
                agent_kwargs["tools"] = function_tools_list

            if options.reasoning_config:
                agent_kwargs["model_settings"] = self._build_model_settings(
                    options.reasoning_config
                )

            # Validate model supports structured output
            if options.output_schema and not _model_supports_json_schema(options.model):
                raise ValueError(
                    f"Model {options.model} does not support json_schema response format. "
                    "Only models with structured output support (gpt-4, gpt-4o, etc.) are "
                    "compatible."
                )

            # Set output_type for structured output
            if options.output_schema:
                agent_kwargs["output_type"] = _RawJsonSchema(
                    options.output_schema, is_native=is_native
                )

            agent = SandboxAgent(**agent_kwargs)

            run_config = RunConfig(
                sandbox=SandboxRunConfig(
                    client=UnixLocalSandboxClient(),
                ),
            )

            result = Runner.run_streamed(
                agent,
                options.prompt,
                max_turns=options.max_turns,
                run_config=run_config,
            )

            # Stream events from the runner
            async for event in result.stream_events():
                # Handle text and reasoning deltas from both Responses and ChatCompletions.
                # Both models emit ResponseTextDeltaEvent and ResponseReasoningTextDeltaEvent
                # (ChatCompletions converts its reasoning_content to Responses event types).
                if isinstance(event, RawResponsesStreamEvent):
                    if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                        yield TextDeltaEvent(text=event.data.delta)
                    elif (
                        isinstance(
                            event.data,
                            (
                                ResponseReasoningTextDeltaEvent,
                                ResponseReasoningSummaryTextDeltaEvent,
                            ),
                        )
                        and event.data.delta
                    ):
                        yield ThinkingDeltaEvent(thinking=event.data.delta)

                # Handle tool call and result events (both Responses and ChatCompletions)
                elif isinstance(event, RunItemStreamEvent):
                    if isinstance(event.item, ToolCallItem):
                        raw = event.item.raw_item
                        name = (
                            getattr(raw, "name", None)
                            or (raw.get("name") if isinstance(raw, dict) else "")
                            or ""
                        )
                        args = getattr(raw, "arguments", None) or ""
                        yield ToolCallEvent(
                            name=name,
                            input=args,
                            call_id=getattr(event.item, "call_id", "") or "",
                        )
                    elif isinstance(event.item, ToolCallOutputItem):
                        full_output = stringify(event.item.output)
                        yield ToolResultEvent(
                            output=full_output,
                            call_id=getattr(event.item, "call_id", "") or "",
                        )

            yield ContentBlockStopEvent()

            usage = result.context_wrapper.usage
            resp_model = getattr(result.context_wrapper, "model", "") or options.model
            details = getattr(usage, "output_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", 0) if details else 0

            final_text = stringify(result.final_output)

            yield ResultEvent(
                text=final_text,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=reasoning,
                response_model=resp_model,
            )
        finally:
            if mcp_manager:
                await mcp_manager.__aexit__(None, None, None)
