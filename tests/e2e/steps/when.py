"""When steps — batch Job runs against the cluster."""

from __future__ import annotations

from typing import Any

from pytest_bdd import when

from tests.e2e.run_result import E2ERunResult, store_run_result


def _run_query(bdd_context: dict[str, Any], run_runner: Any, **kwargs: Any) -> None:
    if bdd_context.get("mount_skills", False):
        kwargs["mount_skills"] = True
    res: E2ERunResult = run_runner(bdd_context["query"], **kwargs)
    store_run_result(bdd_context, res)


@when("I run the agent with a simple reasoning query")
def post_simple_reasoning(bdd_context: dict[str, Any], run_runner: Any) -> None:
    bdd_context["query"] = "What is 17 * 23? Reply with just the number."
    _run_query(bdd_context, run_runner)


@when("I run the agent with the prepared echo-token query")
def post_echo_token_query(bdd_context: dict[str, Any], run_runner: Any, provider_name: str) -> None:
    _ = provider_name
    kwargs: dict[str, Any] = {"output_schema": bdd_context.get("output_schema")}
    if "system_prompt" in bdd_context:
        kwargs["system_prompt"] = bdd_context["system_prompt"]
    _run_query(bdd_context, run_runner, **kwargs)


@when("I run the agent with the prepared find-token analysis query")
def post_find_token_analysis_query(
    bdd_context: dict[str, Any],
    run_runner: Any,
    provider_name: str,
) -> None:
    _ = provider_name
    kwargs: dict[str, Any] = {
        "output_schema": bdd_context.get("output_schema"),
        "system_prompt": bdd_context["system_prompt"],
    }
    wait_timeout = bdd_context.get("wait_timeout_seconds")
    if wait_timeout is not None:
        kwargs["wait_timeout_seconds"] = wait_timeout
    _run_query(bdd_context, run_runner, **kwargs)


@when("I run the agent with the prepared schema and query")
def post_with_schema(bdd_context: dict[str, Any], run_runner: Any) -> None:
    kwargs: dict[str, Any] = {"output_schema": bdd_context.get("output_schema")}
    if "system_prompt" in bdd_context:
        kwargs["system_prompt"] = bdd_context["system_prompt"]
    _run_query(bdd_context, run_runner, **kwargs)


@when("I run the agent with the prepared query and no output schema")
def post_without_schema(bdd_context: dict[str, Any], run_runner: Any) -> None:
    _run_query(bdd_context, run_runner, output_schema=None)


@when("I run the agent with the prepared context and schema")
def post_with_context_and_schema(bdd_context: dict[str, Any], run_runner: Any) -> None:
    _run_query(
        bdd_context,
        run_runner,
        output_schema=bdd_context["output_schema"],
        context=bdd_context["context"],
    )
