"""Tests for per-request GenAI and actual tool lifecycle metrics."""

from __future__ import annotations

from typing import Any

import pytest
from prometheus_client import REGISTRY

from lightspeed_agentic.audit import AuditLogger
from lightspeed_agentic.metrics import operation_duration, token_usage


def _sample(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _bucket_bounds(metric_name: str, labels: dict[str, str]) -> tuple[float, ...]:
    family = next(metric for metric in REGISTRY.collect() if metric.name == metric_name)
    bounds = (
        float(sample.labels["le"])
        for sample in family.samples
        if sample.name == f"{metric_name}_bucket"
        and sample.labels["le"] != "+Inf"
        and all(sample.labels.get(key) == value for key, value in labels.items())
    )
    return tuple(sorted(bounds))


def _recorder(model: str = "metrics-model") -> AuditLogger:
    return AuditLogger(phase="analysis", model=model, provider="metrics-provider")


def test_semconv_histogram_boundaries_are_exposed_to_consumers() -> None:
    token_labels = {
        "gen_ai_token_type": "input",
        "gen_ai_request_model": "metrics-boundary-probe",
        "gen_ai_provider_name": "metrics-boundary-probe",
        "gen_ai_operation_name": "boundary_probe",
    }
    duration_labels = {
        "gen_ai_request_model": "metrics-boundary-probe",
        "gen_ai_provider_name": "metrics-boundary-probe",
        "gen_ai_operation_name": "boundary_probe",
        "error_type": "",
    }
    token_usage.labels(**token_labels)
    operation_duration.labels(**duration_labels)

    assert _bucket_bounds("gen_ai_client_token_usage", token_labels) == (
        1,
        4,
        16,
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        16777216,
        67108864,
    )
    assert _bucket_bounds("gen_ai_client_operation_duration_seconds", duration_labels) == (
        0.01,
        0.02,
        0.04,
        0.08,
        0.16,
        0.32,
        0.64,
        1.28,
        2.56,
        5.12,
        10.24,
        20.48,
        40.96,
        81.92,
    )


def test_inference_metrics_are_recorded_per_request_with_observed_zero_tokens() -> None:
    labels_input = {
        "gen_ai_token_type": "input",
        "gen_ai_request_model": "metrics-model",
        "gen_ai_provider_name": "metrics-provider",
        "gen_ai_operation_name": "chat",
    }
    labels_output = {**labels_input, "gen_ai_token_type": "output"}
    labels_duration = {
        "gen_ai_request_model": "metrics-model",
        "gen_ai_provider_name": "metrics-provider",
        "gen_ai_operation_name": "chat",
        "error_type": "",
    }
    input_count = _sample("gen_ai_client_token_usage_count", labels_input)
    input_sum = _sample("gen_ai_client_token_usage_sum", labels_input)
    output_count = _sample("gen_ai_client_token_usage_count", labels_output)
    output_sum = _sample("gen_ai_client_token_usage_sum", labels_output)
    duration_count = _sample("gen_ai_client_operation_duration_seconds_count", labels_duration)
    duration_sum = _sample("gen_ai_client_operation_duration_seconds_sum", labels_duration)
    start_time = 1_000_000_000

    recorder = _recorder()
    span = recorder.start_inference(
        model="metrics-model",
        operation="chat",
        input_messages=[],
        start_time=start_time,
    )
    recorder.end_inference(
        span,
        input_tokens=0,
        output_tokens=5,
        end_time=start_time + 250_000_000,
    )

    assert _sample("gen_ai_client_token_usage_count", labels_input) == input_count + 1
    assert _sample("gen_ai_client_token_usage_sum", labels_input) == input_sum
    assert _sample("gen_ai_client_token_usage_count", labels_output) == output_count + 1
    assert _sample("gen_ai_client_token_usage_sum", labels_output) == output_sum + 5
    assert (
        _sample("gen_ai_client_operation_duration_seconds_count", labels_duration)
        == duration_count + 1
    )
    assert (
        _sample("gen_ai_client_operation_duration_seconds_sum", labels_duration) - duration_sum
    ) == pytest.approx(0.25)


def test_unavailable_token_counts_are_not_recorded() -> None:
    labels = {
        "gen_ai_token_type": "input",
        "gen_ai_request_model": "metrics-model",
        "gen_ai_provider_name": "metrics-provider",
        "gen_ai_operation_name": "chat",
    }
    before = _sample("gen_ai_client_token_usage_count", labels)

    recorder = _recorder()
    span = recorder.start_inference(
        model="metrics-model",
        operation="chat",
        input_messages=None,
    )
    recorder.end_inference(span, input_tokens=None, output_tokens=None)

    assert _sample("gen_ai_client_token_usage_count", labels) == before


def test_failed_inference_duration_has_error_type_label() -> None:
    labels = {
        "gen_ai_request_model": "metrics-model",
        "gen_ai_provider_name": "metrics-provider",
        "gen_ai_operation_name": "generate_content",
        "error_type": "TimeoutError",
    }
    before = _sample("gen_ai_client_operation_duration_seconds_count", labels)
    before_sum = _sample("gen_ai_client_operation_duration_seconds_sum", labels)
    start_time = 2_000_000_000

    recorder = _recorder()
    span = recorder.start_inference(
        model="metrics-model",
        operation="generate_content",
        input_messages=None,
        start_time=start_time,
    )
    recorder.end_inference(
        span,
        error=TimeoutError("private message"),
        end_time=start_time + 125_000_000,
    )

    assert _sample("gen_ai_client_operation_duration_seconds_count", labels) == before + 1
    assert (
        _sample("gen_ai_client_operation_duration_seconds_sum", labels) - before_sum
    ) == pytest.approx(0.125)


def test_tool_duration_uses_actual_tool_lifecycle() -> None:
    labels = {"gen_ai_tool_name": "execute"}
    before_count = _sample("gen_ai_execute_tool_duration_seconds_count", labels)
    before_sum = _sample("gen_ai_execute_tool_duration_seconds_sum", labels)

    recorder = _recorder()
    start_time = 3_000_000_000
    span = recorder.start_tool(
        name="execute",
        call_id="call-1",
        arguments={"command": "true"},
        start_time=start_time,
    )
    recorder.end_tool(span, result={"exit_code": 0}, end_time=start_time + 500_000_000)

    assert _sample("gen_ai_execute_tool_duration_seconds_count", labels) == before_count + 1
    assert (
        _sample("gen_ai_execute_tool_duration_seconds_sum", labels) - before_sum
    ) == pytest.approx(0.5)


@pytest.mark.parametrize("finish", ["end_inference", "close"])
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("private diagnostic"), "_OTHER"),
        ("arbitrary private error text", "_OTHER"),
        (TimeoutError(), "TimeoutError"),
        ("response.failed", "response.failed"),
        ("operation_cancelled", "operation_cancelled"),
    ],
)
def test_inference_error_metrics_use_bounded_categories(
    finish: str, error: BaseException | str, expected: str, span_exporter: Any
) -> None:
    model = f"bounded-errors-{finish}"
    labels = {
        "gen_ai_request_model": model,
        "gen_ai_provider_name": "metrics-provider",
        "gen_ai_operation_name": "chat",
        "error_type": expected,
    }
    before = _sample("gen_ai_client_operation_duration_seconds_count", labels)
    recorder = _recorder(model)
    span = recorder.start_inference(model=model, operation="chat", input_messages=None)
    if finish == "end_inference":
        recorder.end_inference(span, error=error)
    else:
        recorder.close(error)

    assert _sample("gen_ai_client_operation_duration_seconds_count", labels) == before + 1
    raw_type = error if isinstance(error, str) else type(error).__name__
    exported = next(
        exported
        for exported in span_exporter.get_finished_spans()
        if exported.context == span.get_span_context()
    )
    assert exported.attributes["error.type"] == raw_type
    if raw_type != expected:
        assert (
            REGISTRY.get_sample_value(
                "gen_ai_client_operation_duration_seconds_count",
                {**labels, "error_type": raw_type},
            )
            is None
        )
