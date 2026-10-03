"""Prometheus histograms for OTel GenAI semantic conventions.

Recorded in-process during agent runs. Batch entrypoint does not scrape or
export them — see audit-logging.md rule 19 (OTLP traces carry usage for ops).
"""

from __future__ import annotations

from prometheus_client import Histogram

TOKEN_BUCKETS = (
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
OPERATION_DURATION_BUCKETS = (
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

token_usage = Histogram(
    "gen_ai_client_token_usage",
    "Token usage distribution",
    ["gen_ai_token_type", "gen_ai_request_model", "gen_ai_provider_name", "gen_ai_operation_name"],
    buckets=TOKEN_BUCKETS,
)

operation_duration = Histogram(
    "gen_ai_client_operation_duration_seconds",
    "LLM operation duration",
    ["gen_ai_request_model", "gen_ai_provider_name", "gen_ai_operation_name", "error_type"],
    buckets=OPERATION_DURATION_BUCKETS,
)

tool_duration = Histogram(
    "gen_ai_execute_tool_duration_seconds",
    "Tool execution duration",
    ["gen_ai_tool_name"],
)
