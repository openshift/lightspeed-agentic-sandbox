"""Evaluate labeled tool results inside the tested image, not through an agent.

This file is mounted unchanged into a sandbox Job. Do not import host test helpers.
Only framed, payload-free records are written to stdout. Misclassifications are
measurements; operational failures, not an invented accuracy threshold, affect exit.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lightspeed_agentic.config import resolve_model_family
from lightspeed_agentic.inspection.chunking import Utf8ByteCodec
from lightspeed_agentic.inspection.client import LangChainClassifierClient
from lightspeed_agentic.inspection.errors import InspectionError
from lightspeed_agentic.inspection.inspector import inspect_tool_result
from lightspeed_agentic.inspection.models import ClassifierCategory
from lightspeed_agentic.providers.deepagents import _close_model_clients
from lightspeed_agentic.providers.model_router import resolve_model_route
from lightspeed_agentic.readiness import run_readiness_checks
from lightspeed_agentic.tls import configure_tls

PREFIX = "INSPECTION_EVAL "
SafeName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
Failure = Literal[
    "timeout",
    "provider_error",
    "invalid_response",
    "size_limit",
    "runner_error",
    "setup_error",
    "cleanup_error",
    "missing_result",
    "protocol_error",
]


class InspectionCase(BaseModel):
    """A labeled fixture; labels are never included in classifier requests."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    case_id: SafeName
    tool_name: SafeName
    result_type: Literal["result", "error"]
    content: str
    expected_outcome: Literal["benign", "malicious"]
    expected_category: ClassifierCategory
    tags: list[SafeName] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_label(self) -> InspectionCase:
        if (self.expected_outcome == "benign") != (self.expected_category == "none"):
            raise ValueError("inconsistent corpus label")
        return self


class CaseResult(BaseModel):
    """Allowlisted observations, with operational errors distinct from decisions."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    case_id: SafeName
    expected_outcome: Literal["benign", "malicious"]
    expected_category: ClassifierCategory | None = None
    tags: list[SafeName] = Field(default_factory=list)
    observed_outcome: Literal["benign", "malicious", "error", "unverified"]
    observed_category: ClassifierCategory | None = None
    failure_type: Failure | None = None
    attempts: int = Field(default=0, ge=0)
    chunks_total: int | None = Field(default=None, ge=1)
    chunks_visited: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0, ge=0)


class CorpusSource(BaseModel):
    """Identity of the unchanged upstream snapshot."""

    model_config = ConfigDict(extra="forbid", strict=True)
    source_repository: Literal["https://github.com/openshift/lightspeed-service"]
    source_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    source_path: Literal["eval/tool_result_inspection/corpus_full.yaml"]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    case_count: int = Field(gt=0)


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate keys rather than silently changing corpus labels/content."""

    def construct_mapping(self, node: Any, deep: bool = False) -> Any:
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate corpus key")
        return super().construct_mapping(node, deep=deep)


def load_corpus(path: Path, source_path: Path) -> tuple[list[InspectionCase], dict[str, Any]]:
    """Validate content/hash/labels without including input in exception messages."""
    try:
        raw = path.read_bytes()
        source = CorpusSource.model_validate(json.loads(source_path.read_text(encoding="utf-8")))
        document = yaml.load(raw.decode("utf-8"), Loader=UniqueLoader)  # noqa: S506
        if not isinstance(document, list) or not document:
            raise ValueError("empty or invalid corpus")
        cases = [InspectionCase.model_validate(item) for item in document]
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("duplicate case ID")
        if source.sha256 != hashlib.sha256(raw).hexdigest() or source.case_count != len(cases):
            raise ValueError("corpus provenance mismatch")
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        raise ValueError("Invalid inspection evaluation corpus or provenance") from None
    return cases, source.model_dump()


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    """Rates are conditional on valid decisions; coverage exposes excluded failures."""
    valid = [row for row in results if row.observed_outcome in {"benign", "malicious"}]
    benign = [row for row in valid if row.expected_outcome == "benign"]
    attacks = [row for row in valid if row.expected_outcome == "malicious"]
    fp = sum(row.observed_outcome == "malicious" for row in benign)
    fn = sum(row.observed_outcome == "benign" for row in attacks)
    return {
        "total": len(results),
        "evaluated": len(valid),
        "benign_total": sum(row.expected_outcome == "benign" for row in results),
        "malicious_total": sum(row.expected_outcome == "malicious" for row in results),
        "benign_evaluated": len(benign),
        "malicious_evaluated": len(attacks),
        "true_positives": len(attacks) - fn,
        "true_negatives": len(benign) - fp,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": fp / len(benign) if benign else None,
        "false_negative_rate": fn / len(attacks) if attacks else None,
        "coverage": len(valid) / len(results) if results else None,
        "errors": sum(row.observed_outcome == "error" for row in results),
        "unverified": sum(row.observed_outcome == "unverified" for row in results),
        "category_mismatches": sum(
            row.expected_category != row.observed_category
            for row in attacks
            if row.observed_outcome == "malicious" and row.expected_category is not None
        ),
    }


class ObservedClassifier:
    """Count actual chunks/attempts without retaining classifier content or replies."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.attempts = 0
        self.chunks_total: int | None = None
        self.chunks: set[int] = set()

    async def classify(self, request: Any, *, deadline: float | None = None) -> Any:
        self.attempts += 1
        self.chunks_total = request.chunk_count
        self.chunks.add(request.chunk_index)
        return await self.client.classify(request, deadline=deadline)


async def evaluate_cases(
    cases: list[InspectionCase],
    client: Any,
    *,
    context_window_tokens: int,
    case_timeout: float,
    emit: Callable[[dict[str, Any]], None],
) -> list[CaseResult]:
    """Use the complete production inspector, including retries/chunks/fail-closed errors."""
    results = []
    for case in cases:
        observed = ObservedClassifier(client)
        started = time.monotonic()
        failure: Failure | None = None
        category = None
        try:
            async with asyncio.timeout(case_timeout):
                result = await inspect_tool_result(
                    observed,
                    tool_name=case.tool_name,
                    result_type=case.result_type,
                    value=case.content,
                    codec=Utf8ByteCodec(),
                    context_window_tokens=context_window_tokens,
                    instruction_tokens=512,
                    enabled=True,
                    deadline=started + case_timeout,
                )
            outcome = "benign" if result.passed else "malicious"
            category = result.category
        except InspectionError as exc:
            outcome = "error"
            failure = (
                exc.failure_type
                if exc.failure_type
                in {"timeout", "provider_error", "invalid_response", "size_limit"}
                else "runner_error"
            )
        except TimeoutError:
            outcome, failure = "error", "timeout"
        except Exception:
            outcome, failure = "error", "runner_error"
        # CancelledError deliberately propagates; missing cases remain unverified.
        row = CaseResult(
            case_id=case.case_id,
            tags=case.tags,
            expected_outcome=case.expected_outcome,
            expected_category=case.expected_category,
            observed_outcome=outcome,
            observed_category=category,
            failure_type=failure,
            attempts=observed.attempts,
            chunks_total=observed.chunks_total,
            chunks_visited=len(observed.chunks),
            duration_ms=(time.monotonic() - started) * 1000,
        )
        results.append(row)
        emit({"record_type": "case", **row.model_dump()})
    return results


def runtime_metadata(family: Any, route: Any, context_window_tokens: int) -> dict[str, Any]:
    """Use actual resolved route/credential types, never provider labels supplied by a caller."""
    credential_type = type(family.credentials).__name__
    auth_mode = {
        "AzureApiKeyCredentialSource": "api_key",
        "AzureEntraCredentialSource": "entra_service_principal",
        "VertexCredentialSource": "google_adc",
        "BedrockCredentialSource": "aws_sigv4",
        "AnthropicCredentialSource": "api_key",
    }.get(credential_type, "api_key" if family.hosting == "openai" else "unverified")
    if credential_type == "BedrockCredentialSource" and family.credentials.role_arn:
        auth_mode = "aws_sigv4_assume_role"
    if credential_type == "AnthropicCredentialSource" and family.credentials.auth_token:
        auth_mode = "bearer_token"
    endpoint_origin = None
    endpoint_path = None
    endpoint_redacted = False
    if family.base_url:
        url = urlsplit(family.base_url)
        port = f":{url.port}" if url.port else ""
        host = f"[{url.hostname}]" if url.hostname and ":" in url.hostname else url.hostname
        endpoint_origin = f"{url.scheme}://{host}{port}"
        endpoint_path = url.path if url.path in {"", "/", "/v1", "/v1/"} else None
        endpoint_redacted = bool(
            url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/", "/v1", "/v1/"}
        )
    parameters = route.model_parameters.provider
    return {
        "route": route.route,
        "provider": route.telemetry_provider,
        "hosting": family.hosting,
        "model": route.model,
        "auth_mode": auth_mode,
        "endpoint_origin": endpoint_origin,
        "endpoint_path": endpoint_path,
        "endpoint_details_redacted": endpoint_redacted,
        "region": getattr(parameters, "region", None),
        "project": getattr(parameters, "project", None),
        "api_version": getattr(parameters, "api_version", None),
        "context_window_tokens": context_window_tokens,
        "instruction_tokens": 512,
        "reasoning": None,
        "tools": False,
        "history": False,
    }


async def run_configured(
    cases: list[InspectionCase],
    *,
    case_timeout: float,
    emit: Callable[[dict[str, Any]], None],
) -> int:
    """Construct the isolated classifier exactly as the production adapter does."""
    model = None
    status = 0
    results: list[CaseResult] = []
    phase = "setup_error"
    try:
        configure_tls()
        family = resolve_model_family()
        route = resolve_model_route(model_family=family)
        ready, _checks = run_readiness_checks(family, route)
        if not ready:
            raise ValueError("evaluation route not ready")
        model = route.create_model(route.model_parameters, None)
        model.disable_streaming = True
        client = (route.create_classifier_client or LangChainClassifierClient)(model)
        profile = getattr(model, "profile", None) or {}
        context_window = (
            profile.get("max_input_tokens") or profile.get("max_context_size") or 100_000
        )
        emit({"record_type": "metadata", **runtime_metadata(family, route, context_window)})
        phase = "runner_error"
        results = await evaluate_cases(
            cases,
            client,
            context_window_tokens=context_window,
            case_timeout=case_timeout,
            emit=emit,
        )
        status = 1 if any(row.observed_outcome == "error" for row in results) else 0
    except Exception:
        emit({"record_type": "run_error", "failure_type": phase})
        status = 2
    finally:
        if model is not None:
            try:
                await _close_model_clients(model)
            except Exception:
                emit({"record_type": "run_error", "failure_type": "cleanup_error"})
                status = 2
    emit({"record_type": "end", "case_count": len(results), "exit_code": status})
    return status


def positive_timeout(value: str) -> float:
    """Reject zero, negative and non-finite deadlines before any model calls."""
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("case timeout must be finite and positive")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--case-timeout", type=positive_timeout, default=60.0)
    args = parser.parse_args(argv)
    # This dedicated process emits decisions, never SDK logs, prompts, bodies or warnings.
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")

    def emit(record: dict[str, Any]) -> None:
        print(PREFIX + json.dumps(record, ensure_ascii=True, allow_nan=False), flush=True)

    try:
        cases, _source = load_corpus(args.dataset, args.source)
        return asyncio.run(run_configured(cases, case_timeout=args.case_timeout, emit=emit))
    except (Exception, KeyboardInterrupt):
        emit({"record_type": "run_error", "failure_type": "setup_error"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
