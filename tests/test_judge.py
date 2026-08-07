"""Unit tests for the LLM judge utility (mocked — no live LLM calls)."""

from __future__ import annotations

from unittest.mock import patch

from tests.e2e.judge import JudgeInput, evaluate_scenario
from tests.e2e.runner import RunHttpResult

_SAMPLE_INPUT = JudgeInput(
    scenario_id="oom",
    request="Diagnose OOMKilled",
    expected_keywords=["OOMKilled"],
    agent_output="The pod memory-hog was OOMKilled due to exceeding its 64Mi memory limit.",
)

_SERVER_URL = "http://localhost:18080"


class TestEvaluateScenario:
    """Tests for evaluate_scenario with mocked run_query."""

    @patch("tests.e2e.judge.run_query")
    def test_passing_scenario(self, mock_run_query) -> None:
        mock_run_query.return_value = RunHttpResult(
            status_code=200,
            body={"passed": True, "reasoning": "Agent correctly identified OOMKilled."},
            raw_text='{"passed": true, "reasoning": "Agent correctly identified OOMKilled."}',
        )

        result = evaluate_scenario(_SERVER_URL, _SAMPLE_INPUT)

        assert result.passed is True
        assert "OOMKilled" in result.reasoning
        assert result.error is None

    @patch("tests.e2e.judge.run_query")
    def test_failing_scenario(self, mock_run_query) -> None:
        mock_run_query.return_value = RunHttpResult(
            status_code=200,
            body={"passed": False, "reasoning": "Agent did not identify the root cause."},
            raw_text='{"passed": false, "reasoning": "Agent did not identify the root cause."}',
        )

        result = evaluate_scenario(_SERVER_URL, _SAMPLE_INPUT)

        assert result.passed is False
        assert result.reasoning == "Agent did not identify the root cause."
        assert result.error is None

    @patch("tests.e2e.judge.run_query")
    def test_transport_error(self, mock_run_query) -> None:
        mock_run_query.return_value = RunHttpResult(
            error="Connection refused",
        )

        result = evaluate_scenario(_SERVER_URL, _SAMPLE_INPUT)

        assert result.passed is False
        assert result.reasoning == ""
        assert result.error is not None
        assert "transport error" in result.error
        assert "Connection refused" in result.error

    @patch("tests.e2e.judge.run_query")
    def test_malformed_response(self, mock_run_query) -> None:
        mock_run_query.return_value = RunHttpResult(
            status_code=200,
            body={"success": True, "summary": "unexpected shape"},
            raw_text='{"success": true, "summary": "unexpected shape"}',
        )

        result = evaluate_scenario(_SERVER_URL, _SAMPLE_INPUT)

        assert result.passed is False
        assert result.reasoning == ""
        assert result.error is not None
        assert "malformed response" in result.error

    @patch("tests.e2e.judge.run_query")
    def test_non_200_status(self, mock_run_query) -> None:
        mock_run_query.return_value = RunHttpResult(
            status_code=500,
            body={},
            raw_text="Internal Server Error",
        )

        result = evaluate_scenario(_SERVER_URL, _SAMPLE_INPUT)

        assert result.passed is False
        assert result.reasoning == ""
        assert result.error is not None
        assert "HTTP 500" in result.error
