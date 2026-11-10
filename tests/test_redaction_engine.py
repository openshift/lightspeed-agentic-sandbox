import json
import re
from pathlib import Path
from unittest.mock import patch

from lightspeed_agentic.redaction import engine as engine_module
from lightspeed_agentic.redaction.engine import RedactionRule, Redactor, RuleSet
from lightspeed_agentic.redaction.rulesets.mcp import MCP_RULES

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tool_output_redaction.json"


def test_sequential_mcp_rules_match_conformance_fixtures() -> None:
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    compiled = MCP_RULES.compile()

    for fixture in fixtures:
        result = Redactor.redact(fixture["input"], compiled)
        assert result.text == fixture["expected"], fixture["name"]
        assert result.redacted is fixture["redacted"], fixture["name"]
        assert result.redacted_count == fixture["redacted_count"], fixture["name"]
        assert result.counts == fixture["counts"], fixture["name"]


def test_sequential_rules_compile_once_and_reuse_compiled_matchers() -> None:
    rules = RuleSet(
        name="test",
        rules=(
            RedactionRule("first", "alpha", "[FIRST]"),
            RedactionRule("second", "beta", "[SECOND]"),
        ),
    )
    original_compile = re.compile
    with patch.object(engine_module.re, "compile", wraps=original_compile) as compile_regex:
        compiled = rules.compile()
        assert compile_regex.call_count == 2
        first = Redactor.redact("alpha beta", compiled)
        second = Redactor.redact("beta", compiled)
        assert compile_regex.call_count == 2

    assert first.text == "[FIRST] [SECOND]"
    assert first.counts == {"first": 1, "second": 1}
    assert second.text == "[SECOND]"
    assert second.counts == {"second": 1}
