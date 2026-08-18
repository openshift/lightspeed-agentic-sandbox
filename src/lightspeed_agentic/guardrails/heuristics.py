"""Pattern-based heuristic detection for pre- and post-execution guardrails."""

from __future__ import annotations

import base64
import logging
import re

from lightspeed_agentic.guardrails.types import CheckResult, GuardrailContext, Verdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-execution patterns
# ---------------------------------------------------------------------------

_EXFIL_COMMANDS = re.compile(r"\b(curl|wget|nc|ncat|netcat)\b", re.IGNORECASE)

_EXFIL_PIPE = re.compile(
    r"(cat|kubectl\s+get\s+secret|oc\s+get\s+secret).*\|.*"
    r"(curl|wget|nc|ncat|netcat|xargs\s+curl)",
    re.IGNORECASE,
)

_PRIV_ESCALATION = re.compile(
    r"\b(oc\s+adm\s+policy|kubectl\s+create\s+clusterrolebinding"
    r"|kubectl\s+create\s+clusterrole)\b",
    re.IGNORECASE,
)

_REMOTE_MANIFEST = re.compile(
    r"\b(kubectl|oc)\s+(apply|create)\s+-f\s+https?://",
    re.IGNORECASE,
)

_DESTRUCTIVE = re.compile(
    r"\b(kubectl|oc)\s+delete\b",
    re.IGNORECASE,
)

_SECRET_ACCESS = re.compile(
    r"\b(kubectl|oc)\s+get\s+secret\b.*-o\s+(yaml|json)",
    re.IGNORECASE,
)

_NAMESPACE_FLAG = re.compile(r"(?:-n|--namespace)[=\s]+(\S+)")

# ---------------------------------------------------------------------------
# Post-execution patterns
# ---------------------------------------------------------------------------

_INJECTION_PHRASES = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|directives)"
    r"|you\s+are\s+now\b"
    r"|new\s+system\s+prompt"
    r"|disregard\s+(all\s+)?(previous|prior|earlier)"
    r"|override\s+(all\s+)?(previous|prior|system)"
    r"|forget\s+(all\s+)?(previous|prior|your)\s+(instructions|rules))",
    re.IGNORECASE,
)

_ROLE_HIJACK = re.compile(
    r"^(SYSTEM|ADMIN|PRIORITY\s+OVERRIDE|IMPORTANT\s+NEW\s+DIRECTIVE)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

_HTML_COMMENT_DIRECTIVE = re.compile(
    r"<!--.*?(ignore|execute|run|delete|curl|wget|override|system\s*prompt).*?-->",
    re.IGNORECASE | re.DOTALL,
)

_CREDENTIAL_PATTERNS = re.compile(
    r"(sk-ant-[a-zA-Z0-9_-]{20,}"
    r"|AKIA[A-Z0-9]{16}"
    r"|ghp_[a-zA-Z0-9]{36}"
    r"|gho_[a-zA-Z0-9]{36}"
    r"|glpat-[a-zA-Z0-9_-]{20,}"
    r"|\bBearer\s+[a-zA-Z0-9._-]{20,}"
    r"|-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----"
    r"|-----BEGIN\s+CERTIFICATE-----"
    r"|password\s*[:=]\s*\S+"
    r"|[a-zA-Z_]*(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?\S{8,})",
    re.IGNORECASE,
)

_MAX_OUTPUT_BYTES = 50_000


def check_pre_execution(command: str, context: GuardrailContext) -> CheckResult:
    """Check a command before execution. Returns PASS, BLOCK, or SUSPICIOUS."""
    if _EXFIL_PIPE.search(command):
        return CheckResult(
            verdict=Verdict.BLOCK,
            reason="Piping secrets to network command detected",
            confidence=0.95,
        )

    if _REMOTE_MANIFEST.search(command):
        return CheckResult(
            verdict=Verdict.BLOCK,
            reason="Applying manifest from external URL",
            confidence=0.95,
        )

    if _PRIV_ESCALATION.search(command):
        return CheckResult(
            verdict=Verdict.BLOCK,
            reason="Privilege escalation command detected",
            confidence=0.95,
        )

    if _EXFIL_COMMANDS.search(command):
        return CheckResult(
            verdict=Verdict.BLOCK,
            reason="Network exfiltration command detected",
            confidence=0.9,
        )

    if context.target_namespaces:
        ns_match = _NAMESPACE_FLAG.search(command)
        if ns_match and ns_match.group(1) not in context.target_namespaces:
            ns = ns_match.group(1)
            allowed = context.target_namespaces
            return CheckResult(
                verdict=Verdict.SUSPICIOUS,
                reason=f"Command targets namespace '{ns}' outside allowed: {allowed}",
                confidence=0.7,
            )

    if _DESTRUCTIVE.search(command):
        return CheckResult(
            verdict=Verdict.SUSPICIOUS,
            reason="Destructive command — judge should evaluate alignment",
            confidence=0.6,
        )

    if _SECRET_ACCESS.search(command):
        return CheckResult(
            verdict=Verdict.SUSPICIOUS,
            reason="Secret access with structured output — judge should evaluate",
            confidence=0.6,
        )

    return CheckResult(verdict=Verdict.PASS, reason="No suspicious patterns", confidence=0.9)


def check_post_execution(output: str, _command: str, _context: GuardrailContext) -> CheckResult:
    """Check tool output after execution. Returns PASS, BLOCK, SANITIZE, or SUSPICIOUS."""
    if len(output.encode("utf-8", errors="replace")) > _MAX_OUTPUT_BYTES:
        truncated = output[: _MAX_OUTPUT_BYTES // 2]
        return CheckResult(
            verdict=Verdict.SANITIZE,
            reason=f"Output exceeds {_MAX_OUTPUT_BYTES} bytes, truncated",
            sanitized_output=truncated + "\n[GUARDRAIL] Output truncated — exceeded size limit.",
        )

    if _INJECTION_PHRASES.search(output):
        return CheckResult(
            verdict=Verdict.BLOCK,
            reason="Prompt injection phrases detected in tool output",
            confidence=0.95,
        )

    if _HTML_COMMENT_DIRECTIVE.search(output):
        return CheckResult(
            verdict=Verdict.BLOCK,
            reason="Hidden directive in HTML/Markdown comment",
            confidence=0.9,
        )

    if _has_encoded_injection(output):
        return CheckResult(
            verdict=Verdict.BLOCK,
            reason="Base64-encoded prompt injection detected",
            confidence=0.9,
        )

    if _ROLE_HIJACK.search(output):
        return CheckResult(
            verdict=Verdict.SUSPICIOUS,
            reason="Role hijacking pattern — judge should evaluate",
            confidence=0.7,
        )

    cred_matches = _CREDENTIAL_PATTERNS.findall(output)
    if cred_matches:
        sanitized = _redact_credentials(output)
        return CheckResult(
            verdict=Verdict.SANITIZE,
            reason=f"Credential patterns detected ({len(cred_matches)} matches)",
            confidence=0.85,
            sanitized_output=sanitized,
        )

    return CheckResult(verdict=Verdict.PASS, reason="No suspicious patterns", confidence=0.9)


def _has_encoded_injection(text: str) -> bool:
    b64_pattern = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
    for match in b64_pattern.finditer(text):
        try:
            decoded = base64.b64decode(match.group()).decode("utf-8", errors="ignore")
            if _INJECTION_PHRASES.search(decoded):
                return True
        except Exception:
            logger.debug("Failed to decode base64 chunk: %s", match.group()[:40])
            continue
    return False


def _redact_credentials(text: str) -> str:
    return _CREDENTIAL_PATTERNS.sub("[REDACTED]", text)
