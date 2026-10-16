"""Offline checks for scripts used by the live skill-script scenarios."""

from __future__ import annotations

import json
import re
import subprocess

from tests.e2e.skills_fixtures import SKILLS_SOURCE

SKILL_DIR = SKILLS_SOURCE / "script-workflows"


def test_single_file_skill_script_accepts_an_argument_containing_spaces() -> None:
    argument = "argument with spaces"
    result = subprocess.run(  # noqa: S603 — execute a fixed test fixture script
        ["/bin/bash", "scripts/single-file.sh", argument],
        cwd=SKILL_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["summary"] == f"argument:{argument}"
    assert re.fullmatch(r"[0-9a-f]{32}", payload["token"])
    assert payload["status"] == "ok"


def test_command_chain_skill_scripts_share_setup_environment() -> None:
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            "source scripts/setup-token.sh && /bin/bash scripts/check-token.sh",
        ],
        cwd=SKILL_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["summary"] == "same-shell-chain"
    assert re.fullmatch(r"[0-9a-f]{32}", payload["token"])
    assert payload["status"] == "ok"
