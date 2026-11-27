"""Tests for the Bedrock E2E credential Secret installer."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_bedrock_installer_writes_sandbox_file_keys_and_optional_sts_inputs(
    tmp_path: Path,
) -> None:
    """The cluster Secret must expose the file keys resolved by the sandbox."""
    capture = tmp_path / "oc-create-args.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_oc = fake_bin / "oc"
    fake_oc.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [ \"${1:-}\" = create ]; then\n"
        f"  printf '%s\\n' \"$@\" > {capture}\n"
        "  printf 'apiVersion: v1\\nkind: Secret\\n'\n"
        "elif [ \"${1:-}\" = apply ]; then\n"
        "  cat >/dev/null\n"
        "else\n"
        "  exit 2\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_oc.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "AWS_ACCESS_KEY_ID": "AKIATEST",
        "AWS_SECRET_ACCESS_KEY": "secret-test",
        "AWS_SESSION_TOKEN": "session-test",
        "BEDROCK_ROLE_ARN": "arn:aws:iam::123456789012:role/e2e-bedrock",
        "E2E_NAMESPACE": "test-namespace",
        "SECRET_NAME": "test-bedrock-secret",
    }

    result = subprocess.run(
        ["/bin/bash", "scripts/e2e-install-bedrock-creds.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = capture.read_text(encoding="utf-8")
    assert "--from-file=aws_access_key_id=" in args
    assert "--from-file=aws_secret_access_key=" in args
    assert "--from-file=aws_session_token=" in args
    assert "--from-file=role_arn=" in args
    assert "--from-file=AWS_ACCESS_KEY_ID=" in args
    assert "--from-file=AWS_SECRET_ACCESS_KEY=" in args
    assert "--from-file=AWS_SESSION_TOKEN=" in args
