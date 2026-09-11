"""Function tools for filesystem operations.

Lightweight FunctionTool wrappers using the agents SDK decorator for file operations.
Shell() capability from the SDK provides exec_command as a native FunctionTool.
These tools bridge filesystem operations for ChatCompletions compatibility.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from agents import function_tool


def _validate_path(path: str) -> str:
    """Validate a file path to prevent directory traversal attacks.

    For relative paths: checks they don't escape the current directory with "..".
    For absolute paths: accepts as-is (sandbox isolation provides boundary control).

    Args:
        path: The path to validate (relative or absolute).

    Returns:
        The resolved absolute path.

    Raises:
        ValueError: If a relative path attempts to escape with "..".
    """
    try:
        cwd = Path.cwd().resolve()
        input_path = Path(path)

        # Absolute paths are accepted as-is (sandbox isolation handles boundaries)
        if input_path.is_absolute():
            return str(input_path.resolve())

        # For relative paths, check they don't use "..") to escape the current directory
        # Normalize the path and check if it escapes
        resolved = (cwd / input_path).resolve()
        try:
            # Verify the resolved path is still within or under cwd
            resolved.relative_to(cwd)
        except ValueError as e:
            raise ValueError(
                f"Relative path '{path}' attempts to escape current directory with '..'"
            ) from e

        return str(resolved)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid path '{path}': {e}") from e


def _read_file_impl(path: str) -> str:
    """Read file contents (implementation).

    Args:
        path: Path to the file to read.

    Returns:
        File contents or error message.
    """
    try:
        validated_path = _validate_path(path)
        return Path(validated_path).read_text()
    except ValueError as e:
        return f"Error: {e}"
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


@function_tool(description_override="Read the contents of a file.")
def read_file(path: str) -> str:
    """Read file contents.

    Args:
        path: Path to the file to read.

    Returns:
        File contents or error message.
    """
    return _read_file_impl(path)


def _write_file_impl(path: str, content: str) -> str:
    """Write content to a file (implementation).

    Args:
        path: Path to the file to write.
        content: Content to write.

    Returns:
        Success message or error.
    """
    try:
        validated_path = _validate_path(path)
        p = Path(validated_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Successfully wrote to {path}"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error writing file: {e}"


@function_tool(description_override="Write content to a file.")
def write_file(path: str, content: str) -> str:
    """Write content to a file.

    Args:
        path: Path to the file to write.
        content: Content to write.

    Returns:
        Success message or error.
    """
    return _write_file_impl(path, content)


def _list_directory_impl(path: str = ".") -> str:
    """List directory contents (implementation).

    Args:
        path: Directory path (default: current directory).

    Returns:
        Directory listing or error message.
    """
    try:
        validated_path = _validate_path(path)
        entries = sorted(Path(validated_path).iterdir())
        lines = [f"{entry.name}{'/' if entry.is_dir() else ''}" for entry in entries]
        return "\n".join(lines) if lines else "(empty directory)"
    except ValueError as e:
        return f"Error: {e}"
    except FileNotFoundError:
        return f"Error: Directory not found: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"


@function_tool(description_override="List files in a directory.")
def list_directory(path: str = ".") -> str:
    """List directory contents.

    Args:
        path: Directory path (default: current directory).

    Returns:
        Directory listing or error message.
    """
    return _list_directory_impl(path)


def _apply_patch_impl(path: str, patch: str) -> str:
    """Apply a unified diff patch (implementation).

    Args:
        path: Path to the file to patch.
        patch: Unified diff patch content.

    Returns:
        Success message or error.
    """
    try:
        import tempfile

        validated_path = _validate_path(path)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(patch)
            patch_file = f.name

        try:
            # Find patch executable in PATH
            patch_cmd = shutil.which("patch")
            if not patch_cmd:
                return "Error: patch command not found in PATH"

            result = subprocess.run(  # noqa: S603
                [patch_cmd, validated_path, patch_file],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return f"Successfully applied patch to {path}\n{result.stdout}"
            else:
                return f"Error applying patch: {result.stderr or result.stdout}"
        finally:
            Path(patch_file).unlink()
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error applying patch: {e}"


@function_tool(description_override="Apply a unified diff patch to a file.")
def apply_patch(path: str, patch: str) -> str:
    """Apply a unified diff patch.

    Args:
        path: Path to the file to patch.
        patch: Unified diff patch content.

    Returns:
        Success message or error.
    """
    return _apply_patch_impl(path, patch)
