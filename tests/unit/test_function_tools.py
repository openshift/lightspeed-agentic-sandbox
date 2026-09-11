"""Unit tests for filesystem function tools.

Shell() from agents SDK provides exec_command as a native FunctionTool.
These tests cover manually implemented filesystem operations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from lightspeed_agentic.function_tools import (
    _apply_patch_impl,
    _list_directory_impl,
    _read_file_impl,
    _write_file_impl,
)


class TestReadFile:
    """Tests for read_file function tool."""

    def test_read_file_success(self) -> None:
        """Test reading an existing file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content\n")
            temp_path = f.name

        try:
            result = _read_file_impl(temp_path)
            assert result == "test content\n"
        finally:
            Path(temp_path).unlink()

    def test_read_file_not_found(self) -> None:
        """Test reading a nonexistent file."""
        result = _read_file_impl("/nonexistent/path/file.txt")
        assert "not found" in result.lower()

    def test_read_file_empty(self) -> None:
        """Test reading an empty file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_path = f.name

        try:
            result = _read_file_impl(temp_path)
            assert result == ""
        finally:
            Path(temp_path).unlink()


class TestWriteFile:
    """Tests for write_file function tool."""

    def test_write_file_success(self) -> None:
        """Test writing to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            result = _write_file_impl(str(file_path), "test content")
            assert "successfully" in result.lower()
            assert file_path.read_text() == "test content"

    def test_write_file_creates_parents(self) -> None:
        """Test that write_file creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "a" / "b" / "c" / "test.txt"
            result = _write_file_impl(str(file_path), "nested")
            assert "successfully" in result.lower()
            assert file_path.read_text() == "nested"

    def test_write_file_overwrites(self) -> None:
        """Test that write_file overwrites existing files."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("original")
            temp_path = f.name

        try:
            result = _write_file_impl(temp_path, "new content")
            assert "successfully" in result.lower()
            assert Path(temp_path).read_text() == "new content"
        finally:
            Path(temp_path).unlink()


class TestListDirectory:
    """Tests for list_directory function tool."""

    def test_list_directory_success(self) -> None:
        """Test listing a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.txt").touch()
            Path(tmpdir, "file2.txt").touch()
            Path(tmpdir, "subdir").mkdir()

            result = _list_directory_impl(tmpdir)
            assert "file1.txt" in result
            assert "file2.txt" in result
            assert "subdir/" in result

    def test_list_directory_empty(self) -> None:
        """Test listing an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _list_directory_impl(tmpdir)
            assert "empty" in result.lower()

    def test_list_directory_not_found(self) -> None:
        """Test listing a nonexistent directory."""
        result = _list_directory_impl("/nonexistent/path")
        assert "not found" in result.lower()

    def test_list_directory_current(self) -> None:
        """Test listing current directory (default)."""
        result = _list_directory_impl()
        # Should not raise an error
        assert isinstance(result, str)


class TestApplyPatch:
    """Tests for apply_patch function tool."""

    def test_apply_patch_success(self) -> None:
        """Test applying a valid patch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("line 1\nline 2\nline 3\n")

            patch = """--- test.txt
+++ test.txt
@@ -1,3 +1,3 @@
 line 1
-line 2
+line 2 modified
 line 3
"""
            result = _apply_patch_impl(str(file_path), patch)
            assert "successfully" in result.lower()
            assert "line 2 modified" in file_path.read_text()

    def test_apply_patch_file_not_found(self) -> None:
        """Test applying patch to nonexistent file."""
        patch = "--- /dev/null\n+++ test.txt\n"
        result = _apply_patch_impl("/nonexistent/file.txt", patch)
        # Could be "not found" or "error"
        assert "error" in result.lower()

    def test_apply_patch_invalid_patch(self) -> None:
        """Test applying an invalid patch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("content")

            invalid_patch = "this is not a valid patch"
            result = _apply_patch_impl(str(file_path), invalid_patch)
            # Should error
            assert "error" in result.lower()
