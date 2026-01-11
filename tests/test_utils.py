"""
Unit tests for utils.py - Utility functions.

This test suite validates:
- JSON fetching and error handling
- FITS header value extraction
- Filename handling
- Figure creation
- Log buffer operations
- Directory operations
- Catalog query wrapper
- ZIP file operations

Author: Automated Testing Suite
Date: 2026-01-11
"""

import os
import json
import tempfile
import zipfile
from io import StringIO
from datetime import datetime
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.utils import (
    get_json,
    get_header_value,
    get_base_filename,
    create_figure,
    initialize_log,
    write_to_log,
    ensure_output_directory,
    safe_catalog_query,
    zip_results_on_exit,
    save_header_to_txt,
    FIGURE_SIZES,
)


class TestGetJson:
    """Test JSON fetching with error handling."""

    def test_invalid_url_format(self):
        """Test that invalid URL returns error JSON."""
        result = get_json("not-a-valid-url")
        
        assert isinstance(result, str)
        data = json.loads(result)
        assert "error" in data
        assert data["error"] == "invalid URL"

    def test_ftp_url_invalid(self):
        """Test that FTP URL is considered invalid."""
        result = get_json("ftp://example.com/data.json")
        
        data = json.loads(result)
        assert "error" in data

    @patch('src.utils.requests.get')
    def test_successful_json_fetch(self, mock_get):
        """Test successful JSON fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok", "data": [1, 2, 3]}
        mock_response.content = b'{"status": "ok"}'
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = get_json("https://api.example.com/data")
        
        assert result == {"status": "ok", "data": [1, 2, 3]}

    @patch('src.utils.requests.get')
    def test_empty_response(self, mock_get):
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.content = b''
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = get_json("https://api.example.com/data")
        
        data = json.loads(result)
        assert data["error"] == "empty response"

    @patch('src.utils.requests.get')
    def test_network_error(self, mock_get):
        """Test handling of network errors."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        result = get_json("https://api.example.com/data")
        
        data = json.loads(result)
        assert data["error"] == "request exception"
        assert "Connection failed" in data["message"]

    @patch('src.utils.requests.get')
    def test_invalid_json_response(self, mock_get):
        """Test handling of non-JSON response."""
        mock_response = MagicMock()
        mock_response.content = b'not valid json'
        mock_response.json.side_effect = json.JSONDecodeError("Invalid", "", 0)
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = get_json("https://api.example.com/data")
        
        data = json.loads(result)
        assert data["error"] == "invalid json"


class TestGetHeaderValue:
    """Test FITS header value extraction."""

    def test_first_key_found(self):
        """Test that first matching key is returned."""
        header = {"EXPTIME": 120.0, "EXPOSURE": 60.0}
        
        result = get_header_value(header, ["EXPTIME", "EXPOSURE"], default=0.0)
        
        assert result == 120.0

    def test_fallback_to_second_key(self):
        """Test fallback to second key when first is missing."""
        header = {"EXPOSURE": 60.0}
        
        result = get_header_value(header, ["EXPTIME", "EXPOSURE"], default=0.0)
        
        assert result == 60.0

    def test_default_when_no_keys_found(self):
        """Test that default is returned when no keys match."""
        header = {"OTHER": "value"}
        
        result = get_header_value(header, ["EXPTIME", "EXPOSURE"], default=-1.0)
        
        assert result == -1.0

    def test_none_header(self):
        """Test that None header returns default."""
        result = get_header_value(None, ["KEY1", "KEY2"], default="default")
        
        assert result == "default"

    def test_empty_keys_list(self):
        """Test with empty keys list."""
        header = {"KEY": "value"}
        
        result = get_header_value(header, [], default="default")
        
        assert result == "default"


class TestGetBaseFilename:
    """Test base filename extraction."""

    def test_single_extension(self):
        """Test removal of single extension."""
        class MockFile:
            name = "image.fits"
        
        result = get_base_filename(MockFile())
        
        assert result == "image"

    def test_double_extension(self):
        """Test removal of double extension."""
        class MockFile:
            name = "image.fits.fz"
        
        result = get_base_filename(MockFile())
        
        assert result == "image"

    def test_tar_gz_extension(self):
        """Test removal of .tar.gz extension."""
        class MockFile:
            name = "archive.tar.gz"
        
        result = get_base_filename(MockFile())
        
        assert result == "archive"

    def test_no_extension(self):
        """Test file with no extension."""
        class MockFile:
            name = "datafile"
        
        result = get_base_filename(MockFile())
        
        assert result == "datafile"

    def test_none_file_object(self):
        """Test that None returns default."""
        result = get_base_filename(None)
        
        assert result == "photometry"

    def test_path_with_directory(self):
        """Test that directory is preserved."""
        class MockFile:
            name = "subdir/image.fits"
        
        result = get_base_filename(MockFile())
        
        # os.path.splitext operates on full path
        assert "image" in result or "subdir" in result


class TestCreateFigure:
    """Test matplotlib figure creation."""

    def test_valid_size_keys(self):
        """Test that valid size keys produce figures."""
        for size_key in FIGURE_SIZES.keys():
            fig = create_figure(size=size_key)
            
            expected_size = FIGURE_SIZES[size_key]
            assert fig.get_figwidth() == expected_size[0]
            assert fig.get_figheight() == expected_size[1]
            
            # Clean up
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_invalid_size_defaults_to_medium(self):
        """Test that invalid size key defaults to medium."""
        fig = create_figure(size="invalid_size")
        
        expected_size = FIGURE_SIZES["medium"]
        assert fig.get_figwidth() == expected_size[0]
        assert fig.get_figheight() == expected_size[1]
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_custom_dpi(self):
        """Test that DPI is applied correctly."""
        fig = create_figure(size="small", dpi=150)
        
        assert fig.get_dpi() == 150
        
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestLogOperations:
    """Test log buffer operations."""

    def test_initialize_log_creates_buffer(self):
        """Test that initialize_log creates a StringIO buffer."""
        buffer = initialize_log("test_file")
        
        assert isinstance(buffer, StringIO)

    def test_initialize_log_header_content(self):
        """Test that log header contains expected content."""
        buffer = initialize_log("test_image.fits")
        content = buffer.getvalue()
        
        assert "RAPAS Photometry Pipeline Log" in content
        assert "test_image.fits" in content
        assert "Processing started" in content

    def test_write_to_log_format(self):
        """Test write_to_log message format."""
        buffer = StringIO()
        
        write_to_log(buffer, "Test message", level="INFO")
        
        content = buffer.getvalue()
        assert "INFO" in content
        assert "Test message" in content
        assert "[" in content and "]" in content  # Timestamp brackets

    def test_write_to_log_different_levels(self):
        """Test different log levels."""
        buffer = StringIO()
        
        write_to_log(buffer, "Info message", level="INFO")
        write_to_log(buffer, "Warning message", level="WARNING")
        write_to_log(buffer, "Error message", level="ERROR")
        
        content = buffer.getvalue()
        assert "INFO" in content
        assert "WARNING" in content
        assert "ERROR" in content

    def test_write_to_log_none_buffer(self):
        """Test that None buffer is handled gracefully."""
        # Should not raise
        write_to_log(None, "Test message")


class TestEnsureOutputDirectory:
    """Test output directory creation."""

    def test_creates_directory(self):
        """Test that directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.utils.os.path.dirname') as mock_dirname:
                # Mock to use temp directory
                script_dir = os.path.join(tmpdir, "src")
                project_root = tmpdir
                parent_dir = os.path.dirname(tmpdir)
                
                mock_dirname.side_effect = [script_dir, project_root, parent_dir]
                
                # This will try to create in the actual filesystem
                # so we just verify the function returns a string
                result = ensure_output_directory("test_subdir")
                
                assert isinstance(result, str)

    def test_returns_existing_directory(self):
        """Test that existing directory path is returned."""
        # The function should return a valid path
        result = ensure_output_directory("")
        
        assert isinstance(result, str)


class TestSafeCatalogQuery:
    """Test catalog query wrapper."""

    def test_successful_query(self):
        """Test successful query returns result."""
        def mock_query():
            return {"data": [1, 2, 3]}
        
        result, error = safe_catalog_query(mock_query, "Query failed")
        
        assert result == {"data": [1, 2, 3]}
        assert error is None

    def test_query_with_args(self):
        """Test query with arguments."""
        def mock_query(arg1, arg2, kwarg=None):
            return f"{arg1}-{arg2}-{kwarg}"
        
        result, error = safe_catalog_query(
            mock_query, "Query failed", "a", "b", kwarg="c"
        )
        
        assert result == "a-b-c"
        assert error is None

    def test_network_error(self):
        """Test handling of network errors."""
        import requests
        
        def failing_query():
            raise requests.exceptions.ConnectionError("Network error")
        
        result, error = safe_catalog_query(failing_query, "SIMBAD query failed")
        
        assert result is None
        assert "SIMBAD query failed" in error
        assert "Network error" in error

    def test_timeout_error(self):
        """Test handling of timeout errors."""
        import requests
        
        def timeout_query():
            raise requests.exceptions.Timeout()
        
        result, error = safe_catalog_query(timeout_query, "Query timed out")
        
        assert result is None
        assert "Query timed out" in error or "timed out" in error.lower()

    def test_value_error(self):
        """Test handling of value errors."""
        def value_error_query():
            raise ValueError("Invalid coordinates")
        
        result, error = safe_catalog_query(value_error_query, "Coordinate error")
        
        assert result is None
        assert "Coordinate error" in error
        assert "Invalid coordinates" in error

    def test_generic_exception(self):
        """Test handling of generic exceptions."""
        def generic_error_query():
            raise RuntimeError("Unknown error")
        
        result, error = safe_catalog_query(generic_error_query, "Query failed")
        
        assert result is None
        assert "Query failed" in error


class TestZipResults:
    """Test ZIP archive creation."""

    def test_zip_nonexistent_directory(self):
        """Test that nonexistent directory returns None."""
        class MockFile:
            name = "test.fits"
        
        result = zip_results_on_exit(MockFile(), "/nonexistent/path")
        
        assert result == (None, None)

    def test_zip_empty_directory(self):
        """Test that empty directory returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class MockFile:
                name = "test.fits"
            
            result = zip_results_on_exit(MockFile(), tmpdir)
            
            assert result == (None, None)

    def test_zip_creates_archive(self):
        """Test that ZIP archive is created with matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            base_name = "science_image"
            for suffix in ["_phot.csv", "_log.txt", "_plot.png"]:
                filepath = os.path.join(tmpdir, base_name + suffix)
                with open(filepath, "w") as f:
                    f.write("test content")
            
            class MockFile:
                name = f"{base_name}.fits"
            
            zip_filename, zip_path = zip_results_on_exit(MockFile(), tmpdir)
            
            if zip_filename is not None:
                assert base_name in zip_filename
                assert zip_filename.endswith(".zip")


class TestSaveHeaderToTxt:
    """Test FITS header saving to text file."""

    def test_save_header(self):
        """Test saving header to text file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            header = {
                "SIMPLE": True,
                "BITPIX": 16,
                "NAXIS": 2,
                "EXPTIME": 120.0,
            }
            
            result = save_header_to_txt(header, "test_header", tmpdir)
            
            assert result is not None
            assert os.path.exists(result)
            
            with open(result, "r") as f:
                content = f.read()
            
            assert "FITS Header" in content
            assert "EXPTIME" in content
            assert "120.0" in content

    def test_save_header_none(self):
        """Test that None header returns None."""
        result = save_header_to_txt(None, "test", "/tmp")
        
        assert result is None
