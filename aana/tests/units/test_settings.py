# ruff: noqa: S101, S108
from pathlib import Path

from aana.configs.settings import Settings


def test_default_tmp_data_dir():
    """Test that the default temporary data directory is set correctly."""
    settings = Settings()
    assert settings.tmp_data_dir == Path("/tmp/aana_data")


def test_custom_tmp_data_dir(monkeypatch):
    """Test that the custom temporary data directory with environment variable is set correctly."""
    test_path = "/tmp/override/path"
    monkeypatch.setenv("TMP_DATA_DIR", test_path)
    settings = Settings()
    assert settings.tmp_data_dir == Path(test_path)
