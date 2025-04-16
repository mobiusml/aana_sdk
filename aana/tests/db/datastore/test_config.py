# ruff: noqa: S101
import pytest

from aana.configs.db import DbSettings, DbType
from aana.configs.settings import settings as aana_settings
from aana.storage.op import DatabaseSessionManager


def test_datastore_config(db_session_manager):
    """Tests datastore config for PostgreSQL and SQLite."""
    engine = db_session_manager._engine
    if db_session_manager._db_config.datastore_type == DbType.POSTGRESQL:
        assert engine.name == "postgresql"
        assert str(engine.url).startswith("postgresql+asyncpg://")
    elif db_session_manager._db_config.datastore_type == DbType.SQLITE:
        assert engine.name == "sqlite"
        assert str(engine.url).startswith("sqlite+aiosqlite://")
    else:
        raise AssertionError("Unsupported database type")  # noqa: TRY003


def test_nonexistent_datastore_config():
    """Tests that datastore config errors on unsupported DB types."""
    db_settings = DbSettings(
        **{
            "datastore_type": "oracle",
            "datastore_config": {
                "host": "0.0.0.0",  # noqa: S104
                "port": "5432",
                "database": "oracle",
                "user": "oracle",
                "password": "bogus",
            },
        }
    )
    aana_settings.db_config = db_settings
    with pytest.raises(ValueError):
        DatabaseSessionManager(aana_settings)
