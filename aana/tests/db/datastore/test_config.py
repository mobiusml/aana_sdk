# ruff: noqa: S101
import pytest

from aana.configs.db import DbSettings, PostgreSQLConfig, SQLiteConfig


@pytest.fixture
def pg_settings():
    """Fixture for working PostgreSQL settings."""
    return DbSettings(
        datastore_type="postgresql",
        datastore_config=PostgreSQLConfig(
            host="0.0.0.0",  # noqa: S104
            port="5432",
            database="postgres",
            user="postgres",
            password="bogus",  # noqa: S106
        ),
    )


@pytest.fixture
def sqlite_settings():
    """Fixture for working sqlite settings."""
    return DbSettings(
        datastore_type="sqlite",
        datastore_config=SQLiteConfig(path="/tmp/deleteme.sqlite"),  # noqa: S108
    )


def test_get_engine_idempotent(pg_settings, sqlite_settings):
    """Tests that get_engine returns the same engine on subsequent calls."""
    for db_settings in (pg_settings, sqlite_settings):
        e1 = db_settings.get_engine()
        e2 = db_settings.get_engine()
        assert e1 is e2


def test_pg_datastore_config(pg_settings):
    """Tests datastore config for postgres."""
    engine = pg_settings.get_engine()

    assert engine.name == "postgresql"
    assert str(engine.url) == "postgresql://postgres:***@0.0.0.0:5432/postgres"


def test_sqlite_datastore_config(sqlite_settings):
    """Tests datastore config for SQLite."""
    engine = sqlite_settings.get_engine()

    assert engine.name == "sqlite"
    assert str(engine.url) == f"sqlite:///{sqlite_settings.datastore_config['path']}"


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
    with pytest.raises(ValueError):
        _ = db_settings.get_engine()


def test_invalid_datastore_config(pg_settings, sqlite_settings):
    """Tests that a datastore with the wrong config raises errors."""
    tmp = pg_settings.datastore_config
    pg_settings.datastore_config = sqlite_settings.datastore_config
    sqlite_settings.datastore_config = tmp

    with pytest.raises(KeyError):
        _ = sqlite_settings.get_engine()
    with pytest.raises(KeyError):
        _ = pg_settings.get_engine()
