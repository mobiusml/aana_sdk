# ruff: noqa: S101
import pytest

from aana.configs.db import create_database_engine


def test_pg_datastore_config():
    """Tests datastore config."""
    db_config = {
        "datastore_type": "postgresql",
        "datastore_config": {
            "host": "0.0.0.0",  # noqa: S104
            "port": "5432",
            "database": "postgres",
            "user": "postgres",
            "password": "bogus",
        },
    }

    engine = create_database_engine(db_config)

    assert engine.name == "postgresql"
    assert str(engine.url) == "postgresql://postgres:***@0.0.0.0:5432/postgres"


def test_sqlite_datastore_config():
    """Tests datastore config."""
    db_config = {
        "datastore_type": "sqlite",
        "datastore_config": {"path": "/tmp/deleteme.sqlite"},  # noqa: S108
    }

    engine = create_database_engine(db_config)

    assert engine.name == "sqlite"
    assert str(engine.url) == f"sqlite:///{db_config['datastore_config']['path']}"


def test_nonexistent_datastore_config():
    """Tests datastore config."""
    db_config = {
        "datastore_type": "oracle🤮",
        "datastore_config": {
            "host": "0.0.0.0",  # noqa: S104
            "port": "5432",
            "database": "oracle",
            "user": "oracle",
            "password": "bogus",
        },
    }
    with pytest.raises(ValueError):
        _ = create_database_engine(db_config)


def test_invalid_datastore_config():
    """Tests that a datastore with the wrong config raises errors."""
    config_1 = {
        "datastore_type": "postgresql",
        "datastore_config": {"path": "/tmp/deleteme.sqlite"},  # noqa: S108
    }
    config_2 = {
        "datastore_type": "sqlite",
        "datastore_config": {
            "host": "0.0.0.0",  # noqa: S104
            "port": "5432",
            "database": "postgres",
            "user": "postgres",
            "password": "bogus",
        },
    }

    with pytest.raises(KeyError):
        _ = create_database_engine(config_1)
    with pytest.raises(KeyError):
        _ = create_database_engine(config_2)
