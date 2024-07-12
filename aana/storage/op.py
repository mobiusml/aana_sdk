from enum import Enum
from pathlib import Path

import orjson
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine

from aana.utils.json import orjson_serializer


class DbType(str, Enum):
    """Engine types for relational database."""

    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


def create_postgresql_engine(config):
    """Create PostgreSQL engine based on the provided configuration.

    Args:
        config (PostgreSQLConfig): Configuration for PostgreSQL.

    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine instance.
    """
    connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    return create_engine(
        connection_string,
        json_serializer=lambda obj: orjson_serializer(obj).decode(),
        json_deserializer=orjson.loads,
    )


def create_sqlite_engine(config):
    """Create an SQLite SQLAlchemy engine based on the provided configuration.

    Args:
        config (SQLite config): Configuration for SQLite.

    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine instance.
    """
    connection_string = f"sqlite:///{config['path']}"
    return create_engine(
        connection_string,
        json_serializer=lambda obj: orjson_serializer(obj).decode(),
        json_deserializer=orjson.loads,
    )


def create_database_engine(db_config):
    """Create SQLAlchemy engine based on the provided configuration.

    Args:
        db_config (DbConfig): Database configuration.

    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine instance.
    """
    db_type = getattr(db_config, "datastore_type", "").lower()

    if db_type == DbType.POSTGRESQL:
        return create_postgresql_engine(db_config.datastore_config)
    elif db_type == DbType.SQLITE:
        return create_sqlite_engine(db_config.datastore_config)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")  # noqa: TRY003


def get_alembic_config(
    app_config, ini_file_path: Path, alembic_data_path: Path
) -> Config:
    """Produces an alembic config to run migrations programmatically."""
    engine = app_config.db_config.get_engine()
    alembic_config = Config(ini_file_path)
    alembic_config.set_main_option("script_location", str(alembic_data_path))
    config_section = alembic_config.get_section(alembic_config.config_ini_section, {})
    config_section["sqlalchemy.url"] = engine.url

    return alembic_config


def run_alembic_migrations(settings):
    """Runs alembic migrations before starting up."""
    # We need the path to aana/alembic and aana/alembic.ini
    # This is a hack until we need something better.
    current_path = Path(__file__)
    aana_root = current_path.parent.parent  # go up two directories
    if aana_root.name != "aana":  # we are not in the right place
        raise RuntimeError("Not in right directory, exiting.")  # noqa: TRY003
    ini_file_path = aana_root / "alembic.ini"
    alembic_data_path = aana_root / "alembic"

    alembic_config = get_alembic_config(settings, ini_file_path, alembic_data_path)
    engine = settings.db_config.get_engine()
    with engine.begin() as connection:
        alembic_config.attributes["connection"] = connection
        command.upgrade(alembic_config, "head")


def drop_all_tables(settings):
    """Drops all tables in the database."""
    # TODO: only allow this in testing mode
    current_path = Path(__file__)
    aana_root = current_path.parent.parent  # go up two directories
    if aana_root.name != "aana":  # we are not in the right place
        raise RuntimeError("Not in right directory, exiting.")  # noqa: TRY003
    ini_file_path = aana_root / "alembic.ini"
    alembic_data_path = aana_root / "alembic"

    alembic_config = get_alembic_config(settings, ini_file_path, alembic_data_path)
    command.downgrade(alembic_config, "base")
