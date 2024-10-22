from enum import Enum
from pathlib import Path

import orjson
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine

from aana.exceptions.runtime import EmptyMigrationsException
from aana.utils.core import get_module_dir
from aana.utils.json import jsonify


class DbType(str, Enum):
    """Engine types for relational database.

    Attributes:
        POSTGRESQL: PostgreSQL database.
        SQLITE: SQLite database.
    """

    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


def create_postgresql_engine(config):
    """Create PostgreSQL engine based on the provided configuration.

    Args:
        config (PostgreSQLConfig): Configuration for PostgreSQL.

    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine instance.
    """
    connection_string = f"postgresql+psycopg://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    return create_engine(
        connection_string,
        json_serializer=lambda obj: jsonify(obj),
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
        json_serializer=lambda obj: jsonify(obj),
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


def run_alembic_migrations(settings, root_path: Path | None = None):
    """Runs alembic migrations before starting up."""
    if root_path is None:
        root_path = get_module_dir("aana")

    ini_file_path = root_path / "alembic.ini"
    alembic_data_path = root_path / "alembic"
    if not alembic_data_path.exists():
        raise RuntimeError("Alembic directory does not exist.")  # noqa: TRY003
    versions_path = alembic_data_path / "versions"
    # Check if the versions directory is empty (no .py files)
    if not versions_path.exists() or not any(Path(versions_path).glob("*.py")):
        raise EmptyMigrationsException()

    alembic_config = get_alembic_config(settings, ini_file_path, alembic_data_path)
    engine = settings.db_config.get_engine()
    with engine.begin() as connection:
        alembic_config.attributes["connection"] = connection
        command.upgrade(alembic_config, "head")


def drop_all_tables(settings, root_path: Path | None = None):
    """Drops all tables in the database."""
    # TODO: only allow this in testing mode
    if root_path is None:
        root_path = get_module_dir("aana")

    ini_file_path = root_path / "alembic.ini"
    alembic_data_path = root_path / "alembic"
    if not alembic_data_path.exists():
        raise RuntimeError("Alembic directory does not exist.")  # noqa: TRY003

    alembic_config = get_alembic_config(settings, ini_file_path, alembic_data_path)
    command.downgrade(alembic_config, "base")
