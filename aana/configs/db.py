from enum import Enum
from os import PathLike
from pathlib import Path
from typing import TypeAlias

from alembic import command
from alembic.config import Config
from sqlalchemy import String, TypeDecorator, create_engine
from typing_extensions import TypedDict

from aana.models.pydantic.media_id import MediaId


class MediaIdType(TypeDecorator):
    """Custom type for handling MediaId objects with SQLAlchemy."""

    impl = String

    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Convert a MediaId instance to a string value for storage."""
        if value is None:
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        """Convert a string value from the database back into a MediaId instance."""
        if value is None:
            return value
        return MediaId(value)


MediaIdSqlType: TypeAlias = MediaIdType


class SQLiteConfig(TypedDict):
    """Config values for SQLite."""

    path: PathLike | str


class PostgreSQLConfig(TypedDict):
    """Config values for PostgreSQL."""

    host: str
    port: str
    user: str
    password: str
    database: str


class DbType(str, Enum):
    """Engine types for relational database."""

    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


class DBConfig(TypedDict):
    """Database configuration."""

    datastore_type: DbType | str
    datastore_config: SQLiteConfig | PostgreSQLConfig


def create_database_engine(db_config):
    """Create SQLAlchemy engine based on the provided configuration.

    Args:
        db_config (DbConfig): Database configuration.

    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine instance.
    """
    db_type = db_config.get("datastore_type", "").lower()

    if db_type == DbType.POSTGRESQL:
        return create_postgresql_engine(db_config["datastore_config"])
    elif db_type == DbType.SQLITE:
        return create_sqlite_engine(db_config["datastore_config"])
    else:
        raise ValueError(f"Unsupported database type: {db_type}")  # noqa: TRY003


def create_postgresql_engine(config):
    """Create PostgreSQL engine based on the provided configuration.

    Args:
        config (PostgreSQLConfig): Configuration for PostgreSQL.

    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine instance.
    """
    connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    return create_engine(connection_string)


def create_sqlite_engine(config):
    """Create an SQLite SQLAlchemy engine based on the provided configuration.

    Args:
        config (SQLite config): Configuration for SQLite.

    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine instance.
    """
    connection_string = f"sqlite:///{config['path']}"
    return create_engine(connection_string)


def get_alembic_config(app_config, ini_file_path, alembic_data_path) -> Config:
    """Produces an alembic config to run migrations programmatically."""
    engine = create_database_engine(app_config.db_config)
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
