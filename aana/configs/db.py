from enum import Enum
from os import PathLike
from typing import TypeAlias, TypedDict

from sqlalchemy import Integer, create_engine

# These are here so we can change types in a single place.

id_type: TypeAlias = int
IdSqlType: TypeAlias = Integer


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
