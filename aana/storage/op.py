import json
import typing
from enum import Enum
from pathlib import Path

import orjson
from alembic import command
from alembic.config import Config
from snowflake.sqlalchemy import URL as SNOWFLAKE_URL
from sqlalchemy import create_engine, event

from aana.exceptions.runtime import EmptyMigrationsException
from aana.storage.custom_types import JSON
from aana.utils.core import get_module_dir
from aana.utils.json import jsonify

if typing.TYPE_CHECKING:
    from aana.configs.db import DbSettings

import re

from snowflake.sqlalchemy.custom_types import OBJECT, VARIANT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import Insert


class DbType(str, Enum):
    """Engine types for relational database.

    Attributes:
        POSTGRESQL: PostgreSQL database.
        SQLITE: SQLite database.
    """

    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    SNOWFLAKE = "snowflake"


def create_postgresql_engine(db_config: "DbSettings"):
    """Create PostgreSQL engine based on the provided configuration.

    Args:
        db_config (DbSettings): Database configuration.

    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine instance.
    """
    datastore_config = db_config.datastore_config
    user = datastore_config["user"]
    password = datastore_config["password"]
    host = datastore_config["host"]
    port = datastore_config["port"]
    database = datastore_config["database"]
    connection_string = (
        f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
    )
    return create_engine(
        connection_string,
        json_serializer=lambda obj: jsonify(obj),
        json_deserializer=orjson.loads,
        pool_size=db_config.pool_size,
        max_overflow=db_config.max_overflow,
        pool_recycle=db_config.pool_recycle,
    )


def create_sqlite_engine(db_config: "DbSettings"):
    """Create an SQLite SQLAlchemy engine based on the provided configuration.

    Args:
        db_config (DbSettings): Database configuration.

    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine instance.
    """
    datastore_config = db_config.datastore_config
    connection_string = f"sqlite:///{datastore_config['path']}"
    return create_engine(
        connection_string,
        json_serializer=lambda obj: jsonify(obj),
        json_deserializer=orjson.loads,
        pool_size=db_config.pool_size,
        max_overflow=db_config.max_overflow,
        pool_recycle=db_config.pool_recycle,
    )


def create_snowflake_engine(db_config: "DbSettings"):
    """Create a Snowflake SQLAlchemy engine based on the provided configuration.

    Args:
        db_config (DbSettings): Database configuration.

    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine instance.
    """
    datastore_config = db_config.datastore_config
    connection_string = SNOWFLAKE_URL(**datastore_config)
    engine = create_engine(
        connection_string,
        pool_size=db_config.pool_size,
        max_overflow=db_config.max_overflow,
        pool_recycle=db_config.pool_recycle,
    )

    @event.listens_for(engine, "before_cursor_execute")
    def preprocess_parameters(
        conn, cursor, statement, parameters, context, executemany
    ):
        """Preprocess parameters before executing a query."""
        if isinstance(parameters, dict):  # Handle dict-style parameters
            for key, value in parameters.items():
                # Convert VARIANT type to JSON string
                if (
                    isinstance(value, dict | list)
                    and context.compiled
                    and key in context.compiled.binds
                    and isinstance(context.compiled.binds[key].type, JSON)
                ):
                    parameters[key] = jsonify(value)

    @compiles(Insert, "default")
    def compile_insert(insert_stmt, compiler, **kwargs):
        """Compile INSERT statements to use SELECT instead of VALUES for Snowflake PARSE_JSON."""
        sql = compiler.visit_insert(insert_stmt, **kwargs)

        # Only transform if PARSE_JSON is present in the SQL
        if "PARSE_JSON" not in sql:
            return sql

        # Locate the VALUES clause and replace it
        def replace_values_with_select(sql):
            # Regex to find `VALUES (...)` ensuring balanced parentheses
            pattern = r"VALUES\s*(\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\))"
            match = re.search(pattern, sql)
            if match:
                values_clause = match.group(1)  # Captures the `(...)` after VALUES
                # Replace VALUES (...) with SELECT ...
                return sql.replace(
                    f"VALUES {values_clause}", f"SELECT {values_clause[1:-1]}"
                )
            return sql

        # Replace the VALUES clause with SELECT
        sql = replace_values_with_select(sql)
        return sql

    return engine


def create_database_engine(db_config: "DbSettings"):
    """Create SQLAlchemy engine based on the provided configuration.

    Args:
        db_config (DbSettings): Database configuration.

    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine instance.
    """
    db_type = getattr(db_config, "datastore_type", "").lower()

    if db_type == DbType.POSTGRESQL:
        return create_postgresql_engine(db_config)
    elif db_type == DbType.SQLITE:
        return create_sqlite_engine(db_config)
    elif db_type == DbType.SNOWFLAKE:
        return create_snowflake_engine(db_config)
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
