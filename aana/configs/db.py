from enum import Enum
from os import PathLike

from pydantic_settings import BaseSettings
from typing_extensions import TypedDict


class DbType(str, Enum):
    """Engine types for relational database.

    Attributes:
        POSTGRESQL: PostgreSQL database.
        SQLITE: SQLite database.
    """

    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


class SQLiteConfig(TypedDict):
    """Config values for SQLite.

    Attributes:
        path (PathLike): The path to the SQLite database file.
    """

    path: PathLike | str


class PostgreSQLConfig(TypedDict):
    """Config values for PostgreSQL.

    Attributes:
        host (str): The host of the PostgreSQL server.
        port (int): The port of the PostgreSQL server.
        user (str): The user to connect to the PostgreSQL server.
        password (str): The password to connect to the PostgreSQL server.
        database (str): The database name.
    """

    host: str
    port: int
    user: str
    password: str
    database: str


class DbSettings(BaseSettings):
    """Database configuration.

    Attributes:
        datastore_type (DbType | str): The type of the datastore. Default is DbType.SQLITE.
        datastore_config (SQLiteConfig | PostgreSQLConfig): The configuration for the datastore.
            Default is SQLiteConfig(path="/var/lib/aana_data").
        pool_size (int): The number of connections to keep in the pool. Default is 5.
        max_overflow (int): The number of connections that can be created when the pool is exhausted.
            Default is 10.
        pool_recycle (int): The number of seconds a connection can be idle in the pool before it is invalidated.
            Default is 3600.
    """

    datastore_type: DbType | str = DbType.SQLITE
    datastore_config: SQLiteConfig | PostgreSQLConfig = SQLiteConfig(
        path="/var/lib/aana_data"
    )
    pool_size: int = 5
    max_overflow: int = 10
    pool_recycle: int = 3600
