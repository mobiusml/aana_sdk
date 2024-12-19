import os
from os import PathLike
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings
from sqlalchemy.engine import Engine
from typing_extensions import TypedDict

from aana.storage.op import DbType, create_database_engine


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
        port (str): The port of the PostgreSQL server.
        user (str): The user to connect to the PostgreSQL server.
        password (str): The password to connect to the PostgreSQL server.
        database (str): The database name.
    """

    host: str
    port: str
    user: str
    password: str
    database: str


class SnowflakeConfig(TypedDict, total=False):
    """Config values for Snowflake.

    For now two connection methods are supported: user/password and OAuth token from session file.

    For user/password connection method you need to provide: account, user, password.
    For OAuth connection method you need to provide: account, host and set authenticator to "oauth".

    For OAuth connection method, we only support getting the token from the session file for now since
    this is what we need to deploy the app in Snowflake cloud.

    Other attributes (database, schema, warehouse, role) are optional and can be set for both connection methods.

    Attributes:
        account (str): The account name.
        user (str | None): The user to connect to the Snowflake server.
        host (str | None): The host of the Snowflake server.
        authenticator (str | None): The authenticator to use to connect to the Snowflake server (only "oauth" or None are supported).
        password (str | None): The password to connect to the Snowflake server.
        database (str | None): The database name.
        schema (str | None): The schema name.
        warehouse (str | None): The warehouse name.
        role (str | None): The role name.
    """

    account: str
    user: str | None
    host: str | None
    authenticator: Literal["oauth"] | None
    password: str | None
    database: str | None
    schema: str | None
    warehouse: str | None
    role: str | None


class DbSettings(BaseSettings):
    """Database configuration.

    Attributes:
        datastore_type (DbType | str): The type of the datastore. Default is DbType.SQLITE.
        datastore_config (SQLiteConfig | PostgreSQLConfig | SnowflakeConfig): The configuration for the datastore.
            Default is SQLiteConfig(path="/var/lib/aana_data").
        pool_size (int): The number of connections to keep in the pool. Default is 5.
        max_overflow (int): The number of connections that can be created when the pool is exhausted.
            Default is 10.
        pool_recycle (int): The number of seconds a connection can be idle in the pool before it is invalidated.
            Default is 3600.
    """

    datastore_type: DbType | str = DbType.SQLITE
    datastore_config: SQLiteConfig | PostgreSQLConfig | SnowflakeConfig = SQLiteConfig(
        path="/var/lib/aana_data"
    )
    pool_size: int = 5
    max_overflow: int = 10
    pool_recycle: int = 3600

    _engine: Engine | None = None

    def get_engine(self):
        """Gets engine. Each instance of DbSettings will create a max.of 1 engine."""
        if not self._engine:
            self._engine = create_database_engine(self)
        return self._engine

    def __getstate__(self):
        """Used by pickle to pickle an object."""
        # We need to remove the "engine" property because SqlAlchemy engines
        # are not picklable
        state = self.__dict__.copy()
        state.pop("engine", None)
        return state

    def __setstate__(self, state):
        """Used to restore a runtime object from pickle; the opposite of __getstate__()."""
        # We don't need to do anything special here, since the engine will be recreated
        # if needed.
        self.__dict__.update(state)

    @model_validator(mode="after")
    def update_from_alias_env_vars(self):
        """Update the database configuration from alias environment variables."""
        if self.datastore_type == DbType.SNOWFLAKE:
            mapping = {
                "SNOWFLAKE_ACCOUNT": "account",
                "SNOWFLAKE_DATABASE": "database",
                "SNOWFLAKE_HOST": "host",
                "SNOWFLAKE_SCHEMA": "schema",
                "SNOWFLAKE_USER": "user",
                "SNOWFLAKE_PASSWORD": "password",
                "SNOWFLAKE_WAREHOUSE": "warehouse",
                "SNOWFLAKE_ROLE": "role",
                "SNOWFLAKE_TOKEN": "token",
                "SNOWFLAKE_AUTHENTICATOR": "authenticator",
            }
            for env_var, key in mapping.items():
                if not self.datastore_config.get(key) and os.environ.get(env_var):
                    self.datastore_config[key] = os.environ[env_var]
        return self
