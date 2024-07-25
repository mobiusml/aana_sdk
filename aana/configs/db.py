from os import PathLike

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


class DbSettings(BaseSettings):
    """Database configuration.

    Attributes:
        datastore_type (DbType | str): The type of the datastore. Default is DbType.SQLITE.
        datastore_config (SQLiteConfig | PostgreSQLConfig): The configuration for the datastore.
            Default is SQLiteConfig(path="/var/lib/aana_data").
    """

    datastore_type: DbType | str = DbType.SQLITE
    datastore_config: SQLiteConfig | PostgreSQLConfig = SQLiteConfig(
        path="/var/lib/aana_data"
    )
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
