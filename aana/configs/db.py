from os import PathLike

from pydantic_settings import BaseSettings
from sqlalchemy.engine import Engine
from typing_extensions import TypedDict

from aana.storage.op import DbType, create_database_engine


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


class DbSettings(BaseSettings):
    """Database configuration."""

    datastore_type: DbType | str = DbType.SQLITE
    datastore_config: SQLiteConfig | PostgreSQLConfig = SQLiteConfig(
        path="/var/lib/aana_data"
    )
    engine: Engine | None = None

    def get_engine(self):
        """Gets engine. Each instance of DbSettings will create a max.of 1 engine."""
        if not self.engine:
            self.engine = create_database_engine(self)
        return self.engine

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
