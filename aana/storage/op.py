"""Database operations and session management utilities."""

import logging
import typing
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import orjson
from alembic.config import Config
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.exc import PendingRollbackError, SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from aana.configs.db import DbType
from aana.exceptions.db import DatabaseException
from aana.exceptions.runtime import EmptyMigrationsException
from aana.storage.models.api_key import ApiServiceBase
from aana.storage.models.base import BaseEntity
from aana.utils.asyncio import run_async
from aana.utils.core import get_module_dir
from aana.utils.json import jsonify

if typing.TYPE_CHECKING:
    from aana.configs.db import DbSettings
    from aana.configs.settings import Settings

logger = logging.getLogger(__name__)


class DatabaseSessionManager:
    """Database session manager for handling database connections and sessions."""

    def __init__(self, settings: "Settings"):
        """Constructor.

        Args:
            settings: Application settings containing database configuration.
        """
        self._settings = settings
        self._is_api_service_enabled = settings.api_service.enabled

        self._db_config = settings.db_config
        self._engine = None

        self._api_service_db_config = settings.api_service_db_config
        self._api_service_engine = None

        self._sessionmaker = None

        self.init_engine()

    def init_engine(self):
        """Creates a new database engine and session maker asynchronously."""
        if self._engine is None:
            self._engine = self.create_database_engine(self._db_config)
        if self._is_api_service_enabled and self._api_service_engine is None:
            self._api_service_engine = self.create_database_engine(
                self._api_service_db_config
            )

        if self._sessionmaker is None:
            if self._is_api_service_enabled:
                self._sessionmaker = async_sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    expire_on_commit=False,
                    binds={
                        ApiServiceBase: self._api_service_engine,
                        BaseEntity: self._engine,
                    },
                    bind=self._engine,  # Default engine
                )
            else:
                self._sessionmaker = async_sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    expire_on_commit=False,
                    bind=self._engine,
                )

    @classmethod
    def create_postgresql_engine(cls, db_config: "DbSettings") -> AsyncEngine:
        """Create PostgreSQL async engine based on the provided configuration.

        Args:
            db_config (DbSettings): Database configuration.

        Returns:
            AsyncEngine: SQLAlchemy async engine instance.
        """
        datastore_config = db_config.datastore_config
        user = datastore_config["user"]
        password = datastore_config["password"]
        host = datastore_config["host"]
        port = datastore_config["port"]
        database = datastore_config["database"]
        connection_string = (
            f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        )
        connect_args = {
            "server_settings": {
                "statement_timeout": str(db_config.query_timeout * 1000)
            }
        }

        return create_async_engine(
            connection_string,
            json_serializer=lambda obj: jsonify(obj),
            json_deserializer=orjson.loads,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_recycle=db_config.pool_recycle,
            pool_timeout=db_config.connection_timeout,
            connect_args=connect_args,
        )

    @classmethod
    def create_sqlite_engine(cls, db_config: "DbSettings") -> AsyncEngine:
        """Create an SQLite async SQLAlchemy engine based on the provided configuration.

        Args:
            db_config (DbSettings): Database configuration.

        Returns:
            AsyncEngine: SQLAlchemy async engine instance.
        """
        datastore_config = db_config.datastore_config
        connection_string = f"sqlite+aiosqlite:///{datastore_config['path']}"
        return create_async_engine(
            connection_string,
            json_serializer=lambda obj: jsonify(obj),
            json_deserializer=orjson.loads,
            pool_recycle=db_config.pool_recycle,
        )

    @classmethod
    def create_database_engine(cls, db_config: "DbSettings") -> AsyncEngine:
        """Create SQLAlchemy async engine based on the provided configuration.

        Args:
            db_config (DbSettings): Database configuration.

        Returns:
            AsyncEngine: SQLAlchemy async engine instance.
        """
        db_type = getattr(db_config, "datastore_type", "").lower()

        if db_type == DbType.POSTGRESQL:
            return cls.create_postgresql_engine(db_config)
        elif db_type == DbType.SQLITE:
            return cls.create_sqlite_engine(db_config)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")  # noqa: TRY003

    async def close(self):
        """Closes the database session manager by disposing of the engine and resetting the sessionmaker.

        Raises:
            InternalException: If the database session manager is not initialized.
        """
        if self._engine is None:
            raise RuntimeError("DatabaseSessionManager is not initialized")  # noqa: TRY003
        await self._engine.dispose()

        self._engine = None
        self._sessionmaker = None

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        """Asynchronously connects to the database and provides a connection iterator.

        Yields:
            AsyncIterator[AsyncConnection]: An asynchronous iterator that provides a database connection.

        Raises:
            InternalException: If the database session manager is not initialized.
            Exception: If an error occurs during the connection, the transaction is rolled back and the exception is re-raised.
        """
        if self._engine is None:
            raise RuntimeError("DatabaseSessionManager is not initialized")  # noqa: TRY003

        async with self._engine.begin() as connection:
            try:
                yield connection
            except Exception:
                await connection.rollback()
                raise

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Provides an asynchronous context manager for database sessions.

        Yields:
            AsyncSession: An instance of the asynchronous database session.

        Raises:
            DatabaseException: If there is an error during the session.
            InternalException: If the session maker is not initialized.
            Exception: If an error occurs during the session, the session is rolled back and the exception is re-raised.
        """
        if self._sessionmaker is None:
            raise RuntimeError("DatabaseSessionManager is not initialized")  # noqa: TRY003

        session = self._sessionmaker()
        try:
            yield session
        except PendingRollbackError as e:
            await session.rollback()
            raise DatabaseException(message="Transaction was rolled back") from e
        except SQLAlchemyError as e:
            raise DatabaseException(message=str(e)) from e
        finally:
            await session.close()


async def run_alembic_migrations_async(
    settings: "Settings", root_path: Path | None = None
):
    """Run database migrations asynchronously.

    Args:
        settings (Settings): Application settings
        root_path (Path | None): Root path of the application

    Raises:
        RuntimeError: If the alembic directory does not exist
        EmptyMigrationsException: If no migrations are found
    """
    logger.info("Running database migrations...")

    session_manager = DatabaseSessionManager(settings)

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

    # Configure Alembic
    alembic_cfg = Config(ini_file_path)
    alembic_cfg.set_main_option("script_location", str(alembic_data_path))
    script = ScriptDirectory.from_config(alembic_cfg)

    # Define the upgrade function
    def upgrade(rev, context):
        """Upgrade to a specific revision."""
        return script._upgrade_revs("head", rev)

    def run_migration(connection):
        """Run the migration using the provided connection."""
        # Create migration context
        metadata = BaseEntity.metadata
        context = MigrationContext.configure(
            connection, opts={"target_metadata": metadata, "fn": upgrade}
        )

        # Run migrations
        with context.begin_transaction(), Operations.context(context):
            context.run_migrations()

        logger.info("Database migrations completed.")

    async with session_manager.connect() as connection:
        await connection.run_sync(run_migration)

    await session_manager.close()


def run_alembic_migrations(settings: "Settings", root_path: Path | None = None):
    """Run database migrations.

    Args:
        settings (Settings): Application settings
        root_path (Path | None): Root path of the application
    """
    run_async(run_alembic_migrations_async(settings, root_path=root_path))
