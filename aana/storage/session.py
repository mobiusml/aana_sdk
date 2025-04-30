import logging
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from aana.configs.settings import settings
from aana.storage.op import DatabaseSessionManager

logger = logging.getLogger(__name__)

__all__ = ["GetDbDependency", "get_db", "get_session"]

session_manager = DatabaseSessionManager(settings)


def get_session() -> AsyncSession:
    """Get a new SQLAlchemy Session object.

    Returns:
        AsyncSession: SQLAlchemy async session
    """
    return session_manager.session()


async def get_db():
    """Get a database session.

    Returns:
        AsyncSession: SQLAlchemy async session
    """
    async with session_manager.session() as session:
        yield session


GetDbDependency = Annotated[AsyncSession, Depends(get_db)]
""" Dependency to get a database session. """
