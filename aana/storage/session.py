from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session, sessionmaker

from aana.configs.settings import settings
from aana.storage.models.api_key import ApiServiceBase
from aana.storage.models.base import BaseEntity

__all__ = ["get_session", "get_db"]

engine = settings.db_config.get_engine()

if settings.api_service.enabled:
    api_service_engine = settings.api_service_db_config.get_engine()
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        binds={ApiServiceBase: api_service_engine, BaseEntity: engine},
        bind=engine,  # Default engine
    )

else:
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session() -> Session:
    """Get a new SQLAlchemy Session object.

    Returns:
        Session: SQLAlchemy Session object.
    """
    return SessionLocal()


def get_db():
    """Get a database session."""
    db = get_session()
    try:
        yield db
    finally:
        db.close()


GetDbDependency = Annotated[Session, Depends(get_db)]
""" Dependency to get a database session. """
