from sqlalchemy.orm import Session, sessionmaker

from aana.storage.engine import engine

__all__ = ["get_session"]

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session() -> Session:
    """Get a new SQLAlchemy Session object.

    Returns:
        Session: SQLAlchemy Session object.
    """
    return SessionLocal()
