from sqlalchemy.orm import Session

from aana.storage.engine import engine

__all__ = ["get_session"]


def get_session() -> Session:
    """Get a new SQLAlchemy Session object.

    Returns:
        Session: SQLAlchemy Session object.
    """
    return Session(engine)
