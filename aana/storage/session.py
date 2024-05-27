from sqlalchemy.orm import Session

from aana.storage.engine import engine


def get_session() -> Session:
    """Provides a SQLAlchemy Session object."""
    return Session(engine)
