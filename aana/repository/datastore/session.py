from sqlalchemy.orm import Session

from aana.repository.datastore.engine import engine


def get_session() -> Session:
    """Provides a SQLAlchemy Session object."""
    return Session(engine)
