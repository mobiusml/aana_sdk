from sqlalchemy import Column, DateTime, func
from sqlalchemy.orm import DeclarativeBase


class BaseModel(DeclarativeBase):
    """Base for all ORM classes."""

    pass


class TimeStampEntity:
    """Mixin for database entities that will have create/update timestamps."""

    create_ts = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="Timestamp when row is inserted",
    )
    update_ts = Column(
        DateTime(timezone=True),
        onupdate=func.now(),
        comment="Timestamp when row is updated",
    )
