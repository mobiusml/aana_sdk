from sqlalchemy import Column, DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase


class BaseEntity(DeclarativeBase):
    """Base for all ORM classes."""

    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_`%(constraint_name)s`",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


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
