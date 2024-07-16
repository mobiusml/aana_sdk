import datetime
from typing import Annotated

from sqlalchemy import DateTime, MetaData, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, registry

from aana.core.models.media import MediaId

timestamp = Annotated[
    datetime.datetime,
    mapped_column(DateTime(timezone=True), server_default=func.now()),
]


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

    registry = registry(
        type_annotation_map={
            MediaId: String(36),
        }
    )

    def __repr__(self) -> str:
        """Get the representation of the entity."""
        return f"{self.__class__.__name__}(id={self.id})"


class TimeStampEntity:
    """Mixin for database entities that will have create/update timestamps."""

    created_at: Mapped[timestamp] = mapped_column(
        comment="Timestamp when row is inserted",
    )
    updated_at: Mapped[timestamp] = mapped_column(
        onupdate=func.now(),
        comment="Timestamp when row is updated",
    )
