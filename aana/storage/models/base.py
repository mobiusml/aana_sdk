import datetime
from typing import Annotated, Any, TypeVar

from sqlalchemy import DateTime, MetaData, String, func
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    object_mapper,
    registry,
)

from aana.core.models.media import MediaId

timestamp = Annotated[
    datetime.datetime,
    mapped_column(DateTime(timezone=True)),
]

T = TypeVar("T", bound="InheritanceReuseMixin")


class InheritanceReuseMixin:
    """Mixin for instantiating child classes from parent instances."""

    @classmethod
    def from_parent(cls: type[T], parent_instance: Any, **kwargs: Any) -> T:
        """Create a new instance of the child class, reusing attributes from the parent instance.

        Args:
            parent_instance (Any): An instance of the parent class
            kwargs (Any): Additional keyword arguments to set on the new instance

        Returns:
            T: A new instance of the child class
        """
        # Get the mapped attributes of the parent class
        mapper = object_mapper(parent_instance)
        attributes = {
            prop.key: getattr(parent_instance, prop.key)
            for prop in mapper.iterate_properties
            if hasattr(parent_instance, prop.key)
            and prop.key
            != mapper.polymorphic_on.name  # don't copy the polymorphic_on attribute from the parent
        }

        # Update attributes with any additional kwargs
        attributes.update(kwargs)

        # Create and return a new instance of the child class
        return cls(**attributes)


class BaseEntity(DeclarativeBase, InheritanceReuseMixin):
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
        server_default=func.now(),
        comment="Timestamp when row is inserted",
    )
    updated_at: Mapped[timestamp] = mapped_column(
        onupdate=func.now(),
        server_default=func.now(),
        comment="Timestamp when row is updated",
    )
