# ruff: noqa: A002
from collections.abc import Iterable
from typing import Generic, TypeVar

from sqlalchemy.orm import Session

from aana.configs.db import id_type
from aana.exceptions.database import NotFoundException
from aana.models.db import BaseModel

T = TypeVar("T", bound=BaseModel)


# Does not yet have an update method because I'm not sure if we'll need one.
class BaseRepository(Generic[T]):
    """Base class for repositories."""

    session: Session
    table_name: str
    model_class: type[T]

    def __init__(self, session: Session, model_class: type[T]):
        """Constructor."""
        self.session = session
        self.table_name = model_class.__tablename__
        self.model_class = model_class

    def create(self, entity: T) -> T:
        """Inserts a single new entity."""
        self.session.add(entity)
        self.session.commit()
        return entity

    def create_multiple(self, entities: Iterable[T]) -> list[T]:
        """Inserts multiple entities.

        Returns:
            list[T] - entities as a list.
        """
        entities = list(entities)
        self.session.add_all(entities)
        self.session.commit()
        return entities

    def read(self, item_id: id_type) -> T:
        """Reads a single item by id from the database.

        Args:
            item_id (id_type): id of the item to retrieve

        Returns:
            The corresponding entity from the database if found.

        Raises:
            NotFoundException: The id does not correspond to a record in the database.
        """
        entity: T | None = self.session.query(self.model_class).get(item_id)
        if not entity:
            raise NotFoundException(self.table_name, item_id)
        return entity

    def delete(self, id: id_type, check: bool = False) -> T | None:
        """Deletes an entity.

        Args:
            id (id_type): the id of the item to be deleted.
            check (bool): whether to raise if the entity is not found (defaults to True).

        Returns:
            The entity, if found. None, if not and `check` is False.

        Raises:
            NotFoundException if the entity is not found and `check` is True.
        """
        entity = self.read(id)
        if entity:
            self.session.delete(entity)
            self.session.commit()
            return entity
        elif check:
            raise NotFoundException(self.table_name, id)
        else:
            return None
