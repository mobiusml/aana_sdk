# ruff: noqa: A002
from collections.abc import Iterable
from typing import Generic, TypeVar
from uuid import UUID

from sqlalchemy.orm import Session

from aana.core.models.media import MediaId
from aana.exceptions.db import NotFoundException
from aana.storage.models.base import BaseEntity

T = TypeVar("T", bound=BaseEntity)


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

    def read(self, item_id: int | MediaId | UUID, check: bool = True) -> T:
        """Reads a single item by id from the database.

        Args:
            item_id (int | MediaId | UUID): id of the item to retrieve
            check (bool): whether to raise if the entity is not found (defaults to True).

        Returns:
            The corresponding entity from the database if found.

        Raises:
            NotFoundException: The id does not correspond to a record in the database.
        """
        entity: T | None = self.session.query(self.model_class).get(item_id)
        if not entity and check:
            raise NotFoundException(self.table_name, item_id)
        return entity

    # def check_id_exists(self, id: int | MediaId) -> bool:
    #     """Checks if a record with the given id exists in the database.

    #     Args:
    #         id (int | MediaId): id to check for.

    #     Returns:
    #         bool: True if the record exists, False otherwise.
    #     """
    #     # return (
    #     #     self.session.query(self.model_class).filter_by(id=str(media_id)).first()
    #     #     is not None
    #     # )

    def delete(self, id: int | MediaId | UUID, check: bool = True) -> T | None:
        """Deletes an entity.

        Args:
            id (int | MediaId | UUID): the id of the item to be deleted.
            check (bool): whether to raise if the entity is not found (defaults to True).

        Returns:
            The entity, if found. None, if not and `check` is False.

        Raises:
            NotFoundException if the entity is not found and `check` is True.
        """
        entity = self.read(id, check=False)
        if entity:
            self.session.delete(entity)
            self.session.flush()
            self.session.commit()
            return entity
        elif check:
            raise NotFoundException(self.table_name, id)
        else:
            return None
