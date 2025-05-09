# ruff: noqa: A002
from collections.abc import Iterable
from typing import Generic, TypeVar
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from aana.core.models.media import MediaId
from aana.exceptions.db import NotFoundException
from aana.storage.models.base import BaseEntity

T = TypeVar("T", bound=BaseEntity)


# Does not yet have an update method because I'm not sure if we'll need one.
class BaseRepository(Generic[T]):
    """Base class for repositories."""

    session: AsyncSession
    table_name: str
    model_class: type[T]

    def __init__(self, session: AsyncSession, model_class: type[T]):
        """Constructor."""
        self.session = session
        self.table_name = model_class.__tablename__
        self.model_class = model_class

    def _convert_to_uuid(self, id: str | UUID) -> UUID | None:
        """Convert string id to UUID if needed.

        Args:
            id (str | UUID): The ID to convert

        Returns:
            UUID | None: The converted UUID, or None if conversion fails
        """
        try:
            if isinstance(id, str):
                return UUID(id)
            else:
                return id
        except ValueError:
            return None

    async def create(self, entity: T) -> T:
        """Inserts a single new entity."""
        self.session.add(entity)
        await self.session.commit()
        return entity

    async def create_multiple(self, entities: Iterable[T]) -> list[T]:
        """Inserts multiple entities.

        Returns:
            list[T] - entities as a list.
        """
        entities = list(entities)
        self.session.add_all(entities)
        await self.session.commit()
        return entities

    async def read(self, item_id: int | MediaId | UUID, check: bool = True) -> T:
        """Reads a single item by id from the database.

        Args:
            item_id (int | MediaId | UUID): id of the item to retrieve
            check (bool): whether to raise if the entity is not found (defaults to True).

        Returns:
            The corresponding entity from the database if found.

        Raises:
            NotFoundException: The id does not correspond to a record in the database.
        """
        result = await self.session.get(self.model_class, item_id)
        if not result and check:
            raise NotFoundException(self.table_name, item_id)
        return result

    async def delete(self, id: int | MediaId | UUID, check: bool = True) -> T | None:
        """Deletes an entity.

        Args:
            id (int | MediaId | UUID): the id of the item to be deleted.
            check (bool): whether to raise if the entity is not found (defaults to True).

        Returns:
            The entity, if found. None, if not and `check` is False.

        Raises:
            NotFoundException if the entity is not found and `check` is True.
        """
        entity = await self.read(id, check=False)
        if entity:
            await self.session.delete(entity)
            await self.session.flush()
            await self.session.commit()
            return entity
        elif check:
            raise NotFoundException(self.table_name, id)
        else:
            return None
