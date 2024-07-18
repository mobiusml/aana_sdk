from typing import TypeVar

from sqlalchemy.orm import Session

from aana.core.models.media import MediaId
from aana.exceptions.db import MediaIdAlreadyExistsException, NotFoundException
from aana.storage.models.media import MediaEntity
from aana.storage.repository.base import BaseRepository

M = TypeVar("M", bound=MediaEntity)


class MediaRepository(BaseRepository[M]):
    """Repository for media files."""

    def __init__(self, session: Session, model_class: type[M] = MediaEntity):
        """Constructor."""
        super().__init__(session, model_class)

    def check_media_exists(self, media_id: MediaId) -> bool:
        """Checks if a media file exists in the database.

        Args:
            media_id (MediaId): The media ID.

        Returns:
            bool: True if the media exists, False otherwise.
        """
        try:
            self.read(media_id)
        except NotFoundException:
            return False

        return True

    def create(self, entity: MediaEntity) -> MediaEntity:
        """Inserts a single new entity.

        Args:
            entity (MediaEntity): The entity to insert.

        Returns:
            MediaEntity: The inserted entity.
        """
        # try:
        #     return super().create(entity)
        # except IntegrityError as e:
        #     self.session.rollback()
        #     raise MediaIdAlreadyExistsException(self.table_name, entity.id) from e

        if self.check_media_exists(entity.id):
            raise MediaIdAlreadyExistsException(self.table_name, entity.id)

        return super().create(entity)
