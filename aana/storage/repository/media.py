from sqlalchemy.orm import Session

from aana.core.models.media import MediaId
from aana.exceptions.db import MediaIdAlreadyExistsException, NotFoundException
from aana.storage.models.media import MediaEntity
from aana.storage.repository.base import BaseRepository


class MediaRepository(BaseRepository[MediaEntity]):
    """Repository for media files."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, MediaEntity)

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
        if self.check_media_exists(entity.id):
            raise MediaIdAlreadyExistsException(self.table_name, entity.id)

        return super().create(entity)
