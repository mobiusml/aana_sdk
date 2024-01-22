from sqlalchemy.orm import Session

from aana.exceptions.database import MediaIdAlreadyExistsException
from aana.models.db import MediaEntity
from aana.repository.datastore.base import BaseRepository


class MediaRepository(BaseRepository[MediaEntity]):
    """Repository for media files."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, MediaEntity)

    def check_media_exists(self, media_id: str) -> bool:
        """Checks if a media file exists in the database.

        Args:
            media_id (str): The media ID.

        Returns:
            bool: True if the media exists, False otherwise.
        """
        return (
            self.session.query(self.model_class).filter_by(id=media_id).first()
            is not None
        )

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
