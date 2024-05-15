from sqlalchemy import select
from sqlalchemy.orm import Session

from aana.api.models.media_id import MediaId
from aana.exceptions.db import NotFoundException
from aana.storage.models import VideoEntity
from aana.storage.repository.base import BaseRepository


class VideoRepository(BaseRepository[VideoEntity]):
    """Repository for media files."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, VideoEntity)

    def get_by_media_id(self, media_id: MediaId) -> VideoEntity:
        """Fetches a video by media_id.

        Args:
            media_id (MediaId): Media ID to query.

        Raises:
            NotFoundException: if no entry in the VideoEntity table matching that media_id is found.

        Returns:
            VideoEntity: the video.
        """
        statement = select(self.model_class).where(
            self.model_class.media_id == media_id
        )
        entity = self.session.scalars(statement).first()
        if not entity:
            raise NotFoundException(self.table_name, media_id)
        return entity

    def delete_by_media_id(self, media_id: MediaId) -> VideoEntity | None:
        """Deletes a video by media_id.

        Args:
            media_id (MediaId): Media ID to query.

        Returns:
            VideoEntity: the video.

        Raises:
            NotFoundException: if no entry in the VideoEntity table matching that media_id is found.
        """
        entity = self.get_by_media_id(media_id)
        self.session.delete(entity)
        self.session.commit()
        return entity
