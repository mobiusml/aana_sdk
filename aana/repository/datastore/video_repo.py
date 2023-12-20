from sqlalchemy import select
from sqlalchemy.orm import Session

from aana.configs.db import media_id_type
from aana.exceptions.database import NotFoundException
from aana.models.db import VideoEntity
from aana.repository.datastore.base import BaseRepository


class VideoRepository(BaseRepository[VideoEntity]):
    """Repository for media files."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, VideoEntity)

    def get_by_media_id(self, media_id: media_id_type) -> VideoEntity:
        """Fetches a video by media_id.

        Args:
            media_id (media_id_type): Media ID to query.

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
