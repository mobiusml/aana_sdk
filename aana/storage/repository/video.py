from sqlalchemy.orm import Session

from aana.core.models.media import MediaId
from aana.storage.models import VideoEntity
from aana.storage.models.video import Status as VideoStatus
from aana.storage.repository.base import BaseRepository


class VideoRepository(BaseRepository[VideoEntity]):
    """Repository for media files."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, VideoEntity)

    def get_status(self, media_id: MediaId) -> VideoStatus:
        """Get the status for a video.

        Args:
            media_id (int | MediaId): id of the item to retrieve

        Returns:
            The corresponding video status from the database if found.

        Raises:
            NotFoundException: The id does not correspond to a video record in the database.
        """
        entity: VideoEntity = self.read(media_id=media_id)
        return entity.status
