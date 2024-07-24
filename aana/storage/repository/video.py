from typing import TypeVar

from sqlalchemy.orm import Session

from aana.core.models.media import MediaId
from aana.core.models.video import Video, VideoMetadata
from aana.storage.models import VideoEntity
from aana.storage.repository.media import MediaRepository

V = TypeVar("V", bound=VideoEntity)


class VideoRepository(MediaRepository[V]):
    """Repository for videos."""

    def __init__(self, session: Session, model_class: type[V] = VideoEntity):
        """Constructor."""
        super().__init__(session, model_class)

    def save(self, video: Video) -> dict:
        """Saves a video to datastore.

        Args:
            video (Video): The video object.

        Returns:
            dict: The dictionary with media ID.
        """
        video_entity = VideoEntity(
            id=video.media_id,
            path=str(video.path),
            url=video.url,
            title=video.title,
            description=video.description,
        )

        self.create(video_entity)
        return {
            "media_id": video_entity.id,
        }

    def get_metadata(self, media_id: MediaId) -> VideoMetadata:
        """Get the metadata of a video.

        Args:
            media_id (MediaId): The media ID.

        Returns:
            VideoMetadata: The video metadata.
        """
        entity: VideoEntity = self.read(media_id)
        return VideoMetadata(title=entity.title, description=entity.description)
