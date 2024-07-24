from sqlalchemy.orm import Session

from aana.core.models.captions import Caption, CaptionsList
from aana.core.models.media import MediaId
from aana.storage.models.extended_video_caption import ExtendedVideoCaptionEntity
from aana.storage.repository.base import BaseRepository


class ExtendedVideoCaptionRepository(BaseRepository[ExtendedVideoCaptionEntity]):
    """Repository for Captions."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, ExtendedVideoCaptionEntity)

    def save(
        self,
        model_name: str,
        media_id: MediaId,
        caption: Caption,
        timestamp: float,
        frame_id: int,
    ):
        """Save a caption.

        Args:
            model_name (str): The name of the model used to generate the caption.
            media_id (MediaId): The media ID.
            caption (Caption): The caption.
            timestamp (float): The timestamp.
            frame_id (int): The frame ID.
        """
        entity = ExtendedVideoCaptionEntity.from_caption_output(
            model_name=model_name,
            media_id=media_id,
            frame_id=frame_id,
            timestamp=timestamp,
            caption=caption,
        )
        self.create(entity)
        return entity

    def save_all(
        self,
        model_name: str,
        media_id: MediaId,
        captions: CaptionsList,
        timestamps: list[float],
        frame_ids: list[int],
    ) -> list[ExtendedVideoCaptionEntity]:
        """Save captions.

        Args:
            model_name (str): The name of the model used to generate the captions.
            media_id (MediaId): the media ID of the video.
            captions (CaptionsList): The captions.
            timestamps (list[float]): The timestamps.
            frame_ids (list[int]): The frame IDs.

        Returns:
            list[ExtendedVideoCaptionEntity]: The list of caption entities.
        """
        entities = [
            ExtendedVideoCaptionEntity.from_caption_output(
                model_name=model_name,
                media_id=media_id,
                frame_id=frame_id,
                timestamp=timestamp,
                caption=caption,
            )
            for caption, timestamp, frame_id in zip(
                captions, timestamps, frame_ids, strict=True
            )
        ]
        results = self.create_multiple(entities)
        return results

    def get_captions(self, model_name: str, media_id: MediaId) -> dict:
        """Get the captions for a video.

        Args:
            model_name (str): The model name.
            media_id (MediaId): The media ID.

        Returns:
            dict: The dictionary with the captions, timestamps, and frame IDs.
        """
        entities: list[ExtendedVideoCaptionEntity] = (
            self.session.query(self.model_class)
            .filter_by(media_id=media_id, model=model_name)
            .order_by(self.model_class.frame_id)
            .all()
        )
        captions = [c.caption for c in entities]
        timestamps = [c.timestamp for c in entities]
        frame_ids = [c.frame_id for c in entities]
        return {
            "captions": captions,
            "timestamps": timestamps,
            "frame_ids": frame_ids,
        }
