from sqlalchemy.orm import Session

from aana.core.models.captions import CaptionsList
from aana.core.models.media import MediaId
from aana.storage.models.caption import CaptionEntity
from aana.storage.repository.base import BaseRepository


class CaptionRepository(BaseRepository[CaptionEntity]):
    """Repository for Captions."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, CaptionEntity)

    def save(
        self,
        model_name: str,
        media_id: MediaId,
        captions: CaptionsList,
        timestamps: list[float],
        frame_ids: list[int],
    ) -> list[CaptionEntity]:
        """Save captions.

        Args:
            model_name (str): The name of the model used to generate the captions.
            media_id (MediaId): the media ID of the video.
            captions (CaptionsList): The captions.
            timestamps (list[float]): The timestamps.
            frame_ids (list[int]): The frame IDs.

        Returns:
            list[CaptionEntity]: The list of caption entities.
        """
        entities = [
            CaptionEntity.from_caption_output(
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
        results = self.session.create_multiple(entities)
        return results

    def get_captions(self, model_name: str, media_id: MediaId) -> dict:
        """Get the captions for a video.

        Args:
            model_name (str): The model name.
            media_id (MediaId): The media ID.

        Returns:
            dict: The dictionary with the captions, timestamps, and frame IDs.
        """
        entities: list[CaptionEntity] = (
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
