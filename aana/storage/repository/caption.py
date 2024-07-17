from typing import TypeVar

from sqlalchemy.orm import Session

from aana.core.models.captions import Caption, CaptionsList
from aana.storage.models.caption import CaptionEntity
from aana.storage.repository.base import BaseRepository

T = TypeVar("T", bound=CaptionEntity)


class CaptionRepository(BaseRepository[T]):
    """Repository for Captions."""

    def __init__(self, session: Session, model_class: type[T] = CaptionEntity):
        """Constructor."""
        super().__init__(session, model_class)

    def save(self, model_name: str, caption: Caption, timestamp: float, frame_id: int):
        """Save a caption.

        Args:
            model_name (str): The name of the model used to generate the caption.
            caption (Caption): The caption.
            timestamp (float): The timestamp.
            frame_id (int): The frame ID.
        """
        entity = CaptionEntity.from_caption_output(
            model_name=model_name,
            frame_id=frame_id,
            timestamp=timestamp,
            caption=caption,
        )
        self.create(entity)
        return entity

    def save_all(
        self,
        model_name: str,
        captions: CaptionsList,
        timestamps: list[float],
        frame_ids: list[int],
    ) -> list[CaptionEntity]:
        """Save captions.

        Args:
            model_name (str): The name of the model used to generate the captions.
            captions (CaptionsList): The captions.
            timestamps (list[float]): The timestamps.
            frame_ids (list[int]): The frame IDs.

        Returns:
            list[CaptionEntity]: The list of caption entities.
        """
        entities = [
            CaptionEntity.from_caption_output(
                model_name=model_name,
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
