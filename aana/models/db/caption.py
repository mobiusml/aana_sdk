from __future__ import annotations  # Let classes use themselves in type annotations

import typing

from sqlalchemy import CheckConstraint, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from aana.configs.db import MediaIdSqlType, media_id_type
from aana.models.db.base import BaseEntity, TimeStampEntity

if typing.TYPE_CHECKING:
    from aana.models.pydantic.captions import Caption


class CaptionEntity(BaseEntity, TimeStampEntity):
    """ORM model for video captions."""

    __tablename__ = "captions"

    id = Column(Integer, autoincrement=True, primary_key=True)  # noqa: A003
    model = Column(
        String, nullable=False, comment="Name of model used to generate the caption"
    )
    media_id = Column(
        MediaIdSqlType,
        ForeignKey("media.id"),
        nullable=False,
        comment="Foreign key to media table",
    )
    video_id = Column(
        Integer,
        ForeignKey("video.id"),
        nullable=False,
        comment="Foreign key to video table",
    )

    frame_id = Column(
        Integer,
        CheckConstraint("frame_id >= 0"),
        comment="The 0-based frame id of video for caption",
    )
    caption = Column(String, comment="Frame caption")
    timestamp = Column(
        Float,
        CheckConstraint("timestamp >= 0"),
        comment="Frame timestamp in seconds",
    )

    video = relationship("VideoEntity", back_populates="captions")

    @classmethod
    def from_caption_output(
        cls,
        model_name: str,
        media_id: media_id_type,
        video_id: int,
        frame_id: int,
        frame_timestamp: float,
        caption: Caption,
    ) -> CaptionEntity:
        """Converts a Caption pydantic model to a CaptionEntity."""
        return CaptionEntity(
            model=model_name,
            media_id=media_id,
            video_id=video_id,
            frame_id=frame_id,
            caption=str(caption),
            timestamp=frame_timestamp,
        )
