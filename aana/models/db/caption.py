from __future__ import annotations  # Let classes use themselves in type annotations

import uuid

from sqlalchemy import CheckConstraint, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from aana.configs.db import IdSqlType, id_type
from aana.models.db.base import BaseModel, TimeStampEntity
from aana.models.pydantic.captions import Caption


class CaptionEntity(BaseModel, TimeStampEntity):
    """ORM model for media captions."""

    __tablename__ = "captions"

    id = Column(IdSqlType, primary_key=True)  # noqa: A003
    model = Column(
        String, nullable=False, comment="Name of model used to generate the caption"
    )
    media_id = Column(
        IdSqlType,
        ForeignKey("media.id"),
        nullable=False,
        comment="Foreign key to media table",
    )
    frame_id = Column(
        Integer,
        CheckConstraint("frame_id >= 0"),
        comment="The 0-based frame id of media for caption",
    )
    caption = Column(String, comment="Frame caption")
    timestamp = Column(
        Float,
        CheckConstraint("timestamp >= 0"),
        comment="Frame timestamp in seconds",
    )

    media = relationship("MediaEntity", back_populates="captions")

    @classmethod
    def from_caption_output(
        cls,
        model_name: str,
        media_id: id_type,
        frame_id: int,
        frame_timestamp: float,
        caption: Caption,
    ) -> CaptionEntity:
        """Converts a Caption pydantic model to a CaptionEntity."""
        return CaptionEntity(
            id=str(uuid.uuid4()),
            model=model_name,
            media_id=media_id,
            frame_id=frame_id,
            caption=str(caption),
            timestamp=frame_timestamp,
        )
