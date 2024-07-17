from __future__ import annotations  # Let classes use themselves in type annotations

import typing

from sqlalchemy import CheckConstraint, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aana.core.models.media import MediaId  # noqa: TCH001
from aana.storage.models.base import BaseEntity, TimeStampEntity

if typing.TYPE_CHECKING:
    from aana.core.models.captions import Caption


class CaptionEntity(BaseEntity, TimeStampEntity):
    """ORM model for video captions."""

    __tablename__ = "caption"

    id: Mapped[int] = mapped_column(autoincrement=True, primary_key=True)
    model: Mapped[str] = mapped_column(
        nullable=False, comment="Name of model used to generate the caption"
    )
    media_id: Mapped[MediaId] = mapped_column(
        ForeignKey("extended_video.id"),
        nullable=False,
        comment="Foreign key to video table",
    )
    frame_id: Mapped[int] = mapped_column(
        CheckConstraint("frame_id >= 0", "frame_id_positive"),
        comment="The 0-based frame id of video for caption",
    )
    caption: Mapped[str] = mapped_column(comment="Frame caption")
    timestamp: Mapped[float] = mapped_column(
        CheckConstraint("timestamp >= 0", name="timestamp_positive"),
        comment="Frame timestamp in seconds",
    )

    video = relationship(
        "ExtendedVideoEntity", back_populates="captions", uselist=False
    )

    @classmethod
    def from_caption_output(
        cls,
        model_name: str,
        media_id: MediaId,
        frame_id: int,
        frame_timestamp: float,
        caption: Caption,
    ) -> CaptionEntity:
        """Converts a Caption pydantic model to a CaptionEntity."""
        return CaptionEntity(
            model=model_name,
            media_id=media_id,
            frame_id=frame_id,
            caption=str(caption),
            timestamp=frame_timestamp,
        )
