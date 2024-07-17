from __future__ import annotations  # Let classes use themselves in type annotations

import typing

from sqlalchemy import CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column

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
    frame_id: Mapped[int] = mapped_column(
        CheckConstraint("frame_id >= 0", "frame_id_positive"),
        comment="The 0-based frame id of video for caption",
    )
    caption: Mapped[str] = mapped_column(comment="Frame caption")
    timestamp: Mapped[float] = mapped_column(
        CheckConstraint("timestamp >= 0", name="timestamp_positive"),
        comment="Frame timestamp in seconds",
    )
    caption_type: Mapped[str] = mapped_column(comment="The type of caption")

    __mapper_args__ = {  # noqa: RUF012
        "polymorphic_identity": "caption",
        "polymorphic_on": "caption_type",
    }

    @classmethod
    def from_caption_output(
        cls,
        model_name: str,
        caption: Caption,
        frame_id: int,
        timestamp: float,
    ) -> CaptionEntity:
        """Converts a Caption pydantic model to a CaptionEntity."""
        return CaptionEntity(
            model=model_name,
            frame_id=frame_id,
            caption=str(caption),
            timestamp=timestamp,
        )
