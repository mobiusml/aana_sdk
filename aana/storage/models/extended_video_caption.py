from __future__ import annotations  # Let classes use themselves in type annotations

import typing

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aana.core.models.media import MediaId  # noqa: TCH001
from aana.storage.models.caption import CaptionEntity

if typing.TYPE_CHECKING:
    from aana.core.models.captions import Caption


class ExtendedVideoCaptionEntity(CaptionEntity):
    """ORM model for video captions in extended video."""

    __tablename__ = "extended_video_caption"

    id: Mapped[int] = mapped_column(ForeignKey("caption.id"), primary_key=True)

    media_id: Mapped[MediaId] = mapped_column(
        ForeignKey("extended_video.id"),
        nullable=False,
        comment="Foreign key to video table",
    )

    video = relationship(
        "ExtendedVideoEntity", back_populates="captions", uselist=False
    )

    __mapper_args__ = {  # noqa: RUF012
        "polymorphic_identity": "extended_video_caption",
    }

    @classmethod
    def from_caption_output(
        cls,
        model_name: str,
        caption: Caption,
        media_id: MediaId,
        frame_id: int,
        timestamp: float,
    ) -> ExtendedVideoCaptionEntity:
        """Converts a Caption pydantic model to a ExtendedVideoCaptionEntity."""
        caption_entity = CaptionEntity.from_caption_output(
            model_name=model_name,
            frame_id=frame_id,
            timestamp=timestamp,
            caption=caption,
        )
        return cls.from_parent(caption_entity, media_id=media_id)
