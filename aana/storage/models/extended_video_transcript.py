from __future__ import annotations  # Let classes use themselves in type annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aana.core.models.media import MediaId  # noqa: TCH001
from aana.storage.models.transcript import TranscriptEntity

if TYPE_CHECKING:
    from aana.core.models.asr import (
        AsrSegments,
        AsrTranscription,
        AsrTranscriptionInfo,
    )


class ExtendedVideoTranscriptEntity(TranscriptEntity):
    """ORM class for extended video transcripts."""

    __tablename__ = "extended_video_transcript"

    id: Mapped[int] = mapped_column(ForeignKey("transcript.id"), primary_key=True)
    media_id: Mapped[MediaId] = mapped_column(
        ForeignKey("extended_video.id"),
        nullable=False,
        comment="Foreign key to video table",
    )

    video = relationship(
        "ExtendedVideoEntity", back_populates="transcript", uselist=False
    )

    __mapper_args__ = {  # noqa: RUF012
        "polymorphic_identity": "extended_video_transcript",
    }

    @classmethod
    def from_asr_output(
        cls,
        model_name: str,
        media_id: MediaId,
        info: AsrTranscriptionInfo,
        transcription: AsrTranscription,
        segments: AsrSegments,
    ) -> ExtendedVideoTranscriptEntity:
        """Converts an AsrTranscriptionInfo and AsrTranscription to a single Transcript entity."""
        transcript_entity = super().from_asr_output(
            model_name=model_name,
            info=info,
            transcription=transcription,
            segments=segments,
        )
        return cls.from_parent(transcript_entity, media_id=media_id)
