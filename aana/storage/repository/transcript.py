from typing import TypeVar

from sqlalchemy.orm import Session

from aana.core.models.asr import (
    AsrSegments,
    AsrTranscription,
    AsrTranscriptionInfo,
)
from aana.storage.models.transcript import TranscriptEntity
from aana.storage.repository.base import BaseRepository

T = TypeVar("T", bound=TranscriptEntity)


class TranscriptRepository(BaseRepository[T]):
    """Repository for Transcripts."""

    def __init__(self, session: Session, model_class: type[T] = TranscriptEntity):
        """Constructor."""
        super().__init__(session, model_class)

    def save(
        self,
        model_name: str,
        transcription_info: AsrTranscriptionInfo,
        transcription: AsrTranscription,
        segments: AsrSegments,
    ) -> TranscriptEntity:
        """Save transcripts.

        Args:
            model_name (str): The name of the model used to generate the transcript.
            transcription_info (AsrTranscriptionInfo): The ASR transcription info.
            transcription (AsrTranscription): The ASR transcription.
            segments (AsrSegments): The ASR segments.

        Returns:
            TranscriptEntity: The transcript entity.
        """
        transcript_entity = TranscriptEntity.from_asr_output(
            model_name=model_name,
            transcription=transcription,
            segments=segments,
            info=transcription_info,
        )
        self.session.add(transcript_entity)
        self.session.commit()
        return transcript_entity
