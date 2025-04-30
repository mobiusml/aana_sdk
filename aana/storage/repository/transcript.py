from typing import TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

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

    def __init__(self, session: AsyncSession, model_class: type[T] = TranscriptEntity):
        """Constructor."""
        super().__init__(session, model_class)

    async def save(
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
        await self.session.commit()
        return transcript_entity
