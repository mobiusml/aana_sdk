
from sqlalchemy.orm import Session

from aana.core.models.asr import (
    AsrSegment,
    AsrSegments,
    AsrTranscription,
    AsrTranscriptionInfo,
)
from aana.core.models.media import MediaId
from aana.exceptions.db import NotFoundException
from aana.storage.models.transcript import TranscriptEntity
from aana.storage.repository.base import BaseRepository


class TranscriptRepository(BaseRepository[TranscriptEntity]):
    """Repository for Transcripts."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session=session, model_class=TranscriptEntity)

    def save(
        self,
        model_name: str,
        media_id: MediaId,
        transcription_info: AsrTranscriptionInfo,
        transcription: AsrTranscription,
        segments: AsrSegments,
    ) -> TranscriptEntity:
        """Save transcripts.

        Args:
            model_name (str): The name of the model used to generate the transcript.
            media_id (MediaId): The media id of the video
            transcription_info (AsrTranscriptionInfo): The ASR transcription info.
            transcription (AsrTranscription): The ASR transcription.
            segments (AsrSegments): The ASR segments.

        Returns:
            TranscriptEntity: The transcript entity.
        """
        transcript_entity = TranscriptEntity.from_asr_output(
            model_name=model_name,
            media_id=media_id,
            transcription=transcription,
            segments=segments,
            info=transcription_info,
        )
        self.session.add(transcript_entity)
        self.session.commit()
        return transcript_entity

    def get_transcript(self, model_name: str, media_id: MediaId) -> dict:
        """Get the transcript for a video.

        Args:
            model_name (str): The name of the model used to generate the transcript.
            media_id (MediaId): The media ID.

        Returns:
            dict: The dictionary with the transcript, segments, and info.
        """
        entity: TranscriptEntity | None = (
            self.session.query(self.model_class)
            .filter_by(model=model_name, media_id=media_id)
            .first()
        )
        if not entity:
            raise NotFoundException(self.table_name, media_id)
        transcription = AsrTranscription(text=entity.transcript)
        segments = [AsrSegment(**s) for s in entity.segments]
        info = AsrTranscriptionInfo(
            language=entity.language,
            language_confidence=entity.language_confidence,
        )
        return {
            "transcription": transcription,
            "segments": segments,
            "info": info,
        }
