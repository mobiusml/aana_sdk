# ruff: noqa: S101

import pytest

from aana.core.models.asr import AsrSegment, AsrTranscription, AsrTranscriptionInfo
from aana.core.models.time import TimeInterval
from aana.exceptions.db import NotFoundException
from aana.storage.models.transcript import TranscriptEntity
from aana.storage.repository.transcript import TranscriptRepository

transcript_entity = TranscriptEntity.from_asr_output(
    model_name="whisper",
    transcription=AsrTranscription(text="This is a transcript"),
    segments=[],
    info=AsrTranscriptionInfo(),
)


@pytest.fixture(scope="function")
def dummy_transcript():
    """Creates a dummy transcript for testing."""
    transcript = AsrTranscription(text="This is a transcript")
    segments = [
        AsrSegment(text="This is a segment", time_interval=TimeInterval(start=0, end=1))
    ]
    info = AsrTranscriptionInfo(language="en", language_confidence=0.9)
    return transcript, segments, info


def test_save_transcript(db_session, dummy_transcript):
    """Tests saving a transcript."""
    transcript, segments, info = dummy_transcript
    model_name = "whisper"

    transcript_repo = TranscriptRepository(db_session)
    transcript_entity = transcript_repo.save(
        model_name=model_name,
        transcription_info=info,
        transcription=transcript,
        segments=segments,
    )

    transcript_id = transcript_entity.id

    transcript_entity = transcript_repo.read(transcript_id)
    assert transcript_entity
    assert transcript_entity.id == transcript_id
    assert transcript_entity.model == model_name
    assert transcript_entity.transcript == transcript.text
    assert len(transcript_entity.segments) == len(segments)
    assert transcript_entity.language == info.language
    assert transcript_entity.language_confidence == info.language_confidence

    transcript_repo.delete(transcript_id)
    with pytest.raises(NotFoundException):
        transcript_repo.read(transcript_id)
