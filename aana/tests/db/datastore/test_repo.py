# ruff: noqa: S101
import pytest
from sqlalchemy.orm import Session

from aana.models.db import CaptionEntity, MediaEntity, TranscriptEntity, VideoEntity
from aana.repository.datastore.caption_repo import CaptionRepository
from aana.repository.datastore.media_repo import MediaRepository
from aana.repository.datastore.transcript_repo import TranscriptRepository


@pytest.fixture
def mocked_session(mocker):
    """Creates a mocked sqlalchemy.Session."""
    return mocker.MagicMock(spec=Session)


def test_create_media(mocked_session):
    """Tests that media creation behaves as expected."""
    repo = MediaRepository(mocked_session)
    media_type = "video"
    media_id = "foo"
    media = MediaEntity(id=media_id, media_type=media_type)
    media2 = repo.create(media)

    # We need an integreation test to ensure that that the id gets set
    # on creation, because the mocked version won't set it.
    assert media2 == media
    assert media2.media_type == media_type
    assert media2.id == media_id

    mocked_session.add.assert_called_once_with(media)
    mocked_session.commit.assert_called_once()


def test_create_caption(mocked_session):
    """Tests caption creation."""
    repo = CaptionRepository(mocked_session)
    media_id = "foo"
    media_type = "video"
    video_duration = 500.25
    model_name = "no_model"
    caption_text = "This is the right caption text."
    frame_id = 32767
    timestamp = 327.6
    media = MediaEntity(id=media_id, media_type=media_type)
    video = VideoEntity(media_id=media_id, duration=video_duration)
    caption = CaptionEntity(
        media_id=media_id,
        video=video,
        model=model_name,
        frame_id=frame_id,
        caption=caption_text,
        timestamp=timestamp,
    )
    caption2 = repo.create(caption)

    # See above
    assert caption2.video == video
    assert caption2.media_id == media_id
    assert caption2.video_id == video.id
    assert caption2.model == model_name
    assert caption2.frame_id == frame_id
    assert caption2.caption == caption_text
    assert caption2.timestamp == timestamp

    mocked_session.add.assert_called_once_with(caption)
    mocked_session.commit.assert_called_once()


def test_create_transcript(mocked_session):
    """Tests transcript creation."""
    repo = TranscriptRepository(mocked_session)
    media_id = "foo"
    duration = 500.25
    media_type = "video"
    model_name = "no_model"
    transcript_text = "This is the right transcript text."
    segments = "This is a segments string."
    language = "en"
    language_confidence = 0.5
    media = MediaEntity(id=media_id, media_type=media_type)
    video = VideoEntity(media_id=media_id, duration=duration)
    transcript = TranscriptEntity(
        media_id=media_id,
        video=video,
        model=model_name,
        transcript=transcript_text,
        segments=segments,
        language=language,
        language_confidence=language_confidence,
    )
    transcript2 = repo.create(transcript)

    # See above
    assert transcript2.video_id == video.id
    assert transcript2.media_id == media.id
    assert transcript2.model == model_name
    assert transcript2.transcript == transcript_text
    assert transcript2.segments == segments
    assert transcript2.language == language
    assert transcript2.language_confidence == language_confidence

    mocked_session.add.assert_called_once_with(transcript)
    mocked_session.commit.assert_called_once()
