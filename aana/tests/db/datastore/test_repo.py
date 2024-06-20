# ruff: noqa: S101
import pytest
from sqlalchemy.orm import Session

from aana.storage.models import (
    CaptionEntity,
    MediaEntity,
    TranscriptEntity,
    VideoEntity,
)
from aana.storage.repository.caption import CaptionRepository
from aana.storage.repository.media import MediaRepository
from aana.storage.repository.transcript import TranscriptRepository
from aana.storage.repository.video import VideoRepository


@pytest.fixture
def mocked_session(mocker):
    """Creates a mocked sqlalchemy.Session."""
    session = mocker.MagicMock(spec=Session)
    # Emulate the behavior of the empty database.
    session.query.return_value.filter_by.return_value.first.return_value = None
    session.query.return_value.get.return_value = None
    return session


def test_create_media(mocked_session):
    """Tests that media creation behaves as expected."""
    repo = MediaRepository(mocked_session)
    media_type = "video"
    media_id = "foo"
    media = MediaEntity(id=media_id, media_type=media_type)
    media2 = repo.create(media)

    # We need an integration test to ensure that that the id gets set
    # on creation, because the mocked version won't set it.
    assert media2 == media
    assert media2.media_type == media_type
    assert media2.id == media_id

    mocked_session.add.assert_called_once_with(media)
    mocked_session.commit.assert_called_once()


def test_create_media_with_video(mocked_session):
    """Tests that video propery is se on media and vice-versa."""
    media_repo = MediaRepository(mocked_session)
    video_repo = VideoRepository(mocked_session)
    media_type = "video"
    media_id = "foo"
    media = MediaEntity(id=media_id, media_type=media_type)
    video = VideoEntity(media=media)
    media.video = video
    media2 = media_repo.create(media)
    mocked_session.add.assert_called_with(media2)
    video2 = video_repo.create(video)
    mocked_session.add.assert_called_with(video2)

    assert media2.video == video2
    assert video2.media == media2


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
    _ = MediaEntity(id=media_id, media_type=media_type)
    _ = VideoEntity(id=media_id, duration=video_duration)

    caption = CaptionEntity(
        media_id=media_id,
        model=model_name,
        frame_id=frame_id,
        caption=caption_text,
        timestamp=timestamp,
    )
    caption2 = repo.create(caption)

    # See above
    assert caption2.media_id == media_id
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
    _ = VideoEntity(id=media_id, duration=duration)
    transcript = TranscriptEntity(
        media_id=media_id,
        model=model_name,
        transcript=transcript_text,
        segments=segments,
        language=language,
        language_confidence=language_confidence,
    )
    transcript2 = repo.create(transcript)

    # See above
    assert transcript2.media_id == media.id
    assert transcript2.model == model_name
    assert transcript2.transcript == transcript_text
    assert transcript2.segments == segments
    assert transcript2.language == language
    assert transcript2.language_confidence == language_confidence

    mocked_session.add.assert_called_once_with(transcript)
    mocked_session.commit.assert_called_once()
