# ruff: noqa: S101
from importlib import resources

import pytest
from sqlalchemy.orm import Session

from aana.models.core.video import Video
from aana.models.pydantic.asr_output import (
    AsrSegment,
    AsrSegments,
    AsrTranscription,
    AsrTranscriptionInfo,
)
from aana.models.pydantic.captions import Caption, CaptionsList
from aana.models.pydantic.time_interval import TimeInterval
from aana.utils.db import (
    save_video,
    save_video_captions,
    save_video_transcription,
)


@pytest.fixture()
def mock_session(mocker):
    """Patches the Session object with a mock."""
    session_mock = mocker.MagicMock(spec=Session)
    context_var_mock = mocker.MagicMock(spec=Session)
    # Ensure that the object used inside a with block is the same.
    # Using `session_mock` doesn't work here, perhaps because it creates a
    # reference cycle.
    session_mock.return_value.__enter__.return_value = context_var_mock
    # Ensure that the context var is visible on the injected mock.
    session_mock.context_var = context_var_mock
    # Emulate the behavior of the empty database.
    context_var_mock.query.return_value.filter_by.return_value.first.return_value = None
    context_var_mock.query.return_value.get.return_value = None
    mocker.patch("aana.utils.db.Session", session_mock)
    return session_mock


def test_save_video(mock_session):
    """Tests save media function."""
    media_id = "foobar"
    duration = 550.25
    path = resources.path("aana.tests.files.videos", "squirrel.mp4")
    video = Video(path=path, media_id=media_id)
    result = save_video(video, duration)

    assert result["media_id"] == media_id
    assert result["video_id"] is None
    # once each for MediaEntity and VideoEntity
    assert mock_session.context_var.add.call_count == 2
    assert mock_session.context_var.commit.call_count == 2


def test_save_video_transcription(mock_session):
    """Tests save transcripts function."""
    media_id = "0"
    model = "test_model"
    transcript = "This is a transcript. And this is another sentence. And a third one."
    lang = "en"
    lang_conf = 0.5
    transcript = AsrTranscription(text=transcript)
    transcription_info = AsrTranscriptionInfo(
        language=lang, language_confidence=lang_conf
    )

    segments = AsrSegments(
        [
            AsrSegment(
                text="This is a transcript.",
                time_interval=TimeInterval(start=0, end=1),
                confidence=0.99,
                no_speech_confidence=0.1,
            ),
            AsrSegment(
                text="And this is another sentence.",
                time_interval=TimeInterval(start=1, end=2),
                confidence=0.99,
                no_speech_confidence=0.1,
            ),
            AsrSegment(
                text="And a third one.",
                time_interval=TimeInterval(start=2, end=3),
                confidence=0.99,
                no_speech_confidence=0.1,
            ),
        ]
    )

    video = Video(
        path=resources.path("aana.tests.files.videos", "squirrel.mp4"),
        media_id=media_id,
    )
    duration = 0.25

    result = save_video_transcription(
        model_name=model,
        video=video,
        duration=duration,
        transcription_info=transcription_info,
        transcription=transcript,
        segments=segments,
    )
    assert "transcription_id" in result

    mock_session.context_var.add_all.assert_called_once()
    # Once for media, once for video, once for transcript
    assert mock_session.context_var.commit.call_count == 3


def test_save_captions_single(mock_session):
    """Tests save captions function."""
    media_id = "0"
    model_name = "test_model"
    captions = ["A caption", "Another caption", "A third caption"]
    captions_list = CaptionsList([Caption(caption) for caption in captions])
    timestamps = [0.1, 0.2, 0.3]
    frame_ids = [0, 1, 2]

    video = Video(
        path=resources.path("aana.tests.files.videos", "squirrel.mp4"),
        media_id=media_id,
    )
    duration = 0.25

    result = save_video_captions(
        model_name, video, duration, captions_list, timestamps, frame_ids
    )

    assert (
        len(result["caption_ids"])
        == len(captions_list)
        == len(timestamps)
        == len(frame_ids)
    )
    mock_session.context_var.add_all.assert_called_once()
    # Once for media, once for video, once for captions
    assert mock_session.context_var.commit.call_count == 3
