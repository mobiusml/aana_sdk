# ruff: noqa: S101
from importlib import resources

import pytest
from sqlalchemy.orm import Session

from aana.core.models.asr import (
    AsrSegment,
    AsrSegments,
    AsrTranscription,
    AsrTranscriptionInfo,
)
from aana.core.models.captions import Caption, CaptionsList
from aana.core.models.time import TimeInterval
from aana.core.models.video import Video
from aana.storage.services.video import (
    save_transcripts_batch,
    save_video,
    save_video_batch,
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
    mocker.patch("aana.storage.services.video.Session", session_mock)
    return session_mock


def test_save_video(mock_session):
    """Tests save media function."""
    media_id = "foobar"
    duration = 550.25
    path = resources.path("aana.tests.files.videos", "squirrel.mp4")
    video = Video(path=path, media_id=media_id)
    result = save_video(video, duration)

    assert result["media_id"] == media_id
    # once each for MediaEntity and VideoEntity
    assert mock_session.context_var.add.call_count == 2
    assert mock_session.context_var.commit.call_count == 2


def test_save_videos_batch(mock_session):
    """Tests save media function."""
    media_ids = ["foo", "bar"]
    path = resources.path("aana.tests.files.videos", "squirrel.mp4")
    videos = [Video(path=path, media_id=m_id) for m_id in media_ids]
    durations = [0.1] * len(media_ids)

    result = save_video_batch(videos, durations)

    assert result["media_ids"] == media_ids
    # once each for MediaEntities and VideoEntities
    assert mock_session.context_var.add_all.call_count == 2
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

    result = save_video_transcription(
        model_name=model,
        media_id=media_id,
        transcription_info=transcription_info,
        transcription=transcript,
        segments=segments,
    )
    assert "transcription_id" in result

    mock_session.context_var.add_all.assert_called_once()
    # Once for media, once for video, once for transcript
    assert mock_session.context_var.commit.call_count == 1


def test_save_transcripts_batch(mock_session):
    """Tests save transcripts function."""
    media_ids = ["0", "1", "2"]
    model = "test_model"
    texts = ("A transcript", "Another transcript", "A third transcript")
    infos = [("en", 0.5), ("de", 0.36), ("fr", 0.99)]
    transcripts = [AsrTranscription(text=text) for text in texts]
    transcription_infos = [
        AsrTranscriptionInfo(language=lang, language_confidence=conf)
        for lang, conf in infos
    ]
    segments = [
        [
            AsrSegment(
                text="",
                time_interval=TimeInterval(start=0, end=1),
                confidence=0.99,
                no_speech_confidence=0.1,
            )
        ]
        * 5
    ] * 3
    result = save_transcripts_batch(
        model, media_ids, transcription_infos, transcripts, segments
    )
    result_ids = result["transcription_ids"]

    assert (
        len(result_ids) == len(transcripts) == len(transcription_infos) == len(segments)
    )
    mock_session.context_var.add_all.assert_called_once()
    mock_session.context_var.commit.assert_called_once()


def test_save_captions_single(mock_session):
    """Tests save captions function."""
    media_id = "0"
    model_name = "test_model"
    captions = ["A caption", "Another caption", "A third caption"]
    captions_list = CaptionsList([Caption(caption) for caption in captions])
    timestamps = [0.1, 0.2, 0.3]
    frame_ids = [0, 1, 2]

    result = save_video_captions(
        model_name, media_id, captions_list, timestamps, frame_ids
    )

    assert (
        len(result["caption_ids"])
        == len(captions_list)
        == len(timestamps)
        == len(frame_ids)
    )
    mock_session.context_var.add_all.assert_called_once()
    # Once for media, once for video, once for captions
    assert mock_session.context_var.commit.call_count == 1
