# ruff: noqa: S101
from importlib import resources
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from aana.models.core.video import Video
from aana.models.pydantic.asr_output import (
    AsrSegments,
    AsrSegmentsList,
    AsrTranscription,
    AsrTranscriptionInfo,
    AsrTranscriptionInfoList,
    AsrTranscriptionList,
)
from aana.models.pydantic.captions import Caption, CaptionsList
from aana.utils.db import (
    save_captions_batch,
    save_transcripts_batch,
    save_video_batch,
    save_video_captions,
    save_video_single,
    save_video_transcripts,
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
    mocker.patch("aana.utils.db.Session", session_mock)
    return session_mock


def test_save_video(mock_session):
    """Tests save media function."""
    media_id = "foobar"
    path = resources.path("aana.tests.files.videos", "squirrel.mp4")
    video = Video(path=path, media_id=media_id)
    result = save_video_single(video)

    assert result["media_id"] == media_id
    assert result["video_id"] is None
    # once each for MediaEntity and VideoEntity
    assert mock_session.context_var.add.call_count == 2
    assert mock_session.context_var.commit.call_count == 2


def test_save_videos_batch(mock_session):
    """Tests save media function."""
    media_ids = ["foo", "bar"]
    path = resources.path("aana.tests.files.videos", "squirrel.mp4")
    videos = [Video(path=path, media_id=m_id) for m_id in media_ids]

    result = save_video_batch(videos)

    assert result["media_ids"] == media_ids
    assert result["video_ids"] == [None, None]
    assert len(result["media_ids"]) == len(result["video_ids"])
    # once each for MediaEntities and VideoEntities
    assert mock_session.context_var.add_all.call_count == 2
    assert mock_session.context_var.commit.call_count == 2


def test_save_transcripts_batch(mock_session):
    """Tests save transcripts function."""
    media_ids = ["0", "1"]
    model = "test_model"
    texts = ("A transcript", "Another transcript", "A third transcript")
    infos = [("en", 0.5), ("de", 0.36), ("fr", 0.99)]
    transcripts = [
        AsrTranscriptionList(__root__=[AsrTranscription(text=text) for text in texts])
    ] * 2
    transcription_infos = [
        AsrTranscriptionInfoList(
            __root__=[
                AsrTranscriptionInfo(language=lang, language_confidence=conf)
                for lang, conf in infos
            ]
        )
    ] * 2
    segments = [AsrSegmentsList(__root__=[AsrSegments(__root__=[])] * 3)] * 2
    video_ids = [0, 1]
    result = save_transcripts_batch(
        model, media_ids, video_ids, transcription_infos, transcripts, segments
    )
    result_ids = result["transcript_ids"]

    assert (
        len(result_ids)
        == len(transcripts[0]) + len(transcripts[1])
        == len(transcription_infos[0]) + len(transcription_infos[1])
        == len(segments[0]) + len(segments[1])
    )
    mock_session.context_var.add_all.assert_called_once()
    mock_session.context_var.commit.assert_called_once()


def test_save_transcripts_single(mock_session):
    """Tests save transcripts function."""
    media_id = "0"
    video_id = 0
    model = "test_model"
    texts = ("A transcript", "Another transcript", "A third transcript")
    infos = [("en", 0.5), ("de", 0.36), ("fr", 0.99)]
    transcripts = AsrTranscriptionList(
        __root__=[AsrTranscription(text=text) for text in texts]
    )
    transcription_infos = AsrTranscriptionInfoList(
        __root__=[
            AsrTranscriptionInfo(language=lang, language_confidence=conf)
            for lang, conf in infos
        ]
    )
    segments = AsrSegmentsList(__root__=[AsrSegments(__root__=[])] * 3)
    result = save_video_transcripts(
        model, media_id, video_id, transcription_infos, transcripts, segments
    )
    result_ids = result["transcript_ids"]

    assert (
        len(result_ids) == len(transcripts) == len(transcription_infos) == len(segments)
    )
    mock_session.context_var.add_all.assert_called_once()
    mock_session.context_var.commit.assert_called_once()


def test_save_captions_batch(mock_session):
    """Tests save captions function."""
    media_ids = ["0"]
    models = "test_model"
    captions = ["A caption", "Another caption", "A third caption"]
    captions_list = [
        CaptionsList(__root__=[Caption(__root__=caption) for caption in captions])
    ]
    timestamps = [[0.1, 0.2, 0.3, 0.4]]
    frame_ids = [[0, 1, 2]]
    with pytest.raises(NotImplementedError):
        result = save_captions_batch(
            media_ids, models, captions_list, timestamps, frame_ids
        )

        assert (
            len(result["caption_ids"])
            == len(captions_list[0])
            == len(timestamps[0][:-1])
            == len(frame_ids[0])
        )
        mock_session.context_var.add_all.assert_called_once()
        mock_session.context_var.commit.assert_called_once()


def test_save_captions_single(mock_session):
    """Tests save captions function."""
    media_id = "0"
    video_id = 0
    model_name = "test_model"
    captions = ["A caption", "Another caption", "A third caption"]
    captions_list = CaptionsList(
        __root__=[Caption(__root__=caption) for caption in captions]
    )
    timestamps = [0.1, 0.2, 0.3]
    frame_ids = [0, 1, 2]

    result = save_video_captions(
        model_name, media_id, video_id, captions_list, timestamps, frame_ids
    )

    assert (
        len(result["caption_ids"])
        == len(captions_list)
        == len(timestamps)
        == len(frame_ids)
    )
    mock_session.context_var.add_all.assert_called_once()
    mock_session.context_var.commit.assert_called_once()