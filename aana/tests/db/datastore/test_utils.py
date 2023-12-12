# ruff: noqa: S101
import pytest
from sqlalchemy.orm import Session

from aana.models.db import MediaType
from aana.models.pydantic.asr_output import (
    AsrSegments,
    AsrSegmentsList,
    AsrTranscription,
    AsrTranscriptionInfo,
    AsrTranscriptionInfoList,
    AsrTranscriptionList,
)
from aana.models.pydantic.captions import Caption, CaptionsList
from aana.utils.db import save_captions_batch, save_media, save_transcripts_batch


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


def test_save_media(mock_session):
    """Tests save media function."""
    media_type = MediaType.VIDEO
    duration = 0.5

    media_id = save_media(media_type, duration)

    assert media_id is None
    mock_session.context_var.add.assert_called_once()
    mock_session.context_var.commit.assert_called_once()


def test_save_transcripts(mock_session):
    """Tests save transcripts function."""
    media_ids = ["0"]
    model = "test_model"
    texts = ("A transcript", "Another transcript", "A third transcript")
    infos = [("en", 0.5), ("de", 0.36), ("fr", 0.99)]
    transcripts = [
        AsrTranscriptionList(__root__=[AsrTranscription(text=text) for text in texts])
    ]
    transcription_infos = [
        AsrTranscriptionInfoList(
            __root__=[
                AsrTranscriptionInfo(language=lang, language_confidence=conf)
                for lang, conf in infos
            ]
        )
    ]
    segments = [AsrSegmentsList(__root__=[AsrSegments(__root__=[])] * 3)]
    result = save_transcripts_batch(
        model, media_ids, transcription_infos, transcripts, segments
    )
    result_ids = result["transcript_ids"]

    assert (
        len(result_ids)
        == len(transcripts[0])
        == len(transcription_infos[0])
        == len(segments[0])
    )
    mock_session.context_var.add_all.assert_called_once()
    mock_session.context_var.commit.assert_called_once()


def test_save_captions(mock_session):
    """Tests save captions function."""
    media_ids = ["0"]
    models = "test_model"
    captions = ["A caption", "Another caption", "A third caption"]
    captions_list = [
        CaptionsList(__root__=[Caption(__root__=caption) for caption in captions])
    ]
    timestamps = [[0.1, 0.2, 0.3, 0.4]]
    frame_ids = [[0, 1, 2]]

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
