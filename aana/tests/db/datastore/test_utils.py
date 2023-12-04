# ruff: noqa: S101
import pytest
from sqlalchemy.orm import Session

from aana.models.db import MediaType
from aana.models.pydantic.asr_output import AsrTranscription, AsrTranscriptionList
from aana.models.pydantic.captions import Caption, CaptionsList
from aana.utils.db import save_captions, save_media, save_transcripts


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
    media_id = 0
    transcripts_list = AsrTranscriptionList.construct()
    transcripts_list.__root__ = []
    for text in ("A transcript", "Another transcript", "A third transcript"):
        tt = AsrTranscription.construct()
        tt.text = text
        transcripts_list.__root__.append(tt)

    ids = save_transcripts(media_id, transcripts_list)

    assert len(ids) == len(transcripts_list)
    mock_session.context_var.add_all.assert_called_once()
    mock_session.context_var.commit.assert_called_once()


def test_save_captions(mock_session):
    """Tests save captions function."""
    media_id = 0
    captions_list = CaptionsList.construct()
    captions_list.__root__ = []
    for caption in ["A caption", "Another caption", "A third caption"]:
        c = Caption.construct()
        c.__root__ = caption
        captions_list.__root__.append(c)

    ids = save_captions(media_id, captions_list)

    assert len(ids) == len(captions_list)
    mock_session.context_var.add_all.assert_called_once()
    mock_session.context_var.commit.assert_called_once()
