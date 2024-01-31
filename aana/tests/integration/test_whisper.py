# ruff: noqa: S101
# Test whisper endpoints.

from importlib import resources

import pytest

from aana.tests.utils import is_gpu_available

TARGET = "whisper"

VIDEO_TRANSCRIBE_ENDPOINT = "/video/transcribe"
VIDEO_GET_TRANSCRIPTION_ENDPOINT = "/video/get_transcription"
VIDEO_DELETE_ENDPOINT = "/video/delete"


@pytest.mark.skipif(not is_gpu_available(), reason="GPU is not available")
@pytest.mark.parametrize("call_endpoint", [TARGET], indirect=True)
@pytest.mark.parametrize(
    "video",
    [
        {
            "path": str(resources.path("aana.tests.files.videos", "physicsworks.webm")),
            "media_id": "physicsworks.webm",
        }
    ],
)
def test_video_transcribe(call_endpoint, video):
    """Test video transcribe endpoint."""
    media_id = video["media_id"]

    # transcribe video
    call_endpoint(
        VIDEO_TRANSCRIBE_ENDPOINT,
        {"video": video},
    )

    # load transcription
    call_endpoint(
        VIDEO_GET_TRANSCRIPTION_ENDPOINT,
        {"media_id": media_id},
    )

    # try to transcribe video again, it should fail with MediaIdAlreadyExistsException
    call_endpoint(
        VIDEO_TRANSCRIBE_ENDPOINT,
        {"video": video},
        expected_error="MediaIdAlreadyExistsException",
    )

    # delete video
    call_endpoint(
        VIDEO_DELETE_ENDPOINT,
        {"media_id": media_id},
        ignore_expected_output=True,
    )

    # try to load transcription again, it should fail with NotFoundException
    call_endpoint(
        VIDEO_GET_TRANSCRIPTION_ENDPOINT,
        {"media_id": media_id},
        expected_error="NotFoundException",
    )

    # try to delete video again, it should fail with NotFoundException
    call_endpoint(
        VIDEO_DELETE_ENDPOINT,
        {"media_id": media_id},
        expected_error="NotFoundException",
    )

    # transcribe video again after deleting it
    call_endpoint(
        VIDEO_TRANSCRIBE_ENDPOINT,
        {"video": video},
    )
