# ruff: noqa: S101
# Test whisper endpoints.

from importlib import resources

import pytest

from aana.tests.utils import is_gpu_available, is_using_deployment_cache

TARGET = "whisper"

VIDEO_TRANSCRIBE_ENDPOINT = "/video/transcribe"
VIDEO_GET_TRANSCRIPTION_ENDPOINT = "/video/get_transcription"
VIDEO_DELETE_ENDPOINT = "/video/delete"
VIDEO_TRANSCRIBE_BATCH_ENDPOINT = "/video/transcribe_in_chunks"


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.parametrize("call_endpoint", [TARGET], indirect=True)
@pytest.mark.parametrize(
    "video, whisper_params, transcription_endpoint",
    [
        (
            {
                "path": str(
                    resources.path("aana.tests.files.videos", "physicsworks.webm")
                ),
                "media_id": "physicsworks.webm",
            },
            {"temperature": 0.0},
            VIDEO_TRANSCRIBE_ENDPOINT,
        ),
        (
            {
                "path": str(
                    resources.path("aana.tests.files.videos", "physicsworks.webm")
                ),
                "media_id": "physicsworks.webm_batched",
            },
            {"temperature": 0.0},
            VIDEO_TRANSCRIBE_BATCH_ENDPOINT,
        ),
        (
            {
                "path": str(
                    resources.path("aana.tests.files.audios", "physicsworks.wav")
                ),
                "media_id": "physicsworks.wav_batched",
            },
            {"temperature": 0.0},
            VIDEO_TRANSCRIBE_BATCH_ENDPOINT,
        ),
        (
            {
                "path": str(
                    resources.path("aana.tests.files.audios", "physicsworks.wav")
                ),
                "media_id": "physicsworks.wav",
            },
            {"temperature": 0.0},
            VIDEO_TRANSCRIBE_ENDPOINT,
        ),
    ],
)
def test_video_transcribe(call_endpoint, video, whisper_params, transcription_endpoint):
    """Test video transcribe endpoint."""
    media_id = video["media_id"]

    # transcribe video
    call_endpoint(
        transcription_endpoint,
        {"video": video, "whisper_params": whisper_params},
    )

    # load transcription
    call_endpoint(
        VIDEO_GET_TRANSCRIPTION_ENDPOINT,
        {"media_id": media_id},
    )

    # try to transcribe video again, it should fail with MediaIdAlreadyExistsException
    call_endpoint(
        transcription_endpoint,
        {"video": video, "whisper_params": whisper_params},
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
        transcription_endpoint,
        {"video": video, "whisper_params": whisper_params},
    )

    # delete video
    call_endpoint(
        VIDEO_DELETE_ENDPOINT,
        {"media_id": media_id},
        ignore_expected_output=True,
    )

    # no found exception for deleted transcription
    call_endpoint(
        VIDEO_GET_TRANSCRIPTION_ENDPOINT,
        {"media_id": media_id},
        expected_error="NotFoundException",
    )
