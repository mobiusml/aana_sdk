# ruff: noqa: S101
# Test whisper endpoints.

from importlib import resources

import pytest

from aana.tests.utils import is_gpu_available, is_using_deployment_cache

TARGET = "whisper"

VIDEO_TRANSCRIBE_ENDPOINT = "/video/transcribe"
VIDEO_GET_TRANSCRIPTION_ENDPOINT = "/video/get_transcription"
VIDEO_DELETE_ENDPOINT = "/video/delete"
VIDEO_TRANSCRIBE_BATCH_ENDPOINT = "/video/transcribe_batch"
#TODO: VIDEO_GET_TRANSCRIPTION_BATCH_ENDPOINT = "/video/get_transcription_batch"
# Note: expected transcription varies between transcribe and transcribe_batch


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.parametrize("call_endpoint", [TARGET], indirect=True)
@pytest.mark.parametrize(
    "video",
    [
        {
            "path": str(resources.path("aana.tests.files.videos", "physicsworks.webm")),
            "media_id": "physicsworks.webm",
        }
    ],
)  # TODO: Should work with audio as well.
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

    # batch transcribe video: fails with media_id exists: 
    call_endpoint(
        VIDEO_TRANSCRIBE_BATCH_ENDPOINT,
        {"video": video},
        expected_error="MediaIdAlreadyExistsException"
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

    # batch transcribe video: no error expected
    call_endpoint(
        VIDEO_TRANSCRIBE_BATCH_ENDPOINT,
        {"video": video},
    )

    # Expected is normal transcription and what you get is batched_transcription 
    # (because both of them has same media_id while storing in db).
    # There will be a run time Assertion Error for below call 
    # (but bypassing it via ignore_expected_output=True ftb) 
    # TODO: This can be avoided if we have a separate endpoint to get batched transcription.

    call_endpoint(
        VIDEO_GET_TRANSCRIPTION_ENDPOINT,
        {"media_id": media_id},
        ignore_expected_output=True,
    )

    # delete video batched transcription
    call_endpoint(
        VIDEO_DELETE_ENDPOINT,
        {"media_id": media_id},
        ignore_expected_output=True,
    )

    # TODO: End point to Load batched transcription: 
    #call_endpoint(
    #    VIDEO_GET_TRANSCRIPTION_BATCH_ENDPOINT,
    #    {"media_id": media_id},
    #    expected_error="NotFoundException",

    #)

    # try to transcribe video again, it should work since we delete the transcription by media_id
    call_endpoint(
        VIDEO_TRANSCRIBE_ENDPOINT,
        {"video": video},
    )

