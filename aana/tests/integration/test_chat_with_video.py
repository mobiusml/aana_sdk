# ruff: noqa: S101
# Test chat with video endpoints.

from importlib import resources

import pytest

from aana.tests.utils import is_gpu_available, is_using_deployment_cache

TARGET = "chat_with_video"

VIDEO_INDEX_ENDPOINT = "/video/index_stream"
VIDEO_METADATA_ENDPOINT = "/video/metadata"
VIDEO_CHAT_ENDPOINT = "/video/chat_stream"
VIDEO_DELETE_ENDPOINT = "/video/delete"


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.parametrize("call_endpoint", [TARGET], indirect=True)
@pytest.mark.parametrize(
    "video",
    [
        {
            "url": "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4",
            "media_id": "squirrel.mp4",
        },
        {
            "path": str(resources.path("aana.tests.files.videos", "physicsworks.webm")),
            "media_id": "physicsworks.webm",
        },
        {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "media_id": "dQw4w9WgXcQ",
        },
    ],
)
def test_chat_with_video(call_endpoint, video):
    """Test chat with video endpoint."""
    media_id = video["media_id"]

    call_endpoint(
        VIDEO_INDEX_ENDPOINT,
        {"video": video},
    )

    # if we try to index the same video again, we should get an error MediaIdAlreadyExistsException
    call_endpoint(
        VIDEO_INDEX_ENDPOINT,
        {"video": video},
        expected_error="MediaIdAlreadyExistsException",
    )

    # load video metadata
    call_endpoint(
        VIDEO_METADATA_ENDPOINT,
        {"media_id": media_id},
    )

    # delete video
    call_endpoint(
        VIDEO_DELETE_ENDPOINT,
        {"media_id": media_id},
        ignore_expected_output=True,
    )

    # after deleting the video video metadata should not be available
    call_endpoint(
        VIDEO_METADATA_ENDPOINT,
        {"media_id": media_id},
        expected_error="NotFoundException",
    )

    # after deleting the video, we should be able to index it again
    call_endpoint(
        VIDEO_INDEX_ENDPOINT,
        {"video": video},
    )

    # load video metadata
    call_endpoint(
        VIDEO_METADATA_ENDPOINT,
        {"media_id": media_id},
    )

    # chat with video
    question = "Summarize the video"

    call_endpoint(
        VIDEO_CHAT_ENDPOINT,
        {"media_id": media_id, "question": question},
    )

    # delete video
    call_endpoint(
        VIDEO_DELETE_ENDPOINT,
        {"media_id": media_id},
        ignore_expected_output=True,
    )

    # after deleting the video, we should not be able to chat with it
    call_endpoint(
        VIDEO_CHAT_ENDPOINT,
        {"media_id": media_id, "question": question},
        expected_error="NotFoundException",
    )


# FIX: See https://github.com/mobiusml/aana_sdk/issues/42
# @pytest.mark.parametrize(
#     "endpoint, data",
#     [
#         (VIDEO_METADATA_ENDPOINT, {}),
#         (VIDEO_CHAT_ENDPOINT, {}),
#         (VIDEO_CHAT_ENDPOINT, {"media_id": "squirrel.mp4"}),
#         (VIDEO_CHAT_ENDPOINT, {"question": "Summarize the video"}),
#         (VIDEO_INDEX_ENDPOINT, {}),
#         (VIDEO_DELETE_ENDPOINT, {}),
#     ],
# )
# @pytest.mark.parametrize("call_endpoint", [TARGET], indirect=True)
# def test_missing_params(call_endpoint, endpoint, data):
#     """Test missing params."""
#     call_endpoint(
#         endpoint,
#         data,
#         expected_error="ValidationError",
#     )
