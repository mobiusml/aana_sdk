# ruff: noqa: S101
# Test chat with video endpoints.

import hashlib
import json
from importlib import resources

import pytest

from aana.tests.utils import call_endpoint, check_output, is_gpu_available
from aana.utils.json import json_serializer_default

TARGET = "chat_with_video"


def index_video_stream(
    target: str,
    port: int,
    route_prefix: str,
    video: dict,
    ignore_expected_output: bool = False,
    expected_error: str | None = None,
) -> dict | list:
    """Index video stream."""
    endpoint_path = "/video/index_stream"
    data = {"video": video}
    data_hash = hashlib.md5(
        json.dumps(data, default=json_serializer_default).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    output = call_endpoint(target, port, route_prefix, endpoint_path, data)
    check_output(
        target, endpoint_path, data_hash, output, ignore_expected_output, expected_error
    )
    return output


def delete_video(
    target: str,
    port: int,
    route_prefix: str,
    media_id: str,
    ignore_expected_output: bool = False,
    expected_error: str | None = None,
) -> dict | list:
    """Delete video."""
    endpoint_path = "/video/delete"
    data = {"media_id": media_id}
    data_hash = hashlib.md5(
        json.dumps(data, default=json_serializer_default).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    output = call_endpoint(target, port, route_prefix, endpoint_path, data)
    check_output(
        target, endpoint_path, data_hash, output, ignore_expected_output, expected_error
    )
    return output


def load_video_metadata(
    target: str,
    port: int,
    route_prefix: str,
    video: str,
    ignore_expected_output: bool = False,
    expected_error: str | None = None,
) -> dict | list:
    """Load video metadata."""
    endpoint_path = "/video/metadata"
    data = {"media_id": video}
    data_hash = hashlib.md5(
        json.dumps(data, default=json_serializer_default).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    output = call_endpoint(target, port, route_prefix, endpoint_path, data)
    check_output(
        target, endpoint_path, data_hash, output, ignore_expected_output, expected_error
    )
    return output


def chat_with_video(
    target: str,
    port: int,
    route_prefix: str,
    media_id: str,
    question: str,
    ignore_expected_output: bool = False,
    expected_error: str | None = None,
) -> dict | list:
    """Chat with video."""
    endpoint_path = "/video/chat_stream"
    data = {"media_id": media_id, "question": question}
    data_hash = hashlib.md5(
        json.dumps(data, default=json_serializer_default).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    output = call_endpoint(target, port, route_prefix, endpoint_path, data)
    check_output(
        target, endpoint_path, data_hash, output, ignore_expected_output, expected_error
    )
    return output


@pytest.fixture(scope="module")
def app(app_setup):
    """Setup app for a specific target."""
    return app_setup(TARGET)


@pytest.mark.skipif(not is_gpu_available(), reason="GPU not available.")
@pytest.mark.parametrize(
    "video, error",
    [
        (
            {
                "url": "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4",
                "media_id": "squirrel.mp4",
            },
            None,
        ),
        (
            {
                "path": str(
                    resources.path("aana.tests.files.videos", "physicsworks.webm")
                ),
                "media_id": "physicsworks.webm",
            },
            None,
        ),
        (
            {
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "media_id": "dQw4w9WgXcQ",
            },
            None,
        ),
    ],
)
def test_chat_with_video(app, video, error):
    """Test chat with video endpoint."""

    target = TARGET
    handle, port, route_prefix = app

    media_id = video["media_id"]

    index_video_stream(target, port, route_prefix, video, expected_error=error)

    # if we try to index the same video again, we should get an error MediaIdAlreadyExistsException
    index_video_stream(
        target,
        port,
        route_prefix,
        video,
        expected_error="MediaIdAlreadyExistsException",
    )

    # load video metadata
    load_video_metadata(target, port, route_prefix, media_id, expected_error=error)

    # delete video
    delete_video(target, port, route_prefix, media_id, ignore_expected_output=True)

    # after deleting the video video metadata should not be available
    load_video_metadata(
        target,
        port,
        route_prefix,
        media_id,
        expected_error="NotFoundException",
    )

    # after deleting the video, we should be able to index it again
    index_video_stream(target, port, route_prefix, video, expected_error=error)

    # load video metadata
    load_video_metadata(target, port, route_prefix, media_id, expected_error=error)

    # chat with video
    question = "Summarize the video"

    chat_with_video(
        target,
        port,
        route_prefix,
        media_id,
        question,
        expected_error=error,
    )

    # delete video
    delete_video(target, port, route_prefix, media_id, ignore_expected_output=True)

    # after deleting the video, we should not be able to chat with it
    chat_with_video(
        target,
        port,
        route_prefix,
        media_id,
        question,
        expected_error="NotFoundException",
    )
