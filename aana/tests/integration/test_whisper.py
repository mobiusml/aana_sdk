# ruff: noqa: S101
# Test whisper endpoints.

from importlib import resources

import pytest

from aana.tests.utils import call_endpoint, check_output, is_gpu_available

TARGET = "whisper"


def transcribe(
    target: str,
    port: int,
    route_prefix: str,
    video: str,
    ignore_expected_output: bool = False,
    expected_error: str | None = None,
) -> dict | list:
    """Transcribe video."""
    endpoint_path = "/video/transcribe"
    path = resources.path("aana.tests.files.videos", video)
    data = {"video": {"path": str(path), "media_id": video}}
    output = call_endpoint(target, port, route_prefix, endpoint_path, data)
    check_output(
        target, endpoint_path, video, output, ignore_expected_output, expected_error
    )
    return output


def load_transcription(
    target: str,
    port: int,
    route_prefix: str,
    video: str,
    ignore_expected_output: bool = False,
    expected_error: str | None = None,
) -> dict | list:
    """Load transcription."""
    endpoint_path = "/video/get_transcription"
    data = {"media_id": video}
    output = call_endpoint(target, port, route_prefix, endpoint_path, data)
    check_output(
        target, endpoint_path, video, output, ignore_expected_output, expected_error
    )
    return output


def delete_video(
    target: str,
    port: int,
    route_prefix: str,
    video: str,
    ignore_expected_output: bool = False,
    expected_error: str | None = None,
) -> dict | list:
    """Delete video."""
    endpoint_path = "/video/delete"
    data = {"media_id": video}
    output = call_endpoint(target, port, route_prefix, endpoint_path, data)
    check_output(
        target, endpoint_path, video, output, ignore_expected_output, expected_error
    )
    return output


@pytest.mark.skipif(not is_gpu_available(), reason="GPU is not available")
@pytest.mark.parametrize(
    "video",
    [
        "physicsworks.webm",
    ],
)
def test_video_transcribe(video, app_setup):
    """Test video transcribe endpoint."""
    target = TARGET
    handle, port, route_prefix = app_setup(target)

    # transcribe video
    transcribe(target, port, route_prefix, video)

    # load transcription
    load_transcription(target, port, route_prefix, video)

    # try to transcribe video again, it should fail with MediaIdAlreadyExistsException
    transcribe(
        target,
        port,
        route_prefix,
        video,
        expected_error="MediaIdAlreadyExistsException",
    )

    # delete video
    delete_video(target, port, route_prefix, video, ignore_expected_output=True)

    # try to load transcription again, it should fail with NotFoundException
    load_transcription(
        target,
        port,
        route_prefix,
        video,
        expected_error="NotFoundException",
    )

    # try to delete video again, it should fail with NotFoundException
    delete_video(
        target,
        port,
        route_prefix,
        video,
        expected_error="NotFoundException",
    )

    # transcribe video again after deleting it
    transcribe(target, port, route_prefix, video)
