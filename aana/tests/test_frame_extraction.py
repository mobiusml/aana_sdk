import pytest
from importlib import resources
from aana.exceptions.general import VideoReadingException
from aana.models.core.image import Image
from aana.models.core.video import Video
from aana.models.pydantic.video_params import VideoParams
from aana.utils.video import extract_frames_decord


@pytest.mark.parametrize(
    "video_name, extract_fps, fast_mode_enabled, expected_duration, expected_num_frames",
    [
        ("squirrel.mp4", 1.0, False, 10.0, 10),
        ("squirrel.mp4", 0.5, False, 10.0, 5),
        ("squirrel.mp4", 1.0, True, 10.0, 4),
    ],
)
def test_extract_frames_success(
    video_name, extract_fps, fast_mode_enabled, expected_duration, expected_num_frames
):
    """
    Test that frames can be extracted from a video successfully.
    """
    video_path = resources.path("aana.tests.files.videos", video_name)
    video = Video(path=video_path)
    params = VideoParams(extract_fps=extract_fps, fast_mode_enabled=fast_mode_enabled)
    result = extract_frames_decord(video=video, params=params)
    assert "frames" in result
    assert "timestamps" in result
    assert "duration" in result
    assert isinstance(result["duration"], float)
    assert isinstance(result["frames"], list)
    assert isinstance(result["frames"][0], Image)
    assert result["duration"] == expected_duration
    assert len(result["frames"]) == expected_num_frames
    assert (
        len(result["frames"]) == len(result["timestamps"]) - 1
    )  # Minus 1 because the duration is added as the last timestamp


def test_extract_frames_failure():
    """
    Test that frames cannot be extracted from a non-existent video.
    """
    # image file instead of video file will create Video object
    # but will fail in extract_frames_decord
    path = resources.path("aana.tests.files.images", "Starry_Night.jpeg")
    invalid_video = Video(path=path)
    params = VideoParams(extract_fps=1.0, fast_mode_enabled=False)
    with pytest.raises(VideoReadingException):
        extract_frames_decord(video=invalid_video, params=params)
