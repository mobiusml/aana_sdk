# ruff: noqa: S101
from importlib import resources

import pytest

from aana.core.models.image import Image
from aana.core.models.stream import StreamInput
from aana.core.models.video import Video, VideoParams
from aana.exceptions.io import StreamReadingException, VideoReadingException
from aana.integrations.external.av import fetch_stream_frames
from aana.integrations.external.decord import extract_frames, generate_frames


@pytest.mark.parametrize(
    "video_name, extract_fps, fast_mode_enabled, expected_duration, expected_num_frames",
    [
        ("squirrel.mp4", 1.0, False, 10.0, 10),
        ("squirrel.mp4", 0.5, False, 10.0, 5),
        ("squirrel.mp4", 1.0, True, 10.0, 4),
        ("physicsworks_audio.webm", 1.0, False, 203.0, 0),
    ],
)
def test_extract_frames_success(
    video_name, extract_fps, fast_mode_enabled, expected_duration, expected_num_frames
):
    """Test that frames can be extracted from a video successfully."""
    video_path = resources.path("aana.tests.files.videos", video_name)
    video = Video(path=video_path)
    params = VideoParams(extract_fps=extract_fps, fast_mode_enabled=fast_mode_enabled)
    result = extract_frames(video=video, params=params)
    assert "frames" in result
    assert "timestamps" in result
    assert "duration" in result
    assert isinstance(result["duration"], float)
    assert isinstance(result["frames"], list)
    if expected_num_frames == 0:
        assert result["frames"] == []
    else:
        assert isinstance(result["frames"][0], Image)
    assert result["duration"] == expected_duration
    assert len(result["frames"]) == expected_num_frames
    assert len(result["timestamps"]) == expected_num_frames


@pytest.mark.parametrize(
    "video_name, extract_fps, fast_mode_enabled, expected_duration, expected_num_frames",
    [
        ("squirrel.mp4", 1.0, False, 10.0, 10),
        ("squirrel.mp4", 0.5, False, 10.0, 5),
        ("squirrel.mp4", 1.0, True, 10.0, 4),
        ("physicsworks_audio.webm", 1.0, False, 203.0, 0),
    ],
)
def test_generate_frames_success(
    video_name, extract_fps, fast_mode_enabled, expected_duration, expected_num_frames
):
    """Test generate_frames.

    generate_frames is a generator function that yields a dictionary
    containing the frames, timestamps and duration of the video.
    """
    video_path = resources.path("aana.tests.files.videos", video_name)
    video = Video(path=video_path)
    params = VideoParams(extract_fps=extract_fps, fast_mode_enabled=fast_mode_enabled)
    gen_frame = generate_frames(video=video, params=params, batch_size=1)
    total_frames = 0
    for result in gen_frame:
        assert "frames" in result
        assert "timestamps" in result
        assert "duration" in result
        assert isinstance(result["duration"], float)
        assert isinstance(result["frames"], list)
        if expected_num_frames == 0:
            assert result["frames"] == []
        else:
            assert isinstance(result["frames"][0], Image)
            assert len(result["frames"]) == 1  # batch_size = 1
            assert len(result["timestamps"]) == 1  # batch_size = 1
            total_frames += 1
        assert result["duration"] == expected_duration

    assert total_frames == expected_num_frames


def test_extract_frames_failure():
    """Test that frames cannot be extracted from a non-existent video."""
    # image file instead of video file will create Video object
    # but will fail in extract_frames
    path = resources.path("aana.tests.files.images", "Starry_Night.jpeg")
    with pytest.raises(VideoReadingException):
        invalid_video = Video(path=path)
        params = VideoParams(extract_fps=1.0, fast_mode_enabled=False)
        extract_frames(video=invalid_video, params=params)


@pytest.mark.parametrize(
    "mode, url, channel_number, extract_fps",
    [
        (
            "hls",
            "https://live-par-2-cdn-alt.livepush.io/live/bigbuckbunnyclip/index.m3u8",
            0,
            3,
        ),
        (
            "dash",
            "https://live-par-2-cdn-alt.livepush.io/live/bigbuckbunnyclip/index.mpd",
            0,
            3,
        ),
        (
            "mp4",
            "https://live-par-2-abr.livepush.io/vod/bigbuckbunnyclip.mp4",
            0,
            3,
        ),
    ],
)
def test_fetch_stream_frames(mode, url, channel_number, extract_fps):
    """Test fetch_stream_frames.

    fetch_stream_frames is a generator function that yields a dictionary
    containing the frames, timestamps and frame_ids of the stream.
    """
    stream_input = StreamInput(
        url=url, channel_number=channel_number, extract_fps=extract_fps
    )
    gen_frame = fetch_stream_frames(stream_input, batch_size=1)
    total_frames = 0
    for result in gen_frame:
        assert "frames" in result
        assert "frame_ids" in result
        assert "timestamps" in result
        assert isinstance(result["frames"], list)
        assert isinstance(result["frame_ids"], list)
        assert isinstance(result["timestamps"], list)

        assert isinstance(result["frames"][0], Image)
        assert len(result["frames"]) == 1  # batch_size = 1
        assert len(result["timestamps"]) == 1  # batch_size = 1

        total_frames += 1
        if total_frames > 10:
            return
    print(f"{mode} is supported")


def test_fetch_stream_frames_failure():
    """Test that frames cannot be extracted from a youtube video."""
    url = "https://www.youtube.com/watch?v=T98dnE2vPdY"
    stream_input = StreamInput(url=url, channel_number=0, extract_fps=3)
    with pytest.raises(StreamReadingException):
        gen_frame = fetch_stream_frames(stream_input, batch_size=1)
        for _ in gen_frame:
            return
