# ruff: noqa: S101
from importlib import resources
from pathlib import Path

import pytest

from aana.configs.settings import settings
from aana.exceptions.general import DownloadException
from aana.models.core.video import Video
from aana.models.pydantic.video_input import VideoInput
from aana.utils.video import download_video


@pytest.fixture
def mock_download_file(mocker):
    """Mock download_file."""
    mock = mocker.patch("aana.models.core.media.download_file", autospec=True)
    path = resources.path("aana.tests.files.videos", "squirrel.mp4")
    content = path.read_bytes()
    mock.return_value = content
    return mock


def test_video(mock_download_file):
    """Test that the video can be created from path, url, or content."""
    # Test creation from a path
    try:
        path = resources.path("aana.tests.files.videos", "squirrel.mp4")
        video = Video(path=path, save_on_disk=False)
        assert video.path == path
        assert video.content is None
        assert video.url is None

        content = video.get_content()
        assert len(content) > 0
    finally:
        video.cleanup()

    # Test creation from a URL
    try:
        url = "http://example.com/squirrel.mp4"
        video = Video(url=url, save_on_disk=False)
        assert video.path is None
        assert video.content is None
        assert video.url == url

        content = video.get_content()
        assert len(content) > 0
    finally:
        video.cleanup()

    # Test creation from content
    try:
        path = resources.path("aana.tests.files.videos", "squirrel.mp4")
        content = path.read_bytes()
        video = Video(content=content, save_on_disk=False)
        assert video.path is None
        assert video.content == content
        assert video.url is None

        assert video.get_content() == content
    finally:
        video.cleanup()


def test_video_path_not_exist():
    """Test that the video can't be created from path if the path doesn't exist."""
    path = Path("path/to/video_that_does_not_exist.mp4")
    with pytest.raises(FileNotFoundError):
        Video(path=path)


def test_save_video(mock_download_file):
    """Test that save_on_disk works for video."""
    # Test that the video is saved to disk when save_on_disk is True
    try:
        path = resources.path("aana.tests.files.videos", "squirrel.mp4")
        video = Video(path=path, save_on_disk=True)
        assert video.path == path
        assert video.content is None
        assert video.url is None
        assert video.path.exists()
    finally:
        video.cleanup()
    assert video.path.exists()  # Cleanup should NOT delete the file if path is provided

    # Test saving from URL to disk
    try:
        url = "http://example.com/squirrel.mp4"
        video = Video(url=url, save_on_disk=True)
        assert video.content is None
        assert video.url == url
        assert video.path.exists()
    finally:
        video.cleanup()

    # Test saving from content to disk
    try:
        path = resources.path("aana.tests.files.videos", "squirrel.mp4")
        content = path.read_bytes()
        video = Video(content=content, save_on_disk=True)
        assert video.content == content
        assert video.url is None
        assert video.path.exists()
    finally:
        video.cleanup()


def test_cleanup(mock_download_file):
    """Test that cleanup works for video."""
    try:
        url = "http://example.com/squirrel.mp4"
        video = Video(url=url, save_on_disk=True)
        assert video.path.exists()
    finally:
        video.cleanup()
        assert not video.path.exists()

    try:
        path = resources.path("aana.tests.files.videos", "squirrel.mp4")
        video = Video(path=path, save_on_disk=True)
        assert video.path.exists()
    finally:
        video.cleanup()
        assert (
            video.path.exists()
        )  # Cleanup should NOT delete the file if path is provided


def test_at_least_one_input():
    """Test that at least one input is provided for video."""
    with pytest.raises(ValueError):
        Video(save_on_disk=False)

    with pytest.raises(ValueError):
        Video(save_on_disk=True)


def test_download_video(mock_download_file):
    """Test download_video."""
    # Test VideoInput
    path = resources.path("aana.tests.files.videos", "squirrel.mp4")
    video_input = VideoInput(path=str(path))
    video = download_video(video_input)
    assert isinstance(video, Video)
    assert video.path == path
    assert video.content is None
    assert video.url is None

    try:
        url = "http://example.com/squirrel.mp4"
        video_input = VideoInput(url=url)
        video = download_video(video_input)
        assert isinstance(video, Video)
        assert video.path is not None
        assert video.content is None
        assert video.url == url
        assert video.path.exists()
    finally:
        video.cleanup()

    # Test Youtube URL
    youtube_url = "https://www.youtube.com/watch?v=yModCU1OVHY"
    youtube_video_dir = settings.youtube_video_dir
    expected_path = youtube_video_dir / "yModCU1OVHY.mp4"
    # remove the file if it exists
    expected_path.unlink(missing_ok=True)

    try:
        youtube_video_input = VideoInput(url=youtube_url)
        video = download_video(youtube_video_input)
        assert isinstance(video, Video)
        assert video.path == expected_path
        assert video.path is not None
        assert video.path.exists()
        assert video.content is None
        assert video.url is None
    finally:
        if video and video.path:
            video.path.unlink(missing_ok=True)

    # Test YoutubeVideoInput with invalid youtube_url
    youtube_url = "https://www.youtube.com/watch?v=invalid_url"
    youtube_video_input = VideoInput(url=youtube_url)
    with pytest.raises(DownloadException):
        download_video(youtube_video_input)
