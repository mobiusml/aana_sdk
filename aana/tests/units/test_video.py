# ruff: noqa: S101
import hashlib
from importlib import resources
from pathlib import Path

import pytest

from aana.configs.settings import settings
from aana.core.models.video import Video, VideoInput
from aana.exceptions.io import DownloadException, VideoReadingException
from aana.integrations.external.yt_dlp import download_video


def test_video():
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
        url = "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4"
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


def test_media_dir():
    """Test that the media_dir is set correctly."""
    # Test saving from URL to disk
    video_dir = settings.video_dir
    try:
        url = "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4"
        video = Video(url=url, save_on_disk=True)
        assert video.media_dir == video_dir
        assert video.content is None
        assert video.url == url
        assert video.path.exists()
        assert str(video.path).startswith(str(video_dir))
    finally:
        video.cleanup()


def test_video_path_not_exist():
    """Test that the video can't be created from path if the path doesn't exist."""
    path = Path("path/to/video_that_does_not_exist.mp4")
    with pytest.raises(FileNotFoundError):
        Video(path=path)


def test_save_video():
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
        url = "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4"
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


def test_cleanup():
    """Test that cleanup works for video."""
    try:
        url = "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4"
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


def test_download_video():
    """Test download_video."""
    # Test VideoInput with path
    path = resources.path("aana.tests.files.videos", "squirrel.mp4")
    video_input = VideoInput(path=str(path))
    video = download_video(video_input)
    assert isinstance(video, Video)
    assert video.path == path
    assert video.content is None
    assert video.url is None

    # Test VideoInput with url
    try:
        url = "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4"
        video_input = VideoInput(url=url)
        video = download_video(video_input)
        assert isinstance(video, Video)
        assert video.path is not None
        assert video.content is None
        assert video.url == url
        assert video.path.exists()
        assert video.media_id == video_input.media_id
    finally:
        video.cleanup()

    # Test Youtube URL
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    youtube_url_hash = hashlib.md5(
        youtube_url.encode(), usedforsecurity=False
    ).hexdigest()
    video_dir = settings.video_dir
    expected_path = video_dir / f"{youtube_url_hash}.webm"
    # remove the file if it exists
    expected_path.unlink(missing_ok=True)

    try:
        youtube_video_input = VideoInput(url=youtube_url, media_id="dQw4w9WgXcQ")
        video = download_video(youtube_video_input)
        assert isinstance(video, Video)
        assert video.path == expected_path
        assert video.path is not None
        assert video.path.exists()
        assert video.content is None
        assert video.url == youtube_url
        assert video.media_id == "dQw4w9WgXcQ"
        assert (
            video.title
            == "Rick Astley - Never Gonna Give You Up (Official Music Video)"
        )
        assert video.description.startswith(
            "The official video for “Never Gonna Give You Up” by Rick Astley."
        )
    finally:
        if video and video.path:
            video.path.unlink(missing_ok=True)

    # Test YoutubeVideoInput with invalid youtube_url
    youtube_url = "https://www.youtube.com/watch?v=invalid_url"
    youtube_video_input = VideoInput(url=youtube_url)
    with pytest.raises(DownloadException):
        download_video(youtube_video_input)

    # Test url that doesn't contain a video
    url = "https://mobius-public.s3.eu-west-1.amazonaws.com/Starry_Night.jpeg"
    video_input = VideoInput(url=url)
    with pytest.raises(VideoReadingException):
        download_video(video_input)

    # Test Video object as input
    path = resources.path("aana.tests.files.videos", "squirrel.mp4")
    video = Video(path=path)
    downloaded_video = download_video(video)
    assert downloaded_video == video
