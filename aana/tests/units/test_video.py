# ruff: noqa: S101
import hashlib
from importlib import resources
from pathlib import Path

import pytest

from aana.configs.settings import settings
from aana.core.models.video import Video, VideoInput, VideoMetadata
from aana.exceptions.io import DownloadException, VideoReadingException
from aana.integrations.external.yt_dlp import download_video, get_video_metadata


def test_video():
    """Test that the video can be created from path, url, or content."""
    # Test creation from a path
    try:
        path = resources.files("aana.tests.files.videos") / "squirrel.mp4"
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
        path = resources.files("aana.tests.files.videos") / "squirrel.mp4"
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
        path = resources.files("aana.tests.files.videos") / "squirrel.mp4"
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
        path = resources.files("aana.tests.files.videos") / "squirrel.mp4"
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
        path = resources.files("aana.tests.files.videos") / "squirrel.mp4"
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
    path = resources.files("aana.tests.files.videos") / "squirrel.mp4"
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

    # Test url that doesn't contain a video
    url = "https://mobius-public.s3.eu-west-1.amazonaws.com/Starry_Night.jpeg"
    video_input = VideoInput(url=url)
    with pytest.raises(VideoReadingException):
        download_video(video_input)

    # Test Video object as input
    path = resources.files("aana.tests.files.videos") / "squirrel.mp4"
    video = Video(path=path)
    downloaded_video = download_video(video)
    assert downloaded_video == video


@pytest.mark.parametrize(
    "url, title, description, duration",
    [
        (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "Rick Astley - Never Gonna Give You Up (Official Music Video)",
            "The official video for “Never Gonna Give You Up” by Rick Astley.",
            212.0,
        ),
        (
            "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4",
            "squirrel",
            "",
            None,
        ),
    ],
)
def test_get_video_metadata_success(mocker, url, title, description, duration):
    """Test getting video metadata."""
    # Mock the yt-dlp YoutubeDL class and its methods
    mock_ydl = mocker.patch("yt_dlp.YoutubeDL", autospec=True)
    mock_ydl_instance = mock_ydl.return_value
    mock_ydl_instance.__enter__.return_value = mock_ydl_instance

    mock_info = {
        "title": title,
        "description": description,
        "duration": duration,
    }
    mock_ydl_instance.extract_info.return_value = mock_info

    metadata = get_video_metadata(url)
    assert isinstance(metadata, VideoMetadata)
    assert metadata.title == title
    assert metadata.description.startswith(description)
    assert metadata.duration == duration


def test_get_video_metadata_failure():
    """Test that get_video_metadata fails for invalid URLs."""
    url = "https://www.youtube.com/watch?v=invalid_url"
    with pytest.raises(DownloadException):
        get_video_metadata(url)


def test_download_youtube_video(mocker):
    """Test downloading a YouTube video."""
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    youtube_url_hash = hashlib.md5(
        youtube_url.encode(), usedforsecurity=False
    ).hexdigest()
    video_dir = settings.video_dir
    expected_path = Path(video_dir) / f"{youtube_url_hash}.mp4"

    # Mock the VideoInput object
    youtube_video_input = VideoInput(url=youtube_url, media_id="dQw4w9WgXcQ")

    # Mock the yt-dlp YoutubeDL class and its methods
    mock_ydl = mocker.patch("yt_dlp.YoutubeDL", autospec=True)
    mock_ydl_instance = mock_ydl.return_value
    mock_ydl_instance.__enter__.return_value = mock_ydl_instance

    mock_info = {
        "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)",
        "description": "The official video for “Never Gonna Give You Up” by Rick Astley.",
        "ext": "mp4",
    }
    mock_ydl_instance.extract_info.return_value = mock_info
    mock_ydl_instance.prepare_filename.return_value = str(expected_path)
    mock_ydl_instance.download.return_value = None

    # # Mock Path.exists to simulate that the file will exist after download
    mocker.patch("aana.integrations.external.yt_dlp.Path.exists", return_value=True)

    # Mock decord.VideoReader to avoid trying to read the non-existent mocked file
    mocker.patch("decord.VideoReader", autospec=True)

    video = download_video(youtube_video_input)

    assert isinstance(video, Video)
    assert video.path.with_suffix("") == expected_path.with_suffix("")
    assert video.path is not None
    assert video.path.exists()
    assert video.content is None
    assert video.url == youtube_url
    assert video.media_id == "dQw4w9WgXcQ"
    assert video.title == "Rick Astley - Never Gonna Give You Up (Official Music Video)"
    assert video.description.startswith(
        "The official video for “Never Gonna Give You Up” by Rick Astley."
    )


def test_download_youtube_video_failure():
    """Test YoutubeVideoInput with invalid youtube URL."""
    youtube_url = "https://www.youtube.com/watch?v=invalid_url"
    youtube_video_input = VideoInput(url=youtube_url)
    with pytest.raises(DownloadException):
        download_video(youtube_video_input)
