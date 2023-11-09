from importlib import resources
from pathlib import Path
import pytest
from aana.models.core.video import Video


@pytest.fixture
def mock_download_file(mocker):
    """
    Mock download_file.
    """
    mock = mocker.patch("aana.models.core.media.download_file", autospec=True)
    path = resources.path("aana.tests.files.videos", "squirrel.mp4")
    content = path.read_bytes()
    mock.return_value = content
    return mock


def test_video(mock_download_file):
    """
    Test that the video can be created from path, url, or content.
    """
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
    """
    Test that the video can't be created from path if the path doesn't exist.
    """
    path = Path("path/to/video_that_does_not_exist.mp4")
    with pytest.raises(FileNotFoundError):
        Video(path=path)


def test_save_video(mock_download_file):
    """
    Test that save_on_disk works for video.
    """
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
    """
    Test that cleanup works for video.
    """
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
    """
    Test that at least one input is provided for video.
    """
    with pytest.raises(ValueError):
        Video(save_on_disk=False)

    with pytest.raises(ValueError):
        Video(save_on_disk=True)
