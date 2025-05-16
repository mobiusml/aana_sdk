# ruff: noqa: S101
from importlib import resources
from pathlib import Path

import pytest
from pydantic import ValidationError

from aana.core.models.video import VideoInput, VideoInputList
from aana.exceptions.runtime import UploadedFileNotFound


@pytest.fixture
def mock_download_file(mocker):
    """Mock download_file."""
    mock = mocker.patch("aana.core.models.media.download_file", autospec=True)
    path = resources.files("aana.tests.files.videos") / "squirrel.mp4"
    content = path.read_bytes()
    mock.return_value = content
    return mock


def test_new_videoinput_success():
    """Test that VideoInput can be created successfully."""
    video_input = VideoInput(path="video.mp4")
    assert video_input.path == "video.mp4"

    video_input = VideoInput(url="http://example.com/video.mp4")
    assert video_input.url == "http://example.com/video.mp4"

    video_input = VideoInput(content="file")
    assert video_input.content == "file"


def test_videoinput_invalid_media_id():
    """Test that VideoInput can't be created if media_id is invalid."""
    with pytest.raises(ValidationError):
        VideoInput(path="video.mp4", media_id="")


@pytest.mark.parametrize(
    "url",
    [
        "domain",
        "domain.com",
        "http://",
        "www.domain.com",
        "/subdomain",
        "../subdomain",
        "",
    ],
)
def test_videoinput_invalid_url(url):
    """Test that VideoInput can't be created if url is invalid."""
    with pytest.raises(ValidationError):
        VideoInput(url=url)


def test_videoinput_check_only_one_field():
    """Test that exactly one of 'path', 'url', or 'content' is provided."""
    fields = {
        "path": "video.mp4",
        "url": "http://example.com/video.mp4",
        "content": b"file",
    }

    # check all combinations of two fields
    for field1 in fields:
        for field2 in fields:
            if field1 != field2:
                with pytest.raises(ValidationError):
                    VideoInput(**{field1: fields[field1], field2: fields[field2]})

    # check all combinations of three fields
    with pytest.raises(ValidationError):
        VideoInput(**fields)

    # check that no fields is also invalid
    with pytest.raises(ValidationError):
        VideoInput()


def test_videoinput_set_files():
    """Test that the files can be set for the video."""
    files = {"file": b"video data"}
    video_input = VideoInput(content="file")
    video_input.set_files(files)
    assert video_input.content == "file"
    assert video_input._file == files["file"]

    # If 'content' is not set to 'file',
    # the file will be ignored.
    video_input = VideoInput(path="video.mp4")
    video_input.set_files(files)

    files = {"file": b"video data", "another_file": b"another video data"}
    video_input = VideoInput(content="file")
    video_input.set_files(files)
    assert video_input.content == "file"
    assert video_input._file == files["file"]

    files = {"file": b"video data"}
    video_input = VideoInput(content="unknown_file")
    with pytest.raises(UploadedFileNotFound):
        video_input.set_files(files)


def test_videoinput_convert_input_to_object(mock_download_file):
    """Test that VideoInput can be converted to Video."""
    path = resources.files("aana.tests.files.videos") / "squirrel.mp4"
    video_input = VideoInput(path=str(path))
    try:
        video_object = video_input.convert_input_to_object()
        assert video_object.path == path
    finally:
        video_object.cleanup()

    url = "http://example.com/squirrel.mp4"
    video_input = VideoInput(url=url)
    try:
        video_object = video_input.convert_input_to_object()
        assert video_object.url == url
    finally:
        video_object.cleanup()

    content = Path(path).read_bytes()
    files = {"file": content}
    video_input = VideoInput(content="file")
    video_input.set_files(files)
    try:
        video_object = video_input.convert_input_to_object()
        assert video_object.content == content
    finally:
        video_object.cleanup()


def test_videoinputlist():
    """Test that VideoInputList can be created successfully."""
    videos = [
        VideoInput(path="video.mp4"),
        VideoInput(url="http://example.com/video.mp4"),
        VideoInput(content=b"file"),
    ]

    video_list_input = VideoInputList(root=videos)
    assert video_list_input.root == videos
    assert len(video_list_input) == len(videos)
    assert video_list_input[0] == videos[0]
    assert video_list_input[1] == videos[1]
    assert video_list_input[2] == videos[2]


def test_videoinputlist_set_files():
    """Test that the files can be set for the video list."""
    files = {
        "file": b"video data",
        "another_file": b"another video data",
    }

    videos = [
        VideoInput(content="file"),
        VideoInput(content="another_file"),
    ]

    video_list_input = VideoInputList(root=videos)
    video_list_input.set_files(files)
    assert video_list_input[0].content == "file"
    assert video_list_input[1].content == "another_file"
    assert video_list_input[0]._file == files["file"]
    assert video_list_input[1]._file == files["another_file"]

    # If there are more files than videos,
    # the extra files will be ignored.
    files = {
        "file": b"video data",
        "another_file": b"another video data",
        "yet_another_file": b"yet another video data",
    }
    video_list_input = VideoInputList(root=videos)
    video_list_input.set_files(files)

    # If files are missing, an error should be raised.
    files = []
    video_list_input = VideoInputList(root=videos)
    with pytest.raises(UploadedFileNotFound):
        video_list_input.set_files(files)


def test_videoinputlist_non_empty():
    """Test that videoinputlist must not be empty."""
    with pytest.raises(ValidationError):
        VideoInputList(root=[])


def test_video_input_extra_fields():
    """Test that extra fields are not allowed in VideoInput."""
    with pytest.raises(ValueError):
        VideoInput(
            url="https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4",
            extra_field="extra_value",
        )
