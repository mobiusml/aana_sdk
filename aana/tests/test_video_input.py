from importlib import resources
from pathlib import Path
import pytest
from pydantic import ValidationError
from aana.models.pydantic.video_input import (
    VideoInput,
    VideoInputList,
    VideoOrYoutubeVideoInputList,
    YoutubeVideoInput,
)


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


def test_new_videoinput_success():
    """
    Test that VideoInput can be created successfully.
    """
    video_input = VideoInput(path="video.mp4")
    assert video_input.path == "video.mp4"

    video_input = VideoInput(url="http://example.com/video.mp4")
    assert video_input.url == "http://example.com/video.mp4"

    video_input = VideoInput(content=b"file")
    assert video_input.content == b"file"


def test_videoinput_check_only_one_field():
    """
    Test that exactly one of 'path', 'url', or 'content' is provided.
    """
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


def test_videoinput_set_file():
    """
    Test that the file can be set for the video.
    """
    file_content = b"video data"
    video_input = VideoInput(content=b"file")
    video_input.set_file(file_content)
    assert video_input.content == file_content

    # If 'content' is not set to 'file',
    # an error should be raised.
    video_input = VideoInput(path="video.mp4")
    with pytest.raises(ValueError):
        video_input.set_file(file_content)


def test_videoinput_set_files():
    """
    Test that the files can be set for the video.
    """
    files = [b"video data"]

    video_input = VideoInput(content=b"file")
    video_input.set_files(files)
    assert video_input.content == files[0]

    # If 'content' is not set to 'file',
    # an error should be raised.
    video_input = VideoInput(path="video.mp4")
    with pytest.raises(ValueError):
        video_input.set_files(files)

    # If the number of files is not 1,
    # an error should be raised.
    files = [b"video data", b"another video data"]
    video_input = VideoInput(content=b"file")
    with pytest.raises(ValidationError):
        video_input.set_files(files)

    files = []
    video_input = VideoInput(content=b"file")
    with pytest.raises(ValidationError):
        video_input.set_files(files)


def test_videoinput_convert_input_to_object(mock_download_file):
    """
    Test that VideoInput can be converted to Video.
    """
    path = resources.path("aana.tests.files.videos", "squirrel.mp4")
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
    video_input = VideoInput(content=content)
    try:
        video_object = video_input.convert_input_to_object()
        assert video_object.content == content
    finally:
        video_object.cleanup()


def test_videolistinput():
    """
    Test that VideoListInput can be created successfully.
    """
    videos = [
        VideoInput(path="video.mp4"),
        VideoInput(url="http://example.com/video.mp4"),
        VideoInput(content=b"file"),
    ]

    video_list_input = VideoInputList(__root__=videos)
    assert video_list_input.__root__ == videos
    assert len(video_list_input) == len(videos)
    assert video_list_input[0] == videos[0]
    assert video_list_input[1] == videos[1]
    assert video_list_input[2] == videos[2]


def test_videolistinput_set_files():
    """
    Test that the files can be set for the video list.
    """
    files = [b"video data", b"another video data"]

    videos = [
        VideoInput(content=b"file"),
        VideoInput(content=b"file"),
    ]

    video_list_input = VideoInputList(__root__=videos)
    video_list_input.set_files(files)
    assert video_list_input[0].content == files[0]
    assert video_list_input[1].content == files[1]

    # If the number of files is not the same as the number of videos,
    # an error should be raised.
    files = [b"video data", b"another video data", b"yet another video data"]
    video_list_input = VideoInputList(__root__=videos)
    with pytest.raises(ValidationError):
        video_list_input.set_files(files)

    files = []
    video_list_input = VideoInputList(__root__=videos)
    with pytest.raises(ValidationError):
        video_list_input.set_files(files)


def test_videolistinput_non_empty():
    """
    Test that VideoListInput must not be empty.
    """
    with pytest.raises(ValidationError):
        VideoInputList(__root__=[])


def test_youtubevideoinput_success():
    """
    Test that YoutubeVideoInput can be created successfully.
    """
    youtube_input = YoutubeVideoInput(
        youtube_url="https://www.youtube.com/watch?v=yModCU1OVHY"
    )
    assert youtube_input.youtube_url == "https://www.youtube.com/watch?v=yModCU1OVHY"

    youtube_input = YoutubeVideoInput(
        youtube_url="http://www.youtube.com/watch?v=yModCU1OVHY"
    )
    assert youtube_input.youtube_url == "http://www.youtube.com/watch?v=yModCU1OVHY"

    youtube_input = YoutubeVideoInput(youtube_url="https://youtu.be/yModCU1OVHY")
    assert youtube_input.youtube_url == "https://youtu.be/yModCU1OVHY"

    youtube_input = YoutubeVideoInput(youtube_url="http://youtu.be/yModCU1OVHY")
    assert youtube_input.youtube_url == "http://youtu.be/yModCU1OVHY"


def test_youtubevideoinput_missing_url():
    """
    Test that a missing youtube URL raises a TypeError.
    """
    with pytest.raises(ValidationError):
        YoutubeVideoInput()


def test_videooryoutubevideoinputlist_success():
    """
    Test that VideoOrYoutubeVideoInputList can be created successfully.
    """
    videos = [
        VideoInput(path="video.mp4"),
        YoutubeVideoInput(youtube_url="https://www.youtube.com/watch?v=yModCU1OVHY"),
    ]

    video_list_input = VideoOrYoutubeVideoInputList(__root__=videos)
    assert video_list_input.__root__ == videos
    assert len(video_list_input) == len(videos)
    assert video_list_input[0] == videos[0]
    assert video_list_input[1] == videos[1]


def test_videooryoutubevideoinputlist_set_files():
    """
    Test that the files can be set for the video list.
    """
    files = [b"video data", b"another video data"]

    videos = [
        VideoInput(content=b"file"),
        VideoInput(content=b"file"),
    ]

    video_list_input = VideoOrYoutubeVideoInputList(__root__=videos)
    video_list_input.set_files(files)
    for i, video in enumerate(video_list_input):
        assert video.content == files[i]

    # If the number of files is not the same as the number of videos,
    # an error should be raised.
    files = [b"video data", b"another video data", b"yet another video data"]
    video_list_input = VideoOrYoutubeVideoInputList(__root__=videos)
    with pytest.raises(ValidationError):
        video_list_input.set_files(files)

    files = []
    video_list_input = VideoOrYoutubeVideoInputList(__root__=videos)
    with pytest.raises(ValidationError):
        video_list_input.set_files(files)


def test_videooryoutubevideoinputlist_non_empty():
    """
    Test that VideoOrYoutubeVideoInputList must not be empty.
    """
    with pytest.raises(ValidationError):
        VideoOrYoutubeVideoInputList(__root__=[])


def test_videooryoutubevideoinputlist_invalid_input():
    """
    Test that VideoOrYoutubeVideoInputList raises an error for invalid input.
    """
    with pytest.raises(ValidationError):
        VideoOrYoutubeVideoInputList(__root__=[1, 2, 3])


def test_videooryoutubevideoinputlist_convert_input_to_object():
    """
    Test that VideoOrYoutubeVideoInputList can be converted to a list of video inputs.
    """
    videos = [
        VideoInput(path="video.mp4"),
        YoutubeVideoInput(youtube_url="https://www.youtube.com/watch?v=yModCU1OVHY"),
    ]

    video_list_input = VideoOrYoutubeVideoInputList(__root__=videos)
    video_inputs = video_list_input.convert_input_to_object()
    assert isinstance(video_inputs, list)
    assert len(video_inputs) == len(videos)
    assert isinstance(video_inputs[0], VideoInput)
    assert isinstance(video_inputs[1], YoutubeVideoInput)
