# ruff: noqa: S101
import hashlib
from importlib import resources
from pathlib import Path

import pytest

from aana.models.core.audio import Audio
from aana.models.core.video import Video
from aana.models.pydantic.video_input import VideoInput
from aana.utils.video import download_video, extract_audio


def test_audio():
    """Test that the audio can be created from path, url(pending), or content."""
    # Test creation from a path
    try:
        path = resources.path("aana.tests.files.audios", "physicsworks.wav")
        audio = Audio(path=path, save_on_disk=False)
        assert audio.path == path
        assert audio.content is None
        assert audio.url is None

        content = audio.get_content()
        assert len(content) > 0
    finally:
        audio.cleanup()

    # Test creation from a URL
    # try:
    #    url = "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4"  # ask to keep audio as well
    #    audio = Audio(url=url, save_on_disk=False)
    #    assert audio.path is None
    #    assert audio.content is None
    #    assert audio.url == url

    #    content = audio.get_content()
    #    assert len(content) > 0
    # finally:
    #    audio.cleanup()

    # Test creation from content
    try:
        path = resources.path("aana.tests.files.audios", "physicsworks.wav")
        content = path.read_bytes()
        audio = Audio(content=content, save_on_disk=False)
        assert audio.path is None
        assert audio.content == content
        assert audio.url is None

        assert audio.get_content() == content
    finally:
        audio.cleanup()


# TODO: Test saving audio from URL to disk


def test_audio_path_not_exist():
    """Test that the audio can't be created from path if the path doesn't exist."""
    path = Path("path/to/audio_that_does_not_exist.mp4")
    with pytest.raises(FileNotFoundError):
        Audio(path=path)


def test_save_audio():
    """Test that save_on_disk works for audio."""
    # Test that the audio is saved to disk when save_on_disk is True
    try:
        path = resources.path("aana.tests.files.audios", "physicsworks.wav")
        audio = Audio(path=path, save_on_disk=True)
        assert audio.path == path
        assert audio.content is None
        assert audio.url is None
        assert audio.path.exists()
    finally:
        audio.cleanup()
    assert audio.path.exists()  # Cleanup should NOT delete the file if path is provided

    # TODO: Test saving from URL to disk (load audio physicsworks.wav to aws )
    # try:
    #    url = "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4"  # need to have an audio input
    #    audio = Audio(url=url, save_on_disk=True)
    #    assert audio.content is None
    #    assert audio.url == url
    #    assert audio.path.exists()
    # finally:
    #    audio.cleanup()

    # Test saving from content to disk
    try:
        path = resources.path("aana.tests.files.audios", "physicsworks.wav")
        content = path.read_bytes()
        audio = Audio(content=content, save_on_disk=True)
        assert audio.content == content
        assert audio.url is None
        assert audio.path.exists()
    finally:
        audio.cleanup()


def test_cleanup():
    """Test that cleanup works for audios."""
    # try:
    #    url = "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4"
    #    video = Video(url=url, save_on_disk=True)
    #    assert video.path.exists()
    # finally:
    #    video.cleanup()
    #    assert not video.path.exists()

    try:
        path = resources.path("aana.tests.files.audios", "physicsworks.wav")
        audio = Audio(path=path, save_on_disk=True)
        assert audio.path.exists()
    finally:
        audio.cleanup()
        assert (
            audio.path.exists()
        )  # Cleanup should NOT delete the file if path is provided


def test_at_least_one_input():
    """Test that at least one input is provided for audio."""
    with pytest.raises(ValueError):
        Audio(save_on_disk=False)

    with pytest.raises(ValueError):
        Audio(save_on_disk=True)


# TODO: some parts are missing because audio needs to be uploaded to a url to test them
def test_extract_audio():
    """Test download_video and extract_audio."""
    # Test VideoInput with video path (tests download_video and extract_audio): return audio bytes

    path = resources.path("aana.tests.files.videos", "physicsworks.webm")
    video_input = VideoInput(path=str(path))
    video = download_video(video_input)
    assert isinstance(video, Video)
    assert video.path == path
    assert video.content is None
    assert video.url is None

    audio = extract_audio(video)
    assert isinstance(audio, Audio)
    assert audio.path is not None  # temporary path created in audio_dir on save()
    assert audio.content is not None  # extract_audio returns audio_bytes
    assert audio.url is None

    # Test audio from VideoInput (for video input): return audio bytes
    path = resources.path(
        "aana.tests.files.videos", "physicsworks.webm"
    )  # ideally some other sampling rate
    video_input = VideoInput(path=str(path))
    audio = extract_audio(video_input)
    assert isinstance(audio, Audio)
    assert audio.content is not None
    assert audio.path is not None
    assert audio.url is None

    # Test audio from VideoInput (for audio input): return audio bytes
    path = resources.path(
        "aana.tests.files.audios", "physicsworks.wav"
    )  # ideally some other sampling rate
    video_input = VideoInput(path=str(path))
    audio = extract_audio(video_input)
    assert isinstance(audio, Audio)
    assert audio.content is not None
    assert audio.path is not None
    assert audio.url is None

    # Test audio from VideoInput (for no audio channel): return empty bytes
    path = resources.path("aana.tests.files.videos", "squirrel_no_audio.mp4")
    video_input = VideoInput(path=str(path))
    audio = extract_audio(video_input)
    assert isinstance(audio, Audio)
    assert audio.content == b""
    assert audio.url is None
    assert audio.path is not None

    # TODO: Test extract_audio with url (upload .wav)
    # try:
    #    url = "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4"
    #    video_input = VideoInput(url=url)
    #    video = download_video(video_input)
    #    assert isinstance(video, Video)
    #    assert video.path is not None
    #    assert video.content is None
    #    assert video.url == url
    #    assert video.path.exists()
    #    assert video.media_id == video_input.media_id
    # finally:
    #    video.cleanup()

    # Test content of Audio objects created in different ways from same source video is the same.
    video_path = resources.path("aana.tests.files.videos", "physicsworks.webm")
    video = Video(path=video_path)
    extracted_audio_1 = extract_audio(video)

    video_input = VideoInput(path=str(video_path))
    downloaded_video = download_video(video_input)
    extracted_audio_2 = extract_audio(downloaded_video)

    # paths will be different since Audio object is created from two different Video objects,
    # but the audio content should be the same
    assert extracted_audio_1.get_numpy().all() == extracted_audio_2.get_numpy().all()