# ruff: noqa: S101
from importlib import resources
from pathlib import Path

import pytest

from aana.core.models.audio import Audio
from aana.core.models.video import Video, VideoInput
from aana.integrations.external.yt_dlp import download_video
from aana.processors.video import extract_audio


@pytest.mark.parametrize("audio_file", ["squirrel.wav", "physicsworks.wav"])
def test_audio(audio_file):
    """Test that the audio can be created from path, url(pending), or content."""
    # Test creation from a path
    try:
        path = resources.path("aana.tests.files.audios", audio_file)
        audio = Audio(path=path, save_on_disk=False)
        assert audio.path == path
        assert audio.content is None
        assert audio.url is None

        content = audio.get_content()
        assert len(content) > 0
    finally:
        audio.cleanup()

    # Test creation from URL
    try:
        url = "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4"
        audio = Audio(url=url, save_on_disk=False)
        assert audio.path is None
        assert audio.content is None
        assert audio.url == url

    finally:
        audio.cleanup()

    # Test creation from content
    try:
        path = resources.path("aana.tests.files.audios", audio_file)
        content = path.read_bytes()
        audio = Audio(content=content, save_on_disk=False)
        assert audio.path is None
        assert audio.content == content
        assert audio.url is None

        assert audio.get_content() == content
    finally:
        audio.cleanup()


def test_audio_path_not_exist():
    """Test that the audio can't be created from path if the path doesn't exist."""
    path = Path("path/to/audio_that_does_not_exist.wav")
    with pytest.raises(FileNotFoundError):
        Audio(path=path)


@pytest.mark.parametrize("audio_file", ["squirrel.wav", "physicsworks.wav"])
def test_save_audio(audio_file):
    """Test that save_on_disk works for audio."""
    # Test that the audio is saved to disk when save_on_disk is True
    try:
        path = resources.path("aana.tests.files.audios", audio_file)
        audio = Audio(path=path, save_on_disk=True)
        assert audio.path == path
        assert audio.content is None
        assert audio.url is None
        assert audio.path.exists()
    finally:
        audio.cleanup()
    assert audio.path.exists()  # Cleanup should NOT delete the file if path is provided

    # Test saving from audio URL to disk and read
    try:
        url = "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.wav"

        audio = Audio(url=url, save_on_disk=True)
        assert audio.content is None
        assert audio.url is not None
    finally:
        audio.cleanup()


@pytest.mark.parametrize("audio_file", ["squirrel.wav", "physicsworks.wav"])
def test_cleanup(audio_file):
    """Test that cleanup works for audios."""
    try:
        path = resources.path("aana.tests.files.audios", audio_file)
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
    assert audio.path is not None
    assert audio.content is not None
    assert audio.url is None

    # Test audio from VideoInput (for audio input): return audio bytes
    path = resources.path("aana.tests.files.audios", "physicsworks.wav")
    video_input = VideoInput(path=str(path))
    video = download_video(video_input)
    assert isinstance(video, Video)
    audio = extract_audio(video)
    assert isinstance(audio, Audio)
    assert audio.content is not None
    assert audio.path is not None
    assert audio.url is None

    # Test audio from VideoInput (for no audio channel): return empty bytes
    path = resources.path("aana.tests.files.videos", "squirrel_no_audio.mp4")
    video_input = VideoInput(path=str(path))
    video = download_video(video_input)
    audio = extract_audio(video)
    assert isinstance(audio, Audio)
    assert audio.content == b""
    assert audio.url is None
    assert audio.path is not None

    try:
        url = "https://mobius-public.s3.eu-west-1.amazonaws.com/squirrel.mp4"
        video_input = VideoInput(url=url)
        video = download_video(video_input)
        audio = extract_audio(video)
        assert isinstance(audio, Audio)
        assert audio.path is not None
        assert audio.content is not None
        assert audio.path.exists()
        assert video.media_id == video_input.media_id
        assert audio.media_id == f"audio_{video.media_id}"

    finally:
        video.cleanup()
        audio.cleanup()

    # Test loading numpy from the Audio object
    path = resources.path("aana.tests.files.audios", "physicsworks.wav")
    audio = Audio(path=path, save_on_disk=True)
    assert audio.get_numpy() is not None

    path = resources.path("aana.tests.files.audios", "squirrel.wav")
    audio = Audio(path=path, save_on_disk=True)
    assert audio.get_numpy().all() == 0

    # Test content of Audio objects created in different ways from same source video is the same.
    video_path = resources.path("aana.tests.files.videos", "physicsworks.webm")
    video = Video(path=video_path)
    extracted_audio_1 = extract_audio(video)

    video_input = VideoInput(path=str(video_path))
    downloaded_video = download_video(video_input)
    extracted_audio_2 = extract_audio(downloaded_video)

    # Paths will be different since Audio object is created from two different Video objects,
    # but the audio content should be the same
    assert extracted_audio_1.path != extracted_audio_2.path
    assert (extracted_audio_1.get_numpy().all()) == extracted_audio_2.get_numpy().all()
