# ruff: noqa: S101

import uuid
from importlib import resources

import pytest

from aana.core.models.video import Video, VideoMetadata
from aana.exceptions.db import MediaIdAlreadyExistsException, NotFoundException
from aana.storage.repository.video import VideoRepository


@pytest.fixture(scope="function")
def dummy_video():
    """Creates a dummy video for testing."""
    media_id = str(uuid.uuid4())
    path = resources.files("aana.tests.files.videos") / "squirrel.mp4"
    video = Video(path=path, media_id=media_id)
    return video


def test_save_video(db_session, dummy_video):
    """Tests saving a video."""
    video_repo = VideoRepository(db_session)
    video_repo.save(dummy_video)

    video_entity = video_repo.read(dummy_video.media_id)
    assert video_entity
    assert video_entity.id == dummy_video.media_id

    # Try to save the same video again
    with pytest.raises(MediaIdAlreadyExistsException):
        video_repo.save(dummy_video)

    video_repo.delete(dummy_video.media_id)
    with pytest.raises(NotFoundException):
        video_repo.read(dummy_video.media_id)


def test_get_metadata(db_session, dummy_video):
    """Tests getting video metadata."""
    video_repo = VideoRepository(db_session)
    video_repo.save(dummy_video)

    metadata = video_repo.get_metadata(dummy_video.media_id)
    assert isinstance(metadata, VideoMetadata)
    assert metadata.title == dummy_video.title
    assert metadata.description == dummy_video.description
    assert metadata.duration == None

    video_repo.delete(dummy_video.media_id)
    with pytest.raises(NotFoundException):
        video_repo.get_metadata(dummy_video.media_id)
