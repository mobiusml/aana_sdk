# ruff: noqa: S101

from importlib import resources
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from aana.configs.db import run_alembic_migrations
from aana.configs.settings import settings
from aana.exceptions.database import NotFoundException
from aana.models.core.video import Video
from aana.repository.datastore.media_repo import MediaRepository
from aana.repository.datastore.video_repo import VideoRepository
from aana.utils.db import delete_media, save_video


def test_save_video():
    """Tests saving a video against an in-memory SQLite database."""
    # Ensure the tmp data directory exists
    settings.tmp_data_dir.mkdir(parents=True, exist_ok=True)
    # Use a temp file and change the settings to use it for the SQLite data path.
    with NamedTemporaryFile(
        dir=settings.tmp_data_dir,
    ) as tmp:
        settings.db_config.datastore_config["path"] = tmp.name
        # Reset the engine if it has been created
        del settings.db_config.engine
        settings.db_config.engine = None

        run_alembic_migrations(settings)

        # We also have to patch the "engine" variable in aana.utils.db
        # because it gets set before the test is run due to import depedencies.
        # TODO: We should stop using the engine as an importable name
        with patch("aana.utils.db.engine", settings.db_config.get_engine()):
            media_id = "foobar"
            duration = 550.25
            path = resources.path("aana.tests.files.videos", "squirrel.mp4")
            video = Video(path=path, media_id=media_id)
            result = save_video(video, duration)

            assert result["video_id"]
            assert result["media_id"] == media_id

            video_id = result["video_id"]
            # Check that saved video is now available from repo
            with Session(settings.db_config.get_engine()) as session:
                video_repo = VideoRepository(session)
                video_by_media_id = video_repo.get_by_media_id(media_id)
                video_by_video_id = video_repo.read(video_id)

                assert video_by_media_id
                assert video_by_video_id
                assert video_by_media_id == video_by_video_id
                # Check that video has a media, that media exists
                # and that media has a video set
                media_repo = MediaRepository(session)
                media_by_media_id = media_repo.read(media_id)
                assert video_by_video_id.media == media_by_media_id
                assert media_by_media_id.video == video_by_video_id

            # Use a second session here because deleted objects sometimes linger
            # as long as a session persists.
            with Session(settings.db_config.get_engine()) as session:
                # Check that delete function works
                result = delete_media(media_id)
                with pytest.raises(NotFoundException):
                    _ = media_repo.read(media_id)
                with pytest.raises(NotFoundException):
                    _ = video_repo.get_by_media_id(media_id)
                with pytest.raises(NotFoundException):
                    _ = video_repo.read(video_id)