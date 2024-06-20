# ruff: noqa: S101

from importlib import resources
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from aana.configs.settings import settings
from aana.core.models.video import Video
from aana.exceptions.db import NotFoundException
from aana.storage.op import run_alembic_migrations
from aana.storage.repository.media import MediaRepository
from aana.storage.repository.video import VideoRepository
from aana.storage.services.video import delete_media, save_video


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
        # because it gets set before the test is run due to import dependencies.
        # TODO: We should stop using the engine as an importable name
        with patch(
            "aana.storage.services.video.engine", settings.db_config.get_engine()
        ):
            media_id = "foobar"
            duration = 550.25
            path = resources.path("aana.tests.files.videos", "squirrel.mp4")
            video = Video(path=path, media_id=media_id)
            result = save_video(video, duration)

            assert result["media_id"] == media_id

            # Check that saved video is now available from repo
            with Session(settings.db_config.get_engine()) as session:
                video_repo = VideoRepository(session)
                video_entity = video_repo.read(media_id)

                assert video_entity
                # Check that video has a media, that media exists
                # and that media has a video set
                media_repo = MediaRepository(session)
                media_entity = media_repo.read(media_id)
                assert media_entity.video == video_entity

            # Use a second session here because deleted objects sometimes linger
            # as long as a session persists.
            with Session(settings.db_config.get_engine()) as session:
                # Check that delete function works
                result = delete_media(media_id)
                with pytest.raises(NotFoundException):
                    _ = media_repo.read(media_id)
                with pytest.raises(NotFoundException):
                    _ = video_repo.read(media_id)
