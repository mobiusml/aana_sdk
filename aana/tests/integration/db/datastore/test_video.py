# ruff: noqa: S101

from importlib import resources
from tempfile import NamedTemporaryFile
from unittest.mock import patch

from aana.configs.db import run_alembic_migrations
from aana.configs.settings import settings
from aana.models.core.video import Video
from aana.utils.db import save_video


def test_save_video():
    """Tests saving a video against an in-memory SQLite database."""
    # Use a temp file and change the settings to use it for the SQLite data path.
    with NamedTemporaryFile(dir=settings.tmp_data_dir) as tmp:
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
