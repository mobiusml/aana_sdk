from pathlib import Path

from pydantic import BaseSettings

from aana.configs.db import DBConfig


class Settings(BaseSettings):
    """A pydantic model for SDK settings."""

    tmp_data_dir: Path = Path("/tmp/aana_data")  # noqa: S108
    image_dir = tmp_data_dir / "images"
    video_dir = tmp_data_dir / "videos"

    db_config: DBConfig = {
        "datastore_type": "sqlite",
        "datastore_config": {"path": Path("/var/lib/aana_data")},
    }


settings = Settings()
