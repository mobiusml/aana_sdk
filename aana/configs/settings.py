from pathlib import Path

from pydantic_settings import BaseSettings

from aana.configs.db import DbSettings


class TestSettings(BaseSettings):
    """A pydantic model for test settings."""

    test_mode: bool = False
    use_deployment_cache: bool = False  # use cached deployment results for testing
    save_deployment_cache: bool = False  # save deployment results to cache for testing


class Settings(BaseSettings):
    """A pydantic model for SDK settings."""

    tmp_data_dir: Path = Path("/tmp/aana_data")  # noqa: S108
    image_dir: Path = tmp_data_dir / "images"
    video_dir: Path = tmp_data_dir / "videos"
    num_workers: int = 2

    db_config: DbSettings = DbSettings()

    test: TestSettings = TestSettings()


settings = Settings()
