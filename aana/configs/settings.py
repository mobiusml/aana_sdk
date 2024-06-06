from pathlib import Path

from pydantic_settings import BaseSettings

from aana.configs.db import DbSettings


class TestSettings(BaseSettings):
    """A pydantic model for test settings."""

    test_mode: bool = False
    use_deployment_cache: bool = False  # use cached deployment results for testing
    save_deployment_cache: bool = False  # save deployment results to cache for testing


class TaskQueueSettings(BaseSettings):
    """A pydantic model for task queue settings."""

    enabled: bool = True
    num_workers: int = 4
    execution_timeout: int = 600


class Settings(BaseSettings):
    """A pydantic model for SDK settings."""

    tmp_data_dir: Path = Path("/tmp/aana_data")  # noqa: S108
    image_dir: Path = tmp_data_dir / "images"
    video_dir: Path = tmp_data_dir / "videos"
    audio_dir: Path = tmp_data_dir / "audios"
    model_dir: Path = tmp_data_dir / "models"
    num_workers: int = 2

    task_queue: TaskQueueSettings = TaskQueueSettings()

    db_config: DbSettings = DbSettings()

    test: TestSettings = TestSettings()


settings = Settings()
