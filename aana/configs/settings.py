from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aana.configs.db import DbSettings
from aana.core.models.base import pydantic_protected_fields


class TestSettings(BaseModel):
    """A pydantic model for test settings.

    Attributes:
        test_mode (bool): Flag indicating if the SDK is in test mode.
        save_expected_output (bool): Flag indicating if the expected output should be saved (to create test cases).
    """

    test_mode: bool = False
    save_expected_output: bool = False


class TaskQueueSettings(BaseModel):
    """A pydantic model for task queue settings.

    Attributes:
        enabled (bool): Flag indicating if the task queue is enabled.
        num_workers (int): The number of workers in the task queue.
        execution_timeout (int): The maximum execution time for a task in seconds.
            After this time, if the task is still running,
            it will be considered as stuck and will be reassign to another worker.
        max_retries (int): The maximum number of retries for a task.
    """

    enabled: bool = True
    num_workers: int = 4
    execution_timeout: int = 600
    max_retries: int = 3


class Settings(BaseSettings):
    """A pydantic model for SDK settings.

    Attributes:
        tmp_data_dir (Path): The temporary data directory.
        image_dir (Path): The temporary image directory.
        video_dir (Path): The temporary video directory.
        audio_dir (Path): The temporary audio directory.
        model_dir (Path): The temporary model directory.
        num_workers (int): The number of web workers.
        task_queue (TaskQueueSettings): The task queue settings.
        db_config (DbSettings): The database configuration.
        test (TestSettings): The test settings.
    """

    tmp_data_dir: Path = Path("/tmp/aana_data")  # noqa: S108
    image_dir: Path = tmp_data_dir / "images"
    video_dir: Path = tmp_data_dir / "videos"
    audio_dir: Path = tmp_data_dir / "audios"
    model_dir: Path = tmp_data_dir / "models"
    num_workers: int = 2

    task_queue: TaskQueueSettings = TaskQueueSettings()

    db_config: DbSettings = DbSettings()

    test: TestSettings = TestSettings()

    @field_validator("tmp_data_dir", mode="after")
    def create_tmp_data_dir(cls, path: Path) -> Path:
        """Create the tmp_data_dir if it doesn't exist."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    model_config = SettingsConfigDict(
        protected_namespaces=("settings", *pydantic_protected_fields),
        env_nested_delimiter="__",
        env_ignore_empty=True,
    )


settings = Settings()
