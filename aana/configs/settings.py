from pathlib import Path

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aana.configs.db import DbSettings, SQLiteConfig
from aana.core.models.base import pydantic_protected_fields
from aana.storage.op import DbType


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
        heartbeat_timeout (int): The maximum time between heartbeats in seconds.
        max_retries (int): The maximum number of retries for a task.
        maximum_active_tasks_per_user (int): The maximum number of active tasks per user (only applicable in the API service).
    """

    enabled: bool = True
    num_workers: int = 4
    execution_timeout: int = 600
    heartbeat_timeout: int = 60
    max_retries: int = 3
    maximum_active_tasks_per_user: int = 25


class ApiServiceSettings(BaseModel):
    """A pydantic model for API service settings.

    Attributes:
        enabled (bool): Flag indicating if the API service is enabled.
        lago_url (str): The URL of the LAGO API service.
        lago_api_key (str): The API key for the LAGO API service.
    """

    enabled: bool = False
    lago_url: str | None = None
    lago_api_key: str | None = None


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
    image_dir: Path | None = None
    video_dir: Path | None = None
    audio_dir: Path | None = None
    model_dir: Path | None = None

    num_workers: int = 2

    task_queue: TaskQueueSettings = TaskQueueSettings()

    db_config: DbSettings = DbSettings()

    test: TestSettings = TestSettings()

    api_service: ApiServiceSettings = ApiServiceSettings()

    api_service_db_config: DbSettings = DbSettings(
        datastore_type=DbType.SQLITE,
        datastore_config=SQLiteConfig(path="/var/lib/aana_api_service_data"),
    )

    @model_validator(mode="after")
    def setup_resource_directories(self):
        """Create the resource directories if they do not exist."""
        if self.image_dir is None:
            self.image_dir = self.tmp_data_dir / "images"
        if self.video_dir is None:
            self.video_dir = self.tmp_data_dir / "videos"
        if self.audio_dir is None:
            self.audio_dir = self.tmp_data_dir / "audios"
        if self.model_dir is None:
            self.model_dir = self.tmp_data_dir / "models"

        self.tmp_data_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def validate_lago_settings(self):
        """Validate the LAGO API service settings."""
        if self.api_service.enabled and (
            self.api_service.lago_url is None or self.api_service.lago_api_key is None
        ):
            raise ValueError(  # noqa: TRY003
                "LAGO API service settings are required when the API service is enabled."
            )
        return self

    model_config = SettingsConfigDict(
        protected_namespaces=("settings", *pydantic_protected_fields),
        env_nested_delimiter="__",
        env_ignore_empty=True,
    )


settings = Settings()
