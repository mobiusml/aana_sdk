import json
from pathlib import Path

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aana.configs.db import DbSettings, DbType, SQLiteConfig
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


class GpuManagerSettings(BaseModel):
    """A pydantic model for GPU manager settings.

    Attributes:
        enabled (bool): Flag indicating if the GPU manager is enabled.
    """

    enabled: bool = False


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


class WebhookSettings(BaseModel):
    """A pydantic model for webhook settings.

    Attributes:
        retry_attempts (int): The number of retry attempts for webhook delivery.
        hmac_secret (str): The secret key for HMAC signature generation.
    """

    retry_attempts: int = 5
    hmac_secret: str = "webhook_secret"  # noqa: S105


class CorsSettings(BaseModel):
    """A pydantic model for CORS settings.

    Attributes:
        allow_origins (list[str]): List of origins that should be permitted to make cross-origin requests.
        allow_origin_regex (str | None): A regex string for origins that should be permitted to make cross-origin requests.
        allow_credentials (bool): Indicate that cookies should be supported for cross-origin requests.
        allow_methods (list[str]): A list of HTTP methods that should be allowed for cross-origin requests.
        allow_headers (list[str]): A list of HTTP request headers that should be supported for cross-origin requests.
    """

    allow_origins: list[str] = ["*"]
    allow_origin_regex: str | None = None
    allow_credentials: bool = True
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]

    class Config:
        json_loads = json.loads


class Settings(BaseSettings):
    """A pydantic model for SDK settings.

    Attributes:
        tmp_data_dir (Path): The temporary data directory.
        image_dir (Path): The temporary image directory.
        video_dir (Path): The temporary video directory.
        audio_dir (Path): The temporary audio directory.
        model_dir (Path): The temporary model directory.
        num_workers (int): The number of web workers.
        openai_endpoint_enabled (bool): Flag indicating if the OpenAI-compatible endpoint is enabled. Enabled by default.
        include_stacktrace (bool): Flag indicating if stacktrace should be included in error messages. Enabled by default.
        task_queue (TaskQueueSettings): The task queue settings.
        db_config (DbSettings): The database configuration.
        test (TestSettings): The test settings.
        cors (CorsSettings): The CORS settings.
    """

    tmp_data_dir: Path = Path("/tmp/aana_data")  # noqa: S108
    image_dir: Path | None = None
    video_dir: Path | None = None
    audio_dir: Path | None = None
    model_dir: Path | None = None

    num_workers: int = 2

    openai_endpoint_enabled: bool = True
    include_stacktrace: bool = True

    task_queue: TaskQueueSettings = TaskQueueSettings()

    db_config: DbSettings = DbSettings()

    test: TestSettings = TestSettings()

    api_service: ApiServiceSettings = ApiServiceSettings()

    api_service_db_config: DbSettings = DbSettings(
        datastore_type=DbType.SQLITE,
        datastore_config=SQLiteConfig(path="/var/lib/aana_api_service_data"),
    )

    webhook: WebhookSettings = WebhookSettings()

    cors: CorsSettings = CorsSettings()

    gpu_manager: GpuManagerSettings = GpuManagerSettings()

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

    model_config = SettingsConfigDict(
        protected_namespaces=("settings", *pydantic_protected_fields),
        env_nested_delimiter="__",
        env_ignore_empty=True,
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
