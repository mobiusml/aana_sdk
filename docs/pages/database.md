# Databases

Aana SDK provides a database layer that uses SQLAlchemy as an ORM layer and Alembic for migrations. 

## Configuration

By default, the SDK uses SQLite as the database.

The database configuration can be set using the environment variable `DB_CONFIG` or by changing the Settings class. The default configuration:

```json
{
  "datastore_type": "sqlite",
  "datastore_config": {
    "path": "/var/lib/aana_data"
  }
}
```

Currently, Aana SDK supports SQLite and PostgreSQL databases. See the [DbSettings](./../reference/settings.md#aana.configs.DbSettings) for more information on available options.

## Migration

The SDK comes with a set of predefined models and migrations that will automatically set up the database.

The migrations are run with `aana_app.migrate()` or with CLI command `aana migrate`.

## Data Models

The SDK has following predefined data models:

- `TaskEntity`: Represents a task for the task queue. It is used internally by the SDK and is not intended to be used directly.
- [`BaseEntity`](./../reference/models/index.md#aana.storage.models.BaseEntity): Base class for all entities in the SDK. Use it as a base class for your custom models.
- [`MediaEntity`](./../reference/models/index.md#aana.storage.models.MediaEntity): Base class for media entities (audio, video, image). You can use it as a base class for your custom media models.
- [`VideoEntity`](./../reference/models/index.md#aana.storage.models.VideoEntity): Represents a video. You can use it in your application directly or as a base class for your custom video models.
- [`TranscriptEntity`](./../reference/models/index.md#aana.storage.models.TranscriptEntity): Represents an ASR transcript. You can use it in your application directly or as a base class for your custom ASR models.
- [`CaptionEntity`](./../reference/models/index.md#aana.storage.models.CaptionEntity): Represents a caption. You can use it in your application directly or as a base class for your custom caption models.

## Repositories

Repositories are classes that provide an interface to interact with the database. The SDK provides repositories for each data model.

The repositories are available in the `aana.storage.repository` module. Here is a list of available repositories:
- `TaskRepository`: Repository for the `TaskEntity` model. It is used internally by the SDK and is not intended to be used directly.
- [`BaseRepository`](./../reference/storage/repositories.md#aana.storage.repository.BaseRepository): Base repository class for all entities in the SDK. Use it as a base class for your custom repositories.
- [`MediaRepository`](./../reference/storage/repositories.md#aana.storage.repository.MediaRepository): Repository for media entities (audio, video, image). You can use it as a base class for your custom media repositories. 
- [`VideoRepository`](./../reference/storage/repositories.md#aana.storage.repository.VideoRepository): Repository for the `VideoEntity` model. You can use it in your application directly or as a base class for your custom video repositories.
- [`TranscriptRepository`](./../reference/storage/repositories.md#aana.storage.repository.TranscriptRepository): Repository for the `TranscriptEntity` model. You can use it in your application directly or as a base class for your custom ASR repositories.
- [`CaptionRepository`](./../reference/storage/repositories.md#aana.storage.repository.CaptionRepository): Repository for the `CaptionEntity` model. You can use it in your application directly or as a base class for your custom caption repositories.

To learn more how to use predefined repositories and models, see the [How to Use Provided Models and Repositories](./../reference/storage/index.md#how-to-use-provided-models-and-repositories) section in the reference documentation.

## Custom Models and Repositories

If predefined models and repositories do not meet your requirements, you can create your own models and repositories by extending the provided ones or creating new ones from scratch.

If you create a completely new data model, it is advisable to inherit it from the `BaseEntity` class. This will ensure that your model is compatible with the SDK's storage and retrieval mechanisms. 

If you want to extend the existing models (e.g., add custom fields to the `VideoEntity` model), you can create a new model class that inherits from the existing one. 

In any case, you should create a corresponding repository class that provides an interface to interact with the database.

Aana SDK uses [Joined Table Inheritance](https://docs.sqlalchemy.org/en/20/orm/inheritance.html#joined-table-inheritance) to implement polymorphic inheritance. This means that you can create a new model class that inherits from an existing one and add custom fields to it. The new fields will be stored in a separate table but will be transparently joined with the base table when querying the database.

As an example, let's say you want to extend the `VideoEntity` model to add two custom fields: `duration` and `status`.

```python
from enum import Enum

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from aana.core.models.media import MediaId
from aana.storage.models.video import VideoEntity

class VideoProcessingStatus(str, Enum):
    """Enum for video status."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExtendedVideoEntity(VideoEntity):
    """ORM class for videos with additional metadata."""

    __tablename__ = "extended_video"

    id: Mapped[MediaId] = mapped_column(ForeignKey("video.id"), primary_key=True)
    duration: Mapped[float | None] = mapped_column(comment="Video duration in seconds")
    status: Mapped[VideoProcessingStatus] = mapped_column(
        nullable=False,
        default=VideoProcessingStatus.CREATED,
        comment="Processing status",
    )

    __mapper_args__ = {  # noqa: RUF012
        "polymorphic_identity": "extended_video",
    }
```

In this example, we created a new model class `ExtendedVideoEntity` that inherits from the `VideoEntity` model. We added two custom fields: `duration` and `status`. The `__mapper_args__` attribute specifies the polymorphic identity of the new model. This is necessary for SQLAlchemy to correctly handle inheritance.


Next, we need to create a repository class that provides an interface to interact with the database. Here is an example of how to create a repository for the `ExtendedVideoEntity` model:

```python
from sqlalchemy.orm import Session

from aana.core.models.media import MediaId
from aana.core.models.video import Video, VideoMetadata
from aana.storage.repository.video import VideoRepository

class ExtendedVideoRepository(VideoRepository[ExtendedVideoEntity]):
    """Repository for videos with additional metadata."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, ExtendedVideoEntity)

    def save(self, video: Video, duration: float | None = None) -> dict:
        """Saves a video to datastore.

        Args:
            video (Video): The video object.
            duration (float): the duration of the video object

        Returns:
            dict: The dictionary with video and media IDs.
        """
        video_entity = ExtendedVideoEntity(
            id=video.media_id,
            path=str(video.path),
            url=video.url,
            title=video.title,
            description=video.description,
            duration=duration,
        )
        self.create(video_entity)
        return video_entity

    def get_status(self, media_id: MediaId) -> VideoProcessingStatus:
        """Get the status of a video.

        Args:
            media_id (str): The media ID.

        Returns:
            VideoProcessingStatus: The status of the video.
        """
        entity: ExtendedVideoEntity = self.read(media_id)
        return entity.status

    def update_status(self, media_id: MediaId, status: VideoProcessingStatus):
        """Update the status of a video.

        Args:
            media_id (str): The media ID.
            status (VideoProcessingStatus): The status of the video.
        """
        entity: ExtendedVideoEntity = self.read(media_id)
        entity.status = status
        self.session.commit()

    def get_metadata(self, media_id: MediaId) -> VideoMetadata:
        """Get the metadata of a video.

        Args:
            media_id (MediaId): The media ID.

        Returns:
            VideoMetadata: The video metadata.
        """
        entity: ExtendedVideoEntity = self.read(media_id)
        return VideoMetadata(
            title=entity.title,
            description=entity.description,
            duration=entity.duration,
        )
```

In this example, we created a new repository class `ExtendedVideoRepository` that inherits from the `VideoRepository` class. We redefined methods `save` and `get_metadata` to handle the custom field `duration` and added two new methods `get_status` and `update_status` to handle the custom field `status`.

Once you have created your custom model and repository classes, you need to create a migration to update the database schema. You can do this by running the following command from the package root directory:

```bash
poetry run alembic revision --autogenerate -m "<Short description of changes in sentence form.>"
```

This will create a new migration file in the `alembic/versions` directory. You can then apply with aana migrate command.

Once the migration is applied, you can start using your custom model and repository classes in your application just like the predefined ones.

```python
from aana.core.models import Video

video = Video(title="My Video", url="https://example.com/video.mp4")
duration = 42
video_repository = ExtendedVideoRepository(session)
video_repository.save(video, duration)
```

To learn more on how to use data models and repositories, see the [How to Use Provided Models and Repositories](./../reference/storage/index.md#how-to-use-provided-models-and-repositories) section in the reference documentation.