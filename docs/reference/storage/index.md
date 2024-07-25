# Storage

Aana SDK provides an integration with an SQL database to store and retrieve data. 

Currently, Aana SDK supports SQLite (default) and PostgreSQL databases. See [Database Configuration](/reference/settings/#aana.configs.DbSettings) for more information.

The database integration is based on the [SQLAlchemy](https://www.sqlalchemy.org/) library and consists of two main components: 

- [Models](/reference/storage/models/) - Database models (entities) that represent tables in the database.
- [Repositories](/reference/storage/repositories/) - Classes that provide an interface to interact with the database models.

To use the database integration, you can either:

- Use the provided models and repositories.
- Create your own models and repositories by extending the provided ones (for example, extending the VideoEntity model to add custom fields).
- Create your own models and repositories from scratch/base classes (for example, creating a new model for a new entity).

## How to Use Provided Models and Repositories

If you want to use the provided models and repositories, you can use the following steps:

### Get session object

You can use `get_session` method from the `aana.storage.session` module:
    
```python
from aana.storage.session import get_session

session = get_session()
```

<!-- ::: aana.storage.session.get_session -->

If you are using Endpoint, you can use the `session` attribute that is available after the endpoint is initialized:

```python
from aana.api import Endpoint

class TranscribeVideoEndpoint(Endpoint):
    async def initialize(self):
        await super().initialize()
        # self.session is available here after the endpoint is initialized

    async def run(self, video: VideoInput) -> WhisperOutput:
        # self.session is available here as well
```


### Create a repository object and use it to interact with the database.

You can use the provided repositories from the `aana.storage.repository` module. See [Repositories](/reference/storage/repositories/) for the list of available repositories.

For example, to work with the [`VideoEntity`](/reference/storage/models/#aana.storage.models.VideoEntity) model, you can create a [`VideoRepository`](/reference/storage/repositories/#aana.storage.repository.VideoRepository) object:

```python
from aana.storage.repository import VideoRepository

video_repository = VideoRepository(session)
```

And then use the repository object to interact with the database. For example, to save a video object to the database (storing media ID, URL, path, title, description, etc.):  

```python
from aana.core.models import Video

video = Video(title="My Video", url="https://example.com/video.mp4")
video_repository.save(video)
```

Or, if you are using Endpoint, you can create a repository object in the `initialize` method:

```python
from aana.api import Endpoint

class TranscribeVideoEndpoint(Endpoint):
    async def initialize(self):
        await super().initialize()
        self.video_repository = VideoRepository(self.session)

    async def run(self, video: VideoInput) -> WhisperOutput:
        video_obj: Video = await run_remote(download_video)(video_input=video)
        self.video_repository.save(video_obj)
        # ...
``` 