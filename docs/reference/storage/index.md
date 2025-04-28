# Storage

Aana SDK provides an integration with an SQL database to store and retrieve data. 

Currently, Aana SDK supports SQLite (default) and PostgreSQL databases. See [Database Configuration](./../settings.md#aana.configs.DbSettings) for more information.

The database integration is based on the [SQLAlchemy](https://www.sqlalchemy.org/) library and consists of two main components: 

- [Models](models.md) - Database models (entities) that represent tables in the database.
- [Repositories](repositories.md) - Classes that provide an interface to interact with the database models.

To use the database integration, you can either:

- Use the provided models and repositories.
- Create your own models and repositories by extending the provided ones (for example, extending the VideoEntity model to add custom fields).
- Create your own models and repositories from scratch/base classes (for example, creating a new model for a new entity).

## How to Use Provided Models and Repositories

If you want to use the provided models and repositories, you can use the following steps:

### Get session object

You can use `get_session` method from the `aana.storage.session` module. It's a context manager that provides a session object to interact with the database. 
    
```python
from aana.storage.session import get_session

async with get_session() as session:
    # Use the session object to interact with the database
```

Here is an example of how to use the session in an endpoint:


```python
from aana.api import Endpoint
from aana.storage.session import get_session

class TranscribeVideoEndpoint(Endpoint):

    async def run(self, video: VideoInput) -> WhisperOutput:
        async with get_session() as session:
            repo = SomeRepository(session)
            await repo.some_method(...)
            # or 
            await SomeRepository(session).some_method(...)
```


### Create a repository object and use it to interact with the database.

You can use the provided repositories from the `aana.storage.repository` module. See [Repositories](repositories.md) for the list of available repositories.

The repositories are designed in asyncronous way, so you can use them in async functions. 

For example, to work with the [`VideoEntity`](models.md#aana.storage.models.VideoEntity) model, you can create a [`VideoRepository`](repositories.md#aana.storage.repository.VideoRepository) object:

```python
from aana.storage.repository import VideoRepository

video_repository = VideoRepository(session)
```

And then use the repository object to interact with the database. For example, to save a video object to the database (storing media ID, URL, path, title, description, etc.):  

```python
from aana.core.models import Video

video = Video(title="My Video", url="https://example.com/video.mp4")
await video_repository.save(video)
```

Or, if you are using Endpoint, you can create a repository object in the `initialize` method:

```python
from aana.api import Endpoint
from aana.storage.repository import VideoRepository
from aana.storage.session import get_session

class TranscribeVideoEndpoint(Endpoint):

    async def run(self, video: VideoInput) -> WhisperOutput:
        video_obj: Video = await run_remote(download_video)(video_input=video)
        async with get_session() as session:
            await VideoRepository(session).save(video_obj)
        # ...
``` 