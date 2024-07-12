# Code overview

```
aana/                         | top level source code directory for the project
├── alembic/                  | directory for database migrations
│   └── versions/             | individual migrations
├── api/                      | API functionality
│   ├── api_generation.py     | API generation code, defines Endpoint class
│   ├── request_handler.py    | request handler routes requests to endpoints
│   ├── exception_handler.py  | exception handler to process exceptions and return them as JSON
│   ├── responses.py          | custom responses for the API
│   └── app.py                | defines the FastAPI app and connects exception handlers
├── config/                   | various configuration objects, including settings, but preconfigured deployments
│   ├── db.py                 | config for the database
│   ├── deployments.py        | preconfigured for deployments
│   └── settings.py           | app settings
├── core/                     | core models and functionality
│   ├── models/               | core data models
│   ├── libraries/            | base libraries for audio, images etc.
│   └── chat/                 | LLM chat templates
├── deployments/              | classes for predefined deployments (e.g. Hugging Face Transformers, Whisper, vLLM)
├── exceptions/               | custom exception classes
├── integrations/             | integrations with 3rd party libraries
│   ├── external/             | integrations with 3rd party libraries for example image, video, audio processing, download youtube videos, etc.
│   └── haystack/             | integrations with Deepset Haystack
├── processors/               | utility functions for processing data
├── projects/                 | example projects
│   ├── chat_with_video/      | a multimodal chat application that allows users to upload a video 
|   |                         |     and ask questions about the video content based on the visual and audio information
│   ├── llama2/               | an application that deploys LLaMa2 7B Chat model
│   ├── lowercase/            | a simple example project with no AI that converts text inputs to lowercase
│   └── whisper/              | an application that demonstrates the Whisper model for automatic speech recognition (ASR)
├── storage/                  | storage functionality
│   ├── models/               | database models
│   ├── repository/           | repository classes for storage
│   └── services/             | utility functions for storage
├── tests/                    | automated tests for the SDK
│   ├── db/                   | tests for database functions
│   ├── deployments/          | tests for model deployments
│   ├── files/                | assets for testing
│   └── units/                | unit tests
├── utils/                    | various utility functionality
├── cli.py                    | command-line interface to build and deploy the SDK
└── sdk.py                    | base class to create an SDK instance
```

# Developing in a Dev Container

If you are using Visual Studio Code, you can run this repository in a 
[dev container](https://code.visualstudio.com/docs/devcontainers/containers). This lets you install and 
run everything you need for the repo in an isolated environment via docker on a host system. 


# Databases
The project includes some useful tools for storing structured metadata in a SQL database.

The datastore uses SQLAlchemy as an ORM layer and Alembic for migrations. The migrations are run 
automatically at startup. If changes are made to the SQLAlchemy models, it is necessary to also 
create an alembic migration that can be run to upgrade the database. 
The easiest way to do so is as follows:

```bash
poetry run alembic revision --autogenerate -m "<Short description of changes in sentence form.>"
```

ORM models referenced in the rest of the code should be imported from `aana.models.db` directly,
not from that model's file for reasons explained in `aana/models/db/__init__.py`. This also means that 
if you add a new model class, it should be imported by `__init__.py` in addition to creating a migration.

Higher level code for interacting with the ORM is available in `aana.repository.data`.