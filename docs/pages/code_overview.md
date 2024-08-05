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
├── storage/                  | storage functionality
│   ├── models/               | database models
│   ├── repository/           | repository classes for storage
│   └── services/             | utility functions for storage
├── tests/                    | automated tests for the SDK
│   ├── db/                   | tests for database functions
│   ├── deployments/          | tests for model deployments
│   ├── files/                | assets for testing
│   ├── integrations/         | tests for integrations
│   ├── projects/             | test projects
│   └── units/                | unit tests
├── utils/                    | various utility functionality
├── cli.py                    | command-line interface to build and deploy the SDK
└── sdk.py                    | base class to create an SDK instance
```

