[![Python package](https://github.com/mobiusml/aana_sdk/actions/workflows/python-package.yml/badge.svg)](https://github.com/mobiusml/aana_sdk/actions/workflows/python-package.yml)

# Aana

Aana is a multi-model SDK for deploying and serving machine learning models.

## Installation

1. Clone this repository.

2. Install additional libraries.

```bash
apt update && apt install -y libgl1
```

3. Install the package with poetry.

It will install the package and all dependencies in a virtual environment.

```bash
sh install.sh
```

4. Run the SDK.

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 CUDA_VISIBLE_DEVICES=0 poetry run aana deploy aana.projects.chat_with_video.app:aana_app --port 8000 --host 0.0.0.0
```

The target parameter specifies the set of endpoints to deploy.

The first run might take a while because the models will be downloaded from the Internet and cached. 

Once you see `Deployed Serve app successfully.` in the logs, the server is ready to accept requests.

You can change the port and CUDA_VISIBLE_DEVICES environment variable to your needs.

The server will be available at http://localhost:8000.

The documentation will be available at http://localhost:8000/docs and http://localhost:8000/redoc.

For HuggingFace Transformers, you need to specify HF_AUTH environment variable with your HuggingFace API token.

5. Send a request to the server.

You can find examples in the [demo notebook](notebooks/demo.ipynb).

## Build Serve Config Files

The Serve config is the recommended way to deploy and update your applications in production. Aana SDK provides a way to build the Serve config files for the Aana applications.

To build the Serve config files, run the following command:

```bash
poetry run aana build aana.projects.chat_with_video.app:aana_app
```

The command will generate the Serve config file and app config file and save them in the `aana.projects.chat_with_video.app` directory.

Use --help to see the available options.

```bash
poetry run aana build --help
```

When you are running the Aana application using the Serve config files, you need to run the migrations to create the database tables for the application. To run the migrations, use the following command:

```bash
poetry run aana migrate aana.projects.chat_with_video.app:aana_app
```

## Run with Docker

1. Clone this repository.

2. Update submodules.

```bash
git submodule update --init --recursive
```

3. Build the Docker image.

```bash
docker build -t aana:0.2.0 .
```

4. Run the Docker container.

```bash
docker run --rm --init -p 8000:8000 --gpus all -e TARGET="llama2" -e CUDA_VISIBLE_DEVICES=0 -v aana_cache:/root/.aana -v aana_hf_cache:/root/.cache/huggingface --name aana_instance aana:0.2.0
```

Use the environment variable TARGET to specify the set of endpoints to deploy.

The first run might take a while because the models will be downloaded from the Internet and cached. The models will be stored in the `aana_cache` volume. The HuggingFace models will be stored in the `aana_hf_cache` volume. If you want to remove the cached models, remove the volume.

Once you see `Deployed Serve app successfully.` in the logs, the server is ready to accept requests.

You can change the port and gpus parameters to your needs.

The server will be available at http://localhost:8000.

The documentation will be available at http://localhost:8000/docs and http://localhost:8000/redoc.

5. Send a request to the server.

You can find examples in the [demo notebook](notebooks/demo.ipynb).

## Developing in a Dev Container

If you are using Visual Studio Code, you can run this repository in a 
[dev container](https://code.visualstudio.com/docs/devcontainers/containers). This lets you install and 
run everything you need for the repo in an isolated environment via docker on a host system. 
Running it somewhere other than a Mobius dev server may cause issues due to the mounts of `/nas` and
`/nas2` inside the container, but you can specify the environment variables for VS Code `PATH_NAS` and
`PATH_NAS2` which will override the default locations used for these mount points (otherise they default 
to look for `/nas` and `/nas2`). You can read more about environment variables for dev containers 
[here](https://containers.dev/implementors/json_reference/).

## Code Standards
This project uses Ruff for linting and formatting. If you want to 
manually run Ruff on the codebase, using poetry it's

```sh
poetry run ruff check aana
```

You can automatically fix some issues with the `--fix`
 and `--unsafe-fixes` options. (Be sure to install the dev 
 dependencies: `poetry install --with=dev`. )

To run the auto-formatter, it's

```sh
poetry run ruff format aana
```

(If you are running code in a non-poetry environment, just leave off `poetry run`.)
If you want to enable this as a local pre-commit hook, additionally
run the following:

```sh
git config core.hooksPath .githooks
```

Depending on your workflow, you may need to ensure that the `ruff` 
command is available in your default shell. You can also simply run
`.githooks/pre-commit` manually if you prefer.

For users of VS Code, the included `settings.json` should ensure
that Ruff problems appear while you edit, and formatting is applied
automatically on save.


## Testing

The project uses pytest for testing. To run the tests, use the following command:

```bash
poetry run pytest
```

If you are using VS Code, you can run the tests using the Test Explorer that is installed with the [Python extension](https://code.visualstudio.com/docs/python/testing).

There are a few environment variables that can be set to control the behavior of the tests:
- `USE_DEPLOYMENT_CACHE`: If set to `true`, the tests will use the deployment cache to avoid downloading the models and running the deployments. This is useful for running integration tests faster and in the environment where GPU is not available.
- `SAVE_DEPLOYMENT_CACHE`: If set to `true`, the tests will save the deployment cache after running the deployments. This is useful for updating the deployment cache if new deployments or tests are added.

### How to use the deployment cache environment variables

Here are some examples of how to use the deployment cache environment variables.

#### Do you want to run the tests normally using GPU?
    
```bash
USE_DEPLOYMENT_CACHE=false
SAVE_DEPLOYMENT_CACHE=false
```

This is the default behavior. The tests will run normally using GPU and the deployment cache will be completely ignored.

#### Do you want to run the tests faster without GPU?

```bash
USE_DEPLOYMENT_CACHE=true
SAVE_DEPLOYMENT_CACHE=false
```

This will run the tests using the deployment cache to avoid downloading the models and running the deployments. The deployment cache will not be updated after running the deployments. Only use it if you are sure that the deployment cache is up to date.

#### Do you want to update the deployment cache?

```bash
USE_DEPLOYMENT_CACHE=false
SAVE_DEPLOYMENT_CACHE=true
```

This will run the tests normally using GPU and save the deployment cache after running the deployments. Use it if you have added new deployments or tests and want to update the deployment cache.


## Databases
The project uses two databases: a vector database as well as a tradtional SQL database,
referred to internally as vectorstore and datastore, respectively.

### Vectorstore
TBD

### Datastore
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

## Settings

Here are the environment variables that can be used to configure the Aaana SDK:
- TMP_DATA_DIR: The directory to store temporary data. Default: `/tmp/aana`.
- NUM_WORKERS: The number of request workers. Default: `2`.
- DB_CONFIG: The database configuration in the format `{"datastore_type": "sqlite", "datastore_config": {"path": "/path/to/sqlite.db"}}`. Currently only SQLite and PostgreSQL are supported. Default: `{"datastore_type": "sqlite", "datastore_config": {"path": "/var/lib/aana_data"}}`.
- USE_DEPLOYMENT_CACHE (testing only): If set to `true`, the tests will use the deployment cache to avoid downloading the models and running the deployments. Default: `false`.
- SAVE_DEPLOYMENT_CACHE (testing only): If set to `true`, the tests will save the deployment cache after running the deployments. Default: `false`.
- HF_HUB_ENABLE_HF_TRANSFER: If set to `1`, the HuggingFace Transformers will use the HF Transfer library to download the models from HuggingFace Hub to speed up the process. Recommended to always set to it `1`. Default: `0`.
