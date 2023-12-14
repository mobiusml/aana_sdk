[![Python package](https://github.com/mobiusml/aana_sdk/actions/workflows/python-package.yml/badge.svg)](https://github.com/mobiusml/aana_sdk/actions/workflows/python-package.yml)

# Aana

Aana is a multi-model SDK for deploying and serving machine learning models.

## Installation

1. Clone this repository.
2. Update submodules.

```bash
git submodule update --init --recursive
```

3. Install additional libraries.

```bash
apt update && apt install -y libgl1
```

4. Install the package with poetry.

It will install the package and all dependencies in a virtual environment.

```bash
sh install.sh
```

5. Run the SDK.

```bash
CUDA_VISIBLE_DEVICES=0 poetry run aana --port 8000 --host 0.0.0.0 --target llama2
```

The target parameter specifies the set of endpoints to deploy.

The first run might take a while because the models will be downloaded from Google Drive and cached.

Once you see `Deployed Serve app successfully.` in the logs, the server is ready to accept requests.

You can change the port and CUDA_VISIBLE_DEVICES environment variable to your needs.

The server will be available at http://localhost:8000.

The documentation will be available at http://localhost:8000/docs and http://localhost:8000/redoc.

For HuggingFace Transformers, you need to specify HF_AUTH environment variable with your HuggingFace API token.

6. Send a request to the server.

You can find examples in the [demo notebook](notebooks/demo.ipynb).

## Run with Docker

1. Clone this repository.

2. Update submodules.

```bash
git submodule update --init --recursive
```

3. Build the Docker image.

```bash
docker build -t aana:0.1.0 .
```

4. Run the Docker container.

```bash
docker run --rm --init -p 8000:8000 --gpus all -e TARGET="llama2" -e CUDA_VISIBLE_DEVICES=0 -v aana_cache:/root/.aana -v aana_hf_cache:/root/.cache/huggingface --name aana_instance aana:0.1.0
```

Use the environment variable TARGET to specify the set of endpoints to deploy.

The first run might take a while because the models will be downloaded from Google Drive and cached. The models will be stored in the `aana_cache` volume. The HuggingFace models will be stored in the `aana_hf_cache` volume. If you want to remove the cached models, remove the volume.

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