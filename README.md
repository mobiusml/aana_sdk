[![Python package](https://github.com/mobiusml/aana_sdk/actions/workflows/python-package.yml/badge.svg)](https://github.com/mobiusml/aana_sdk/actions/workflows/python-package.yml)

# Aana

Aana SDK is a powerful serving layer designed for constructing web applications utilizing advanced multimodal models and RAG (Retrieval-Augmented Generation) systems. It facilitates the deployment of large-scale machine learning models, including those for vision, audio, and language, enabling the development of multimodal content applications such as search engines, recommendation systems, and data insights platforms.

## Features

- *Multimodal Input Handling:* Aana SDK seamlessly handles various types of multimodal inputs, including videos, audio, and text, providing versatility in application development.
- *Streaming Support:* With streaming support for both input and output, Aana SDK ensures smooth data processing for real-time applications.
- *Scalability:* Leveraging the capabilities of Ray serve, Aana SDK allows the deployment of multimodal models and applications across GPU clusters, ensuring scalability and efficient resource utilization.
- *Rapid Development:* Aana SDK enables developers to swiftly create robust multimodal applications in a Pythonic manner, utilizing its underlying components for fast configurations and RAG setups.

## Usage

### Installing via PyPI

TBD

### Installing from Github

1. Clone this repository.

2. Install additional libraries.

```bash
apt update && apt install -y libgl1
```

You should also install [PyTorch](https://pytorch.org/get-started/locally/) version >=2.1 appropriate for your system. You can continue directly to the next step, but it will install a default version that may not make optimal se of your system's resources, for example a GPU  or even some SIMD operations. Therefore we recommend installing choosing your PyTorch package carefully and installing it manually.

3. Install the package with poetry.

It will install the package and all dependencies in a virtual environment.

```bash
sh install.sh
```

### Creating Application 
You can quickly develop multimodal applications using Aana SDK's intuitive APIs and components:

```python
from aana.sdk import AanaSDK
from aana.configs.deployments import (
    hf_blip2_opt_2_7b_deployment,
)
from aana.projects.chat_with_video.endpoints import (
    IndexVideoEndpoint,
    VideoChatEndpoint,
)

endpoints = [
    {
        "name": "index_video_stream",
        "path": "/video/index_stream",
        "summary": "Index a video and return the captions and transcriptions as a stream",
        "endpoint_cls": IndexVideoEndpoint,
    },
]

if __name__ == "__main__":
    """Runs the application."""
    # Construct an app instance
    aana_app = AanaSDK(name="demo app")
    # bind the app to a network address.
    # setting show_logs=`False` will produce a LOT of logs!
    aana_ap.conect(port=9000, host="127.0.0.1", show_logs=False)

    aana_app.register_deployment(
        name="aana_deployment",
        instance=hf_blip2_opt_2_7b_deployment,
    )

    for endpoint in endpoints:
        aana_app.register_endpoint(
            name=endpoint["name"],
            path=endpoint["path"],
            summary=endpoint["summary"],
            endpoint_cls=endpoint["endpoint_cls"],
        )

    aana_app.deploy(blocking=True)
```

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

## Run with Docker

1. Clone this repository.

2. Build the Docker image.

```bash
docker build -t aana:0.2.0 .
```

3. Run the Docker container.

```bash
docker run --rm --init -p 8000:8000 --gpus all -e TARGET="llama2" -e CUDA_VISIBLE_DEVICES=0 -v aana_cache:/root/.aana -v aana_hf_cache:/root/.cache/huggingface --name aana_instance aana:0.2.0
```

Use the environment variable TARGET to specify the set of endpoints to deploy.

The first run might take a while because the models will be downloaded from the Internet and cached. The models will be stored in the `aana_cache` volume. The HuggingFace models will be stored in the `aana_hf_cache` volume. If you want to remove the cached models, remove the volume.

Once you see `Deployed Serve app successfully.` in the logs, the server is ready to accept requests.

You can change the port and gpus parameters to your needs.

The server will be available at http://localhost:8000.

The app documentation will be available at http://localhost:8000/docs and http://localhost:8000/redoc.

5. Send a request to the server.

You can find examples in the [demo notebook](notebooks/demo.ipynb).

## Developing in a Dev Container

If you are using Visual Studio Code, you can run this repository in a 
[dev container](https://code.visualstudio.com/docs/devcontainers/containers). This lets you install and 
run everything you need for the repo in an isolated environment via docker on a host system. 

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

For users of VS Code, the included `settings.json` should ensure
that Ruff problems appear while you edit, and formatting is applied
automatically on save.


## Testing

The project uses pytest for testing. To run the tests, use the following command:

```bash
poetry run pytest
```

If you are using VS Code, you can run the tests using the Test Explorer that is installed with the [Python extension](https://code.visualstudio.com/docs/python/testing).

Testing ML models poses a couple of problems: loading and running models may be very time consuming, and you may wish to run tests on systems that lack hardware support necessary for the models, for example a subnotebook without a GPU or a CI/CD server. To solve this issue, we created a **deployment test cache**. See [the documentation](docs/deployment_test_cache.md).


## Databases
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

## License
Aana SDK is licensed under the [Apache License 2.0](./LICENSE.md). Commercial licensing options are also available.

## Contributing
We welcome contributions from the community to enhance Aana SDK's functionality and usability. Feel free to open issues for bug reports, feature requests, or submit pull requests to contribute code improvements.

We have adopted the [Contributor Convenant](https://www.contributor-covenant.org/) as our code of conduct.
