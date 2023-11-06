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
