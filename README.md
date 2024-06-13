[![Python package](https://github.com/mobiusml/aana_sdk/actions/workflows/python-package.yml/badge.svg)](https://github.com/mobiusml/aana_sdk/actions/workflows/python-package.yml)

# Aana

Aana SDK is a powerful serving layer designed for constructing web applications utilizing advanced multimodal models and RAG (Retrieval-Augmented Generation) systems. It facilitates the deployment of large-scale machine learning models, including those for vision, audio, and language, enabling the development of multimodal content applications such as search engines, recommendation systems, and data insights platforms.

## Features

- *Multimodal Input Handling:* Aana SDK seamlessly handles various types of multimodal inputs, providing versatility in application development.
- *Streaming Support:* With streaming support for both input and output, Aana SDK ensures smooth data processing for real-time applications.
- *Scalability:* Leveraging the capabilities of Ray serve, Aana SDK allows the deployment of multimodal models and applications across clusters, ensuring scalability and efficient resource utilization.
- *Rapid Development:* Aana SDK enables developers to swiftly create robust multimodal applications in a Pythonic manner, utilizing its underlying components for fast configurations and RAG setups.

## Usage

### Installing via PyPI

To install Aana SDK via PyPI, you can use the following command:

```bash
pip install aana
```

Make sure you have the necessary dependencies installed, such as `libgl1` for OpenCV.

### Installing from Github

1. Clone this repository.

2. Install additional libraries.

```bash
apt update && apt install -y libgl1
```

You should also install [PyTorch](https://pytorch.org/get-started/locally/) version >=2.1 appropriate for your system. You can continue directly to the next step, but it will install a default version that may not make optimal se of your system's resources, for example a GPU  or even some SIMD operations. Therefore we recommend installing choosing your PyTorch package carefully and installing it manually.

3. Install the package with poetry.

The project is managed with [Poetry](https://python-poetry.org/docs/). See the [Poetry installation instructions](https://python-poetry.org/docs/#installation) on how to install it on your system.

It will install the package and all dependencies in a virtual environment.

```bash
sh install.sh
```

### Creating Application

You can quickly develop multimodal applications using Aana SDK's intuitive APIs and components:


Let's create a simple application that transcribes a video using the Aana SDK. The application will download a video from YouTube, extract the audio, and transcribe it using an ASR model.

Aana SDK already provides a deployment for ASR (Automatic Speech Recognition) based on the Whisper model. We will use this deployment in the example.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # If you have a GPU, set the GPU ID here.

from aana.api.api_generation import Endpoint
from aana.core.models.video import VideoInput
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.whisper_deployment import (
    WhisperComputeType,
    WhisperConfig,
    WhisperDeployment,
    WhisperModelSize,
    WhisperOutput,
)
from aana.integrations.external.yt_dlp import download_video
from aana.processors.remote import run_remote
from aana.processors.video import extract_audio
from aana.sdk import AanaSDK


# Define the model deployments.
asr_deployment = WhisperDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.25},
    user_config=WhisperConfig(
        model_size=WhisperModelSize.MEDIUM,
        compute_type=WhisperComputeType.FLOAT16,
    ).model_dump(mode="json"),
)
deployments = [{"name": "asr_deployment", "instance": asr_deployment}]


# Define the endpoint to transcribe the video.
class TranscribeVideoEndpoint(Endpoint):
    """Transcribe video endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.asr_handle = await AanaDeploymentHandle.create("asr_deployment")
        await super().initialize()

    async def run(self, video: VideoInput) -> WhisperOutput:
        """Transcribe video."""
        video_obj = await run_remote(download_video)(video_input=video)
        audio = extract_audio(video=video_obj)
        transcription = await self.asr_handle.transcribe(audio=audio)
        return transcription

endpoints = [
    {
        "name": "transcribe_video",
        "path": "/video/transcribe",
        "summary": "Transcribe a video",
        "endpoint_cls": TranscribeVideoEndpoint,
    },
]

if __name__ == "__main__":
    aana_app = AanaSDK(name="transcribe_video_app")
    aana_app.connect(host="127.0.0.1", port=8000, show_logs=False)

    for deployment in deployments:
        aana_app.register_deployment(
            name=deployment["name"],
            instance=deployment["instance"],
        )

    for endpoint in endpoints:
        aana_app.register_endpoint(
            name=endpoint["name"],
            path=endpoint["path"],
            summary=endpoint["summary"],
            endpoint_cls=endpoint["endpoint_cls"],
        )

    aana_app.migrate()
    aana_app.deploy(blocking=True)
```

You can copy the code above to a Python file, for example `app.py`, and run it or run it in a Jupyter notebook.

Once the application is running, you will see the message `Deployed successfully.` in the logs. You can now send a request to the application to transcribe a video.

Let's transcribe [Gordon Ramsay's perfect scrambled eggs tutorial](https://www.youtube.com/watch?v=VhJFyyukAzA) using the application.

```bash
curl -X POST http://127.0.0.1:8000/video/transcribe -Fbody='{"video":{"url":"https://www.youtube.com/watch?v=VhJFyyukAzA"}}'
```

This will return the full transcription of the video, transcription for each segment, and transcription info like identified language. You can also use the [Swagger UI](http://127.0.0.1:8000/docs) to send the request.

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

## Running Serve Config Files

When you are running the Aana application using the Serve config files, you need to run the migrations to create the database tables for the application. To run the migrations, use the following command:

```bash
poetry run aana migrate aana.projects.chat_with_video.app:aana_app
```

Once you have generated YAML files in the previous step, you can run them with `serve deploy`.

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
