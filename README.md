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
from typing_extensions import TypedDict

from aana.api.api_generation import Endpoint
from aana.sdk import AanaSDK
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle

from aana.deployments.hf_pipeline_deployment import HfPipelineConfig, HfPipelineDeployment


deployment = HfPipelineDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
    user_config=HfPipelineConfig(
        model_id="google-t5/t5-small",
        task="summarization",
    ).model_dump(mode="json"),
)

deployment_name = "aana_deployment"

class SummarizationOutput(TypedDict):
    """Simple class for summarization output.

    You can also use a Pydantic object here.
    """
    summary: str

class SummarizeTextEndpoint(Endpoint):
    """Class for an endoint for the app's API. In this case, text summary."""
    async def initialize(self):
        """Initializes the endpoint's deployment handles and any other state.'"""
        self.deployment_handle = await AanaDeploymentHandle.create(deployment_name)

    async def run(self, text: str) -> SummarizationOutput:
        """Method run for each endpoint call.

        Documentation for the endpoint (see below) is generated automatically based on
        this method's type annotations, so annotations for parameters and return types
        should a) exist, and b) be as specific as possible. Something like

        `async def run(self, arg: dict) -> dict: ...`

        will *work* but not be very helpful to API consumers."""

        result = await self.deployment_handle.call(text)
        return {"summary": result[0]["summary_text"]}

if __name__ == '__main__':
    # Construct an app instance
    aana_app = AanaSDK(name="demo app")
    # bind the app to a network address.
    # setting show_logs=`True` will produce a LOT of logs!
    aana_app.connect(port=9000, host="127.0.0.1", show_logs=True)

    aana_app.register_deployment(
        name="aana_deployment",
        instance=deployment,
    )

    aana_app.register_endpoint(
        name="summarize_text",
        path="/text/summarize",
        summary="Summarize a text",
        endpoint_cls=SummarizeTextEndpoint,
    )

    # Setting `blocking=False` will cause the app to exit as soon as it is set up, which may be useful for debugging initialization issues. 
    aana_app.deploy(blocking=True)
```

You can send a request like

```bash
curl -X POST 0.0.0.0:9000/text/summarize -F body='{"text": "Teachers of jurisprudence, when speaking of rights and claims, distinguish in a cause the question of right (quid juris) from the question of fact (quid facti), and while they demand proof of both, they give to the proof of the former, which goes to establish right or claim in law, the name of deduction. Now we make use of a great number of empirical conceptions, without opposition from any one, and consider ourselves, even without any attempt at deduction, justified in attaching to them a sense, and a supposititious signification, because we have always experience at hand to demonstrate their objective reality."}'
```

Here is an example with video transcription:
```python
from aana.configs.deployments import (
    whisper_medium_deployment,
)
from aana.projects.chat_with_video.endpoints import SimpleTranscribeVideoEndpoint
from aana.sdk import AanaSDK

deployments = [
    {
        "name": "asr_deployment",
        "instance": whisper_medium_deployment,
    }
]

endpoints = [
    {
        "name": "transcribe_video",
        "path": "/video/transcribe",
        "summary": "Transcribe a video as a stream",
        "endpoint_cls": SimpleTranscribeVideoEndpoint,
    },
]
if __name__ == '__main__':
    aana_app = AanaSDK(name="transcribe_video_app")
    aana_app.connect(host='127.0.0.1', port=9000, show_logs=True)

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

    aana_app.deploy(blocking=True)

```

```bash
curl -X POST 0.0.0.0:9000/video/transcribe -Fbody='{"video":{"url":"https://www.youtube.com/watch?v=CfX_su1AUwE"}}'

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

## Running Serve Config Files

Once you have generated YAML files in the previous step, you can run them with `ray serve deploy`.

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
