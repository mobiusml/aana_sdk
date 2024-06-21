[![Python package](https://github.com/mobiusml/aana_sdk/actions/workflows/python-package.yml/badge.svg)](https://github.com/mobiusml/aana_sdk/actions/workflows/python-package.yml)

# Aana

Aana SDK is a powerful framework for building multimodal applications. It facilitates the large-scale deployment of machine learning models, including those for vision, audio, and language, and supports Retrieval-Augmented Generation (RAG) systems. This enables the development of advanced applications such as search engines, recommendation systems, and data insights platforms.

The SDK is designed according to the following principles:

- **Reliability**: Aana is designed to be reliable and robust. It is built to be fault-tolerant and to handle failures gracefully.
- **Scalability**: Aana is designed to be scalable. It is built on top of Ray, a distributed computing framework, and can be easily scaled to multiple servers.
- **Efficiency**: Aana is designed to be efficient. It is built to be fast and parallel and to use resources efficiently.
- **Easy to Use**: Aana is designed to be easy to use by developers. It is built to be modular, with a lot of automation and abstraction.

The SDK is still in development, and not all features are fully implemented. We are constantly working on improving the SDK, and we welcome any feedback or suggestions.

## Why use Aana SDK?

Nowadays, it is getting easier to experiment with machine learning models and build prototypes. However, deploying these models at scale and integrating them into real-world applications is still a challenge. 

Aana SDK simplifies this process by providing a framework that allows:
- Deploy and scale machine learning models on a single machine or a cluster.
- Build multimodal applications that combine multiple different machine learning models.

### Key Features

- **Model Deployment**:
  - Deploy models on a single machine or scale them across a cluster.

- **API Generation**:
  - Automatically generate an API for your application based on the endpoints you define.
  - Input and output of the endpoints will be automatically validated.
  - Simply annotate the types of input and output of the endpoint functions.
- **Predefined Types**:
  - Comes with a set of predefined types for various data such as images, videos, etc.

- **Documentation Generation**:
  - Automatically generate documentation for your application based on the defined endpoints.

- **Streaming Support**:
  - Stream the output of the endpoint to the client as it is generated.
  - Ideal for real-time applications and Large Language Models (LLMs).

- **Task Queue Support**:
  - Run every endpoint you define as a task in the background without any changes to your code.
- **Integrations**:  
   - Aana SDK has integrations with various machine learning models and libraries: Whisper, vLLM, Hugging Face Transformers, Deepset Haystack, and more to come (for more information see [Integrations](docs/integrations.md)). 

## Installation

### Installing via PyPI

To install Aana SDK via PyPI, you can use the following command:

```bash
pip install aana
```

Make sure you have the necessary dependencies installed, such as `libgl1` for OpenCV.

### Installing from GitHub

1. Clone the repository.

```bash
git clone https://github.com/mobiusml/aana_sdk.git
```

2. Install additional libraries.

```bash
apt update && apt install -y libgl1
```

You should also install [PyTorch](https://pytorch.org/get-started/locally/) version >=2.1 appropriate for your system. You can continue directly to the next step, but it will install a default version that may not make optimal use of your system's resources, for example, a GPU  or even some SIMD operations. Therefore we recommend choosing your PyTorch package carefully and installing it manually.

3. Install the package with poetry.

The project is managed with [Poetry](https://python-poetry.org/docs/). See the [Poetry installation instructions](https://python-poetry.org/docs/#installation) on how to install it on your system.

It will install the package and all dependencies in a virtual environment.

```bash
sh install.sh
```

## Getting Started

### Creating a New Application

You can quickly develop multimodal applications using Aana SDK's intuitive APIs and components.

Let's create a simple application that transcribes a video. The application will download a video from YouTube, extract the audio, and transcribe it using an ASR model.

Aana SDK already provides a deployment for ASR (Automatic Speech Recognition) based on the Whisper model. We will use this [deployment](#Deployments) in the example.

```python
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
    ray_actor_options={"num_gpus": 0.25}, # Remove this line if you want to run Whisper on a CPU.
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

aana_app = AanaSDK(name="transcribe_video_app")

for deployment in deployments:
    aana_app.register_deployment(**deployment)

for endpoint in endpoints:
    aana_app.register_endpoint(**endpoint)

if __name__ == "__main__":
    aana_app.connect(host="127.0.0.1", port=8000, show_logs=False)  # Connects to the Ray cluster or starts a new one.
    aana_app.migrate()                                              # Runs the migrations to create the database tables.
    aana_app.deploy(blocking=True)                                  # Deploys the application.
```

You have a few options to run the application:
- Copy the code above and run it in a Jupyter notebook.
- Save the code to a Python file, for example `app.py`, and run it as a Python script: `python app.py`.
- Save the code to a Python file, for example `app.py`, and run it using the Aana CLI: `aana deploy app:aana_app --host 127.0.0.1 --port 8000 --hide-logs`.

Once the application is running, you will see the message `Deployed successfully.` in the logs. You can now send a request to the application to transcribe a video.

Let's transcribe [Gordon Ramsay's perfect scrambled eggs tutorial](https://www.youtube.com/watch?v=VhJFyyukAzA) using the application.

```bash
curl -X POST http://127.0.0.1:8000/video/transcribe -Fbody='{"video":{"url":"https://www.youtube.com/watch?v=VhJFyyukAzA"}}'
```

This will return the full transcription of the video, transcription for each segment, and transcription info like identified language. You can also use the [Swagger UI](http://127.0.0.1:8000/docs) to send the request.

### Running Example Applications

Aana SDK comes with a set of example applications that demonstrate the capabilities of the SDK. You can run the example applications using the Aana CLI.

The following applications are available:
- `chat_with_video`: A multimodal chat application that allows users to upload a video and ask questions about the video content based on the visual and audio information.
- `whisper`: An application that demonstrates the Whisper model for automatic speech recognition (ASR).
- `llama2`: An application that deploys LLaMa2 7B Chat model.

To run an example application, use the following command:

```bash
aana deploy aana.projects.<app_name>.app:aana_app
```

For example, to run the `whisper` application, use the following command:

```bash
aana deploy aana.projects.whisper.app:aana_app
```

> **⚠️ Warning**
>
> The example applications require a GPU to run. 
>
> The applications will detect the available GPU automatically but you need to make sure that `CUDA_VISIBLE_DEVICES` is set correctly.
> 
> Sometimes `CUDA_VISIBLE_DEVICES` is set to an empty string and the application will not be able to detect the GPU. Use `unset CUDA_VISIBLE_DEVICES` to unset the variable.
> 
> You can also set the `CUDA_VISIBLE_DEVICES` environment variable to the GPU index you want to use: `export CUDA_VISIBLE_DEVICES=0`.
>
> Different applications have different requirements for the GPU memory:
> - `chat_with_video` requires at least 48GB.
> - `llama2` requires at least 16GB.
> - `whisper` requires at least 4GB.



### Main components

There are three main components in Aana SDK: deployments, endpoints, and AanaSDK.

#### Deployments

Deployments are the building blocks of Aana SDK. They represent the machine learning models that you want to deploy. Aana SDK comes with a set of predefined deployments that you can use or you can define your own deployments. See [Integrations](#integrations) section for more information about predefined deployments.

Each deployment has a main class that defines it and a configuration class that allows you to specify the deployment parameters.

For example, we have a predefined deployment for the Whisper model that allows you to transcribe audio. You can define the deployment like this:

```python
from aana.deployments.whisper_deployment import WhisperDeployment, WhisperConfig, WhisperModelSize, WhisperComputeType

asr_deployment = WhisperDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.25},
    user_config=WhisperConfig(model_size=WhisperModelSize.MEDIUM, compute_type=WhisperComputeType.FLOAT16).model_dump(mode="json"),
)
```

#### Endpoints

Endpoints define the functionality of your application. They allow you to connect multiple deployments (models) to each other and define the input and output of your application.

Each endpoint is defined as a class that inherits from the `Endpoint` class. The class has two main methods: `initialize` and `run`.

For example, you can define an endpoint that transcribes a video like this:

```python
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
```

#### AanaSDK

AanaSDK is the main class that you use to build your application. It allows you to deploy the deployments and endpoints you defined and start the application.

For example, you can define an application that transcribes a video like this:

```python
aana_app = AanaSDK(name="transcribe_video_app")

aana_app.register_deployment(name="asr_deployment", instance=asr_deployment)
aana_app.register_endpoint(
    name="transcribe_video",
    path="/video/transcribe",
    summary="Transcribe a video",
    endpoint_cls=TranscribeVideoEndpoint,
)

aana_app.connect()  # Connects to the Ray cluster or starts a new one.
aana_app.migrate()  # Runs the migrations to create the database tables.
aana_app.deploy()   # Deploys the application.
```

All you need to do is define the deployments and endpoints you want to use in your application, and Aana SDK will take care of the rest.

## Serve Config Files

The [Serve Config Files](https://docs.ray.io/en/latest/serve/production-guide/config.html#serve-config-files) is the recommended way to deploy and update your applications in production. Aana SDK provides a way to build the Serve Config Files for the Aana applications. See the [Serve Config Files documentation](docs/serve_config_files.md) on how to build and deploy the applications using the Serve Config Files.


## Run with Docker

You can deploy example applications using Docker. See the [documentation on how to run Aana SDK with Docker](docs/docker.md).

## License

Aana SDK is licensed under the [Apache License 2.0](./LICENSE). Commercial licensing options are also available.

## Contributing

We welcome contributions from the community to enhance Aana SDK's functionality and usability. Feel free to open issues for bug reports, feature requests, or submit pull requests to contribute code improvements.

Before contributing, please read our [Code Standards](docs/code_standards.md) and [Development Documentation](docs/development.md).

We have adopted the [Contributor Covenant](https://www.contributor-covenant.org/) as our code of conduct.
