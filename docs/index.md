---
hide:
  - navigation
---

<style>
.md-content .md-typeset h1 { 
  position: absolute;
  left: -999px;
}
</style>


[![Build Status](https://github.com/mobiusml/aana_sdk/actions/workflows/python-package.yml/badge.svg)](https://github.com/mobiusml/aana_sdk/actions/workflows/python-package.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/mobiusml/aana_sdk?tab=Apache-2.0-1-ov-file#readme)
[![Website](https://img.shields.io/badge/website-online-brightgreen.svg)](http://www.mobiuslabs.com)
[![PyPI version](https://img.shields.io/pypi/v/aana.svg)](https://pypi.org/project/aana/)
[![GitHub release](https://img.shields.io/github/v/release/mobiusml/aana_sdk.svg)](https://github.com/mobiusml/aana_sdk/releases)

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

**Model Deployment**:

  - Deploy models on a single machine or scale them across a cluster.

**API Generation**:

  - Automatically generate an API for your application based on the endpoints you define.
  - Input and output of the endpoints will be automatically validated.
  - Simply annotate the types of input and output of the endpoint functions.

**Predefined Types**:

  - Comes with a set of predefined types for various data such as images, videos, etc.

**Documentation Generation**:

  - Automatically generate documentation for your application based on the defined endpoints.

**Streaming Support**:

  - Stream the output of the endpoint to the client as it is generated.
  - Ideal for real-time applications and Large Language Models (LLMs).

**Task Queue Support**:

  - Run every endpoint you define as a task in the background without any changes to your code.

**Integrations**:  

   - Aana SDK has integrations with various machine learning models and libraries: Whisper, vLLM, Hugging Face Transformers, Deepset Haystack, and more to come (for more information see [Integrations](pages/integrations.md)).

## Installation

### Installing via PyPI

To install Aana SDK via PyPI, you can use the following command:

```bash
pip install aana
```

By default `aana` installs only the core dependencies. The deployment-specific dependencies are not installed by default. You have two options:
- Install the dependencies manually. You will be prompted to install the dependencies when you try to use a deployment that requires them.
- Use extras to install all dependencies. Here are the available extras:
  - `all`: Install dependencies for all deployments.
  - `vllm`: Install dependencies for the vLLM deployment.
  - `asr`: Install dependencies for the Automatic Speech Recognition (Whisper) deployment and other ASR models (diarization, voice activity detection, etc.).
  - `transformers`: Install dependencies for the Hugging Face Transformers deployment. There are multiple deployments that use Transformers.
  - `hqq`: Install dependencies for Half-Quadratic Quantization (HQQ) deployment.

For example, to install all dependencies, you can use the following command:
    
```bash 
pip install aana[all]
```

For optimal performance install [PyTorch](https://pytorch.org/get-started/locally/) version >=2.1 appropriate for your system. You can skip it, but it will install a default version that may not make optimal use of your system's resources, for example, a GPU or even some SIMD operations. Therefore we recommend choosing your PyTorch package carefully and installing it manually.

Some models use Flash Attention. Install Flash Attention library for better performance. See [flash attention installation instructions](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) for more details and supported GPUs.

### Installing from GitHub

1. Clone the repository.

```bash
git clone https://github.com/mobiusml/aana_sdk.git
```

2. Install additional libraries.

For optimal performance install [PyTorch](https://pytorch.org/get-started/locally/) version >=2.1 appropriate for your system. You can continue directly to the next step, but it will install a default version that may not make optimal use of your system's resources, for example, a GPU or even some SIMD operations. Therefore we recommend choosing your PyTorch package carefully and installing it manually.

Some models use Flash Attention. Install Flash Attention library for better performance. See [flash attention installation instructions](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) for more details and supported GPUs.

3. Install the package with uv.

The project is managed with [uv](https://docs.astral.sh/uv/), a fast Python package manager. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) on how to install it on your system.

It will install the package in the virtual environment created by uv. Add `--extra all` to install all extra dependencies.

```bash
uv sync --extra all
```

For the development environment, it is recommended to install all extras and dev and test dependencies. You can do this by running the following command:

```bash
uv sync --extra all --extra dev --extra tests
```

## Integrations    
- [Integrations](pages/integrations.md): Overview of the available predefined deployments like Whisper, vLLM, Hugging Face Transformers, Haystack etc.
- [OpenAI API](pages/openai_api.md): Overview of the OpenAI-compatible Chat Completions API.

## Deployment
- [Docker](pages/docker.md): Instructions for using Docker with Aana SDK.
- [Serve Config Files](pages/serve_config_files.md): Information about [Serve Config Files](https://docs.ray.io/en/latest/serve/production-guide/config.html#serve-config-files) for production deployment, how to build them, and deploy applications using them.
- [Cluster Setup](pages/cluster_setup.md): Instructions for setting up a Ray cluster for deployment.

## Configuration
- [Settings](pages/settings.md): Documentation on the available settings and configuration options for the project.

## Development
- [Code Overview](pages/code_overview.md): An overview of the structure of the project.
- [Development Environment](pages/dev_environment.md): Instructions for setting up the development environment.
- [Code Standards](pages/code_standards.md): Learn about our coding standards and best practices for contributing to the project.
- [Database](pages/database.md): Learn how to use SQL databases in the project.
- [Testing](pages/testing.md): Information on how to run tests and write new tests.

## Getting Started

If you're new to the project, we recommend starting with the [Tutorial](pages/tutorial.md) to get a hands-on introduction. From there, you can explore the other documentation files based on your specific needs or interests.

For developers looking to contribute, make sure to review the [Code Standards](pages/code_standards.md) and [Development Guide](pages/development.md).

If you have any questions or need further assistance, please don't hesitate to reach out to our support team or community forums.

### Creating a New Application

You can quickly develop multimodal applications using Aana SDK's intuitive APIs and components.

If you want to start building a new application, you can use the following GitHub template: [Aana App Template](https://github.com/mobiusml/aana_app_template). It will help you get started with the Aana SDK and provide you with a basic structure for your application and its dependencies.

Let's create a simple application that transcribes a video. The application will download a video from YouTube, extract the audio, and transcribe it using an ASR model.

Aana SDK already provides a deployment for ASR (Automatic Speech Recognition) based on the Whisper model. We will use this [deployment](#deployments) in the example.

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
        await super().initialize()
        self.asr_handle = await AanaDeploymentHandle.create("asr_deployment")

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

To get an overview of the Ray cluster, you can use the Ray Dashboard. The Ray Dashboard is available at `http://127.0.0.1:8265` by default. You can see the status of the Ray cluster, the resources used, running applications and deployments, logs, and more. It is a useful tool for monitoring and debugging your applications. See [Ray Dashboard documentation](https://docs.ray.io/en/latest/ray-observability/getting-started.html) for more information.

Let's transcribe [Gordon Ramsay's perfect scrambled eggs tutorial](https://www.youtube.com/watch?v=VhJFyyukAzA) using the application.

```bash
curl -X POST http://127.0.0.1:8000/video/transcribe -Fbody='{"video":{"url":"https://www.youtube.com/watch?v=VhJFyyukAzA"}}'
```

This will return the full transcription of the video, transcription for each segment, and transcription info like identified language. You can also use the [Swagger UI](http://127.0.0.1:8000/docs) to send the request.

### Running Example Applications

We provide a few example applications that demonstrate the capabilities of Aana SDK.

- [Chat with Video](https://github.com/mobiusml/aana_chat_with_video): A multimodal chat application that allows users to upload a video and ask questions about the video content based on the visual and audio information. See [Chat with Video Demo notebook](https://github.com/mobiusml/aana_chat_with_video/blob/main/notebooks/chat_with_video_demo.ipynb) to see how to use the application.
- [Summarize Video](https://github.com/mobiusml/aana_summarize_video): An Aana application that summarizes a video by extracting transcription from the audio and generating a summary using a Language Model (LLM). This application is a part of the [tutorial](https://mobiusml.github.io/aana_sdk/pages/tutorial/) on how to build multimodal applications with Aana SDK.


See the README files of the applications for more information on how to install and run them.

The full list of example applications is available in the [Aana Examples](https://github.com/mobiusml/aana_examples) repository. You can use these examples as a starting point for building your own applications.

### Main components

There are three main components in Aana SDK: deployments, endpoints, and AanaSDK.

#### Deployments

Deployments are the building blocks of Aana SDK. They represent the machine learning models that you want to deploy. Aana SDK comes with a set of predefined deployments that you can use or you can define your own deployments. See [Integrations](pages/integrations.md) section for more information about predefined deployments.

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

See [Model Hub](./pages/model_hub/index.md) for a collection of configurations for different models that can be used with the predefined deployments.

#### Endpoints

Endpoints define the functionality of your application. They allow you to connect multiple deployments (models) to each other and define the input and output of your application.

Each endpoint is defined as a class that inherits from the `Endpoint` class. The class has two main methods: `initialize` and `run`.

For example, you can define an endpoint that transcribes a video like this:

```python
class TranscribeVideoEndpoint(Endpoint):
    """Transcribe video endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        await super().initialize()
        self.asr_handle = await AanaDeploymentHandle.create("asr_deployment")

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


## API

Aana SDK uses form data for API requests, which allows sending both binary data and structured fields in a single request. The request body is sent as a JSON string in the `body` field, and any binary data is sent as files.

### Making API Requests

You can send requests to the SDK endpoints with only structured data or a combination of structured data and binary data.

#### Only Structured Data
When your request includes only structured data, you can send it as a JSON string in the `body` field.

- **cURL Example:**
    ```bash
    curl http://127.0.0.1:8000/endpoint \
        -F body='{"input": "data", "param": "value"}'
    ```

- **Python Example:**
    ```python
    import json, requests

    url = "http://127.0.0.1:8000/endpoint"
    body = {
        "input": "data",
        "param": "value"
    }

    response = requests.post(
        url,
        data={"body": json.dumps(body)}
    )

    print(response.json())
    ```

#### With Binary Data
When your request includes binary files (images, audio, etc.), you can send them as files in the request and include the names of the files in the `body` field as a reference.

For example, if you want to send an image, you can use [`aana.core.models.image.ImageInput`](reference/models/media.md#aana.core.models.ImageInput) as the input type that supports binary data upload. The `content` field in the input type should be set to the name of the file you are sending.

You can send multiple files in a single request by including multiple files in the request and referencing them in the `body` field even if they are of different types.

- **cURL Example:**
    ```bash
    curl http://127.0.0.1:8000/process_images \
        -H "Content-Type: multipart/form-data" \
        -F body='{"image": {"content": "file1"}}' \
        -F file1="@image.jpeg"
    ```

- **Python Example:**
    ```python
    import json, requests 

    url = "http://127.0.0.1:8000/process_images"
    body = {
        "image": {"content": "file1"}
    }
    with open("image.jpeg", "rb") as file:
        files = {"file1": file}

        response = requests.post(
            url,
            data={"body": json.dumps(body)},
            files=files
        )

        print(response.text)
    ```