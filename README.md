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

### Installing from GitHub

1. Clone the repository.

```bash
git clone https://github.com/mobiusml/aana_sdk.git
```

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

### Run Example Application

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


### Creating Application

You can quickly develop multimodal applications using Aana SDK's intuitive APIs and components.

Let's create a simple application that transcribes a video. The application will download a video from YouTube, extract the audio, and transcribe it using an ASR model.

Aana SDK already provides a deployment for ASR (Automatic Speech Recognition) based on the Whisper model. We will use this deployment in the example.

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

## Serve Config Files

The [Serve Config Files](https://docs.ray.io/en/latest/serve/production-guide/config.html#serve-config-files) is the recommended way to deploy and update your applications in production. Aana SDK provides a way to build the Serve Config Files for the Aana applications.

### Building Serve Config Files

To build the Serve config file, run the following command:

```bash
aana build <app_module>:<app_name>
```

For example:

```bash
aana build aana.projects.chat_with_video.app:aana_app
```

The command will generate the Serve Config file and App Config file and save them in the project directory. You can then use these files to deploy the application using the Ray Serve CLI.

### Deploying with Serve Config Files

When you are running the Aana application using the Serve config files, you need to run the migrations to create the database tables for the application. To run the migrations, use the following command:

```bash
aana migrate <app_module>:<app_name>
```

For example:

```bash
aana migrate aana.projects.chat_with_video.app:aana_app
```

Before deploying the application, make sure you have the Ray cluster running. If you want to start a new Ray cluster on a single machine, you can use the following command:

```bash
ray start --head
```

For more info on how to start a Ray cluster, see the [Ray documentation](https://docs.ray.io/en/latest/ray-core/starting-ray.html#starting-ray-via-the-cli-ray-start).

To deploy the application using the Serve config files, use [`serve deploy`](https://docs.ray.io/en/latest/serve/advanced-guides/deploy-vm.html#serve-in-production-deploying) command provided by Ray Serve. For example:

```bash
serve deploy config.yaml
```

## Run with Docker

You can deploy example applications using Docker. 

1. Clone the repository.

2. Build the Docker image.

```bash
docker build -t aana:latest .
```

3. Run the Docker container.

```bash
docker run --rm --init -p 8000:8000 --gpus all -e TARGET="llama2" -v aana_cache:/root/.aana -v aana_hf_cache:/root/.cache/huggingface --name aana_instance aana:latest
```

Use the environment variable TARGET to specify the application you want to run. The available applications are `chat_with_video`, `whisper`, and `llama2`.

The first run might take a while because the models will be downloaded from the Internet and cached. The models will be stored in the `aana_cache` volume. The HuggingFace models will be stored in the `aana_hf_cache` volume. If you want to remove the cached models, remove the volume.

Once you see `Deployed successfully.` in the logs, the server is ready to accept requests.

You can change the port and gpus parameters to your needs.

The server will be available at http://localhost:8000.

The app documentation available as a [Swagger UI](http://localhost:8000/docs) and [ReDoc](http://localhost:8000/redoc).

5. Send a request to the server.

You can find examples in the [demo notebook](notebooks/demo.ipynb).

## License
Aana SDK is licensed under the [Apache License 2.0](./LICENSE.md). Commercial licensing options are also available.

## Contributing
We welcome contributions from the community to enhance Aana SDK's functionality and usability. Feel free to open issues for bug reports, feature requests, or submit pull requests to contribute code improvements.

Before contributing, please read our [Code Standards](docs/code_standards.md) and [Development Documentation](docs/development.md).

We have adopted the [Contributor Convenant](https://www.contributor-covenant.org/) as our code of conduct.
