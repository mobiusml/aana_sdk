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
    

# How to Create a New Project with Aana SDK

Aana SDK is a powerful framework for building multimodal applications. It facilitates the large-scale deployment of machine learning models, including those for vision, audio, and language, and supports Retrieval-Augmented Generation (RAG) systems. This enables the development of advanced applications such as search engines, recommendation systems, and data insights platforms.

Aana SDK comes with a set of example applications that demonstrate the capabilities of the SDK. These applications can be used as a reference to build your own applications. See the [projects](https://github.com/mobiusml/aana_sdk/tree/main/aana/projects/) directory for the example applications.

If you want to start building a new application, you can use the following GitHub template: [Aana App Template](https://github.com/mobiusml/aana_app_template). It will help you get started with the Aana SDK and provide you with a basic structure for your application and its dependencies.

In this tutorial, we will walk you through the process of creating a new project with Aana SDK. By the end of this tutorial, you will have a runnable application that transcribes a video and summarizes the transcript using a Language Model (LLM). We will use [the video transcription application](./../index.md#creating-a-new-application) as a starting point and extend it to include the LLM model for summarization and a new endpoints.



## Prerequisites

Before you begin, make sure you have a working installation of Aana SDK. See the [installation instructions](./../index.md#installation) for more information.

## Video Transcription Application

First, let's review a video transcription application. Here is the code for it:

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

## Running the Application

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

## Application Components

The application consists of 3 main components: the deployment, the endpoint, and the application itself. 

### Deployments

Deployments are the building blocks of Aana SDK. They represent the machine learning models that you want to deploy. Aana SDK comes with a set of predefined deployments that you can use or you can define your own deployments. See [Integrations](integrations.md) for more information about predefined deployments.

Each deployment has a main class that defines it and a configuration class that allows you to specify the deployment parameters.

In the example above, we define a deployment for the Whisper model that allows you to transcribe audio. The deployment is defined as follows:

```python
from aana.deployments.whisper_deployment import WhisperDeployment, WhisperConfig, WhisperModelSize, WhisperComputeType

asr_deployment = WhisperDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.25},
    user_config=WhisperConfig(model_size=WhisperModelSize.MEDIUM, compute_type=WhisperComputeType.FLOAT16).model_dump(mode="json"),
)
```

### Endpoints

Endpoints define the functionality of your application. They allow you to connect multiple deployments (models) to each other and define the input and output of your application.

Each endpoint is defined as a class that inherits from the `Endpoint` class. The class has two main methods: `initialize` and `run`.

`initialize` method contains actions that need to be performed before the endpoint is run. For example, you can create handles for the deployments that the endpoint will use.

`run` method is the main method of the endpoint that is called when the endpoint receives a request. 

The `run` method should be annotated with the input and output types. The types should be pydantic models. Return type should be a typed dictionary where the keys are strings and the values are pydantic models.

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

### Application

AanaSDK is the main class that represents the application. It allows you to deploy the deployments and endpoints you defined and start the application.

For example, you can define an application that transcribes a video like this:

```python
aana_app = AanaSDK(name="transcribe_video_app")

# Register the ASR deployment.
aana_app.register_deployment(name="asr_deployment", instance=asr_deployment)

# Register the transcribe video endpoint.
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

### Connecting to the DeploymentS

Once the application is deployed, you can also access the deployments from other processes or applications using the `AanaSDK` class. For example, you can access the ASR deployment like this:

```python
from aana.sdk import AanaSDK
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle

aana_app = AanaSDK().connect()
asr_handle = AanaDeploymentHandle.create("asr_deployment")
```

The name of the deployment used to create the handle should match the name of the deployment when `register_deployment` was called.

This is quite useful for testing or debugging your application.

## Transcript Summarization Application

Now that we have reviewed the video transcription application, let's build on it to create a video transcript summarization application. For the summarization we will need a few extra components:

- An LLM model to summarize the transcript.
- An endpoint to summarize the transcript.

### LLM Model

LLM model can be registered as a deployment in the application. Aana SDK provides two deployments that can be used to deploy LLM models:

- `HfTextGenerationDeployment`: A deployment based on Hugging Face Transformers library.
- `VLLMDeployment`: A deployment based on vLLM library.

Both deployments have the same interface and can be used interchangeably. In this example, we will use `HfTextGenerationDeployment`.

Here is an example of how to define an LLM deployment:

```python
from aana.deployments.hf_text_generation_deployment import HfTextGenerationConfig, HfTextGenerationDeployment
llm_deployment = HfTextGenerationDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.25},
    user_config=HfTextGenerationConfig(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        model_kwargs={
            "trust_remote_code": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=False, load_in_4bit=True
            ),
        },
    ).model_dump(mode="json"),
)
```

Let's take a closer look at the configuration options:

- `HfTextGenerationDeployment` is the deployment class.
- `num_replicas=1` specifies the number of replicas to deploy. If you want to scale the deployment, you can increase this number to deploy more replicas on more GPUs or nodes.
- `ray_actor_options={"num_gpus": 0.25}` specifies the number of GPUs that each replica requires. This will be used to allocate resources on the Ray cluster but keep in mind that it will not limit the deployment to use only this amount of GPUs, it's only used for resource allocation. If you want to run the deployment on a CPU, you can remove this line.
- `user_config` is the configuration object for the deployment. In this case, we are using `HfTextGenerationConfig` that is specific to `HfTextGenerationDeployment`. 
- `model_id` is the Hugging Face model ID that you want to deploy. You can find the model ID on the Hugging Face model hub.
- `model_kwargs` is a dictionary of additional keyword arguments that you want to pass to the model. In this case, we are using `trust_remote_code` as it's required by the model and `quantization_config` to load the model in 4-bit precision. Check the model documentation for the required keyword arguments.

### Summarization Endpoint

Now that we have the LLM, we can define an endpoint to summarize the transcript and use the LLM deployment in it.

Here is an example of how to define a summarization endpoint:

```python
class SummarizeVideoEndpointOutput(TypedDict):
    """Summarize video endpoint output."""
    summary: Annotated[str, Field(description="The summary of the video.")]

class SummarizeVideoEndpoint(Endpoint):
    """Summarize video endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        await super().initialize()
        self.asr_handle = await AanaDeploymentHandle.create("asr_deployment")
        self.llm_handle = await AanaDeploymentHandle.create("llm_deployment")

    async def run(self, video: VideoInput) -> SummarizeVideoEndpointOutput:
        """Summarize video."""
        video_obj = await run_remote(download_video)(video_input=video)
        audio = extract_audio(video=video_obj)
        transcription = await self.asr_handle.transcribe(audio=audio)
        transcription_text = transcription["transcription"].text
        dialog = ChatDialog(
            messages=[
                ChatMessage(
                    role="system",
                    content="You are a helpful assistant that can summarize audio transcripts.",
                ),
                ChatMessage(
                    role="user",
                    content=f"Summarize the following video transcript into a list of bullet points: {transcription_text}",
                ),
            ]
        )
        summary_response = await self.llm_handle.chat(dialog=dialog)
        summary_message: ChatMessage = summary_response["message"]
        summary = summary_message.content
        return {"summary": summary}
```

In this example, we define a new endpoint class `SummarizeVideoEndpoint` that inherits from the `Endpoint` class. The class has two main methods: `initialize` and `run`.

In the `initialize` method, we create handles for the ASR and LLM deployments.

`run` method is the main method of the endpoint. The `run` method should be annotated with the input and output types. In this case, we only have one input `video` of type `VideoInput` and the output is a dictionary with one key `summary` of type `str`. The output type is defined as a `TypedDict` called `SummarizeVideoEndpointOutput`.

The `run` method performs the following steps:

- Downloading the video: We use the `download_video` function from the `yt_dlp` integration to download the video. The `run_remote` function is used to run the function as a remote task in the Ray cluster. It is useful to run heavy tasks with `run_remote` to offload the main thread.
- Extracting the audio from the video: We use the `extract_audio` function from the `video` processor to extract the audio.
- Transcribing the audio: We call ASR deployment to transcribe the audio. Here we use ASR deployment handle that we created in the `initialize` method. Calling the deployment is as simple as calling a method on the handle. The call is asynchronous so we use `await` to wait for the result.
- Creating a chat dialog: We create a chat dialog with a system message and a user message. The user message contains the transcription text that we want to summarize.
- Summarizing the transcript: We call the LLM deployment to summarize the transcript. Here we use the LLM deployment handle that we created in the `initialize` method. The call is similar to the ASR deployment call. We pass the chat dialog to the LLM deployment and wait for the response with `await`.
- Returning the summary: We return the summary as a dictionary with one key `summary`. Just like the type annotation tells us.
- The summary has type `Annotated[str, Field(description="The summary of the video.")]` which means it's a string with a description. The description is used to generate the API documentation. You can use `Annotated` and `Field` to add more metadata to your types (input and output) to generate better API documentation.

### Extending the Application

Now that we have the LLM deployment and the summarization endpoint, we can extend the application to include the LLM deployment and the summarization endpoint.

Here is an example of how to extend the application:

```python
aana_app = AanaSDK(name="summarize_video_app")

# Register the ASR deployment.
aana_app.register_deployment(name="asr_deployment", instance=asr_deployment)
# Register the LLM deployment.
aana_app.register_deployment(name="llm_deployment", instance=llm_deployment)

# Register the transcribe video endpoint.
aana_app.register_endpoint(
    name="transcribe_video",
    path="/video/transcribe",
    summary="Transcribe a video",
    endpoint_cls=TranscribeVideoEndpoint,
)
# Register the summarize video endpoint.
aana_app.register_endpoint(
    name="summarize_video",
    path="/video/summarize",
    summary="Summarize a video transcript",
    endpoint_cls=SummarizeVideoEndpoint,
)

aana_app.connect()  # Connects to the Ray cluster or starts a new one.
aana_app.migrate()  # Runs the migrations to create the database tables.
aana_app.deploy()   # Deploys the application.
```

In this example, we define a new application called `summarize_video_app`. We register the ASR and LLM deployments and the transcribe and summarize video endpoints.

Here is the full code for the application:

```python
from collections.abc import AsyncGenerator
from typing import Annotated, TypedDict

from pydantic import Field
from transformers import BitsAndBytesConfig

from aana.api.api_generation import Endpoint
from aana.core.models.chat import ChatDialog, ChatMessage
from aana.core.models.video import VideoInput
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.hf_text_generation_deployment import (
    HfTextGenerationConfig,
    HfTextGenerationDeployment,
)
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

llm_deployment = HfTextGenerationDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.25},
    user_config=HfTextGenerationConfig(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        model_kwargs={
            "trust_remote_code": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=False, load_in_4bit=True
            ),
        },
    ).model_dump(mode="json"),
)


deployments = [
    {"name": "asr_deployment", "instance": asr_deployment},
    {"name": "llm_deployment", "instance": llm_deployment},
]


class SummarizeVideoEndpointOutput(TypedDict):
    """Summarize video endpoint output."""

    summary: Annotated[str, Field(description="The summary of the video.")]


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


class SummarizeVideoEndpoint(Endpoint):
    """Summarize video endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        await super().initialize()
        self.asr_handle = await AanaDeploymentHandle.create("asr_deployment")
        self.llm_handle = await AanaDeploymentHandle.create("llm_deployment")

    async def run(self, video: VideoInput) -> SummarizeVideoEndpointOutput:
        """Summarize video."""
        video_obj = await run_remote(download_video)(video_input=video)
        audio = extract_audio(video=video_obj)
        transcription = await self.asr_handle.transcribe(audio=audio)
        transcription_text = transcription["transcription"].text
        dialog = ChatDialog(
            messages=[
                ChatMessage(
                    role="system",
                    content="You are a helpful assistant that can summarize audio transcripts.",
                ),
                ChatMessage(
                    role="user",
                    content=f"Summarize the following video transcript into a list of bullet points: {transcription_text}",
                ),
            ]
        )
        summary_response = await self.llm_handle.chat(dialog=dialog)
        summary_message: ChatMessage = summary_response["message"]
        summary = summary_message.content
        return {"summary": summary}

endpoints = [
    {
        "name": "transcribe_video",
        "path": "/video/transcribe",
        "summary": "Transcribe a video",
        "endpoint_cls": TranscribeVideoEndpoint,
    },
    {
        "name": "summarize_video",
        "path": "/video/summarize",
        "summary": "Summarize a video",
        "endpoint_cls": SummarizeVideoEndpoint,
    },
]

aana_app = AanaSDK(name="summarize_video_app")

for deployment in deployments:
    aana_app.register_deployment(**deployment)

for endpoint in endpoints:
    aana_app.register_endpoint(**endpoint)

if __name__ == "__main__":
    aana_app.connect()  # Connects to the Ray cluster or starts a new one.
    aana_app.migrate()  # Runs the migrations to create the database tables.
    aana_app.deploy()  # Deploys the application.
```

Now you can run the application as described in the previous section and send a request to summarize the video transcript.

```bash
curl -X POST http://127.0.0.1:8000/video/summarize -Fbody='{"video":{"url":"https://www.youtube.com/watch?v=VhJFyyukAzA"}}'
```

We will get a response with the summary key containing the bullet points on how to make perfect scrambled eggs.

```json
- Never season eggs before cooking to prevent them from turning gray and watery
- Fill the pan 3/4 full and clean it as you go to ensure a clean cooking surface
- Do not whisk eggs before cooking to avoid over-mixing
- Cook eggs off the heat and continue stirring to control the cooking process
- Use butter for flavor and to create a non-stick surface
- Stop cooking eggs as soon as they reach the desired level of doneness to prevent rubbery texture
- Add a teaspoon of sour cream or creme fraiche for extra creaminess
- Season with salt and pepper at the end for taste
- Stir continuously for fluffy and light scrambled eggs
- Avoid seasoning or whisking eggs before cooking for the best results
- Clean the pan throughout the cooking process for optimal results
- Use a non-stick pan and a spatula for easy cooking and stirring
- Cook eggs off the heat and continue stirring to maintain the desired texture
- Add sour cream or creme fraiche for added creaminess and flavor
```

## Streaming LLM Output

In the example above, we used the `chat` method to interact with the LLM model. The `chat` method returns a single response message once the model has finished processing the input. However, in some cases, you may want to stream the output from the LLM model as it is being generated.

Our LLM deployments support streaming output using the `chat_stream` method. It's implemented as an asynchronous generator that yields text as it is generated by the model. 

```python
async for chunk in llm_handle.chat_stream(dialog=dialog):
    print(chunk)
```

Now we can modify the `SummarizeVideoEndpoint` to stream the output from the LLM model. For demonstration purposes, we will create a new endpoint `SummarizeVideoStreamEndpoint` that is a streaming version of the `SummarizeVideoEndpoint`.

```python
class SummarizeVideoStreamEndpointOutput(TypedDict):
    """Summarize video endpoint output."""

    text: Annotated[str, Field(description="The text chunk.")]

class SummarizeVideoStreamEndpoint(Endpoint):
    """Summarize video endpoint with streaming output."""

    async def initialize(self):
        """Initialize the endpoint."""
        await super().initialize()
        self.asr_handle = await AanaDeploymentHandle.create("asr_deployment")
        self.llm_handle = await AanaDeploymentHandle.create("llm_deployment")
       
    async def run(
        self, video: VideoInput
    ) -> AsyncGenerator[SummarizeVideoStreamEndpointOutput, None]:
        """Summarize video."""
        video_obj = await run_remote(download_video)(video_input=video)
        audio = extract_audio(video=video_obj)
        transcription = await self.asr_handle.transcribe(audio=audio)
        transcription_text = transcription["transcription"].text
        dialog = ChatDialog(
            messages=[
                ChatMessage(
                    role="system",
                    content="You are a helpful assistant that can summarize audio transcripts.",
                ),
                ChatMessage(
                    role="user",
                    content=f"Summarize the following video transcript into a list of bullet points: {transcription_text}",
                ),
            ]
        )
        async for chunk in self.llm_handle.chat_stream(dialog=dialog):
            chunk_text = chunk["text"]
            yield {"text": chunk_text}

```

The difference between the non-streaming version and the streaming version:

- `run` method is now an asynchronous generator that yields `SummarizeVideoEndpointOutput` objects: `AsyncGenerator[SummarizeVideoEndpointOutput, None]`. If you want endpoint to be able to stream output, you need to define the output type as an asynchronous generator to let the SDK know that the endpoint will be streaming output.
- We use the `chat_stream` method to interact with the LLM model. The `chat_stream` method returns an asynchronous generator that yields text as it is generated by the model. We use `async for` to iterate over the generator and yield the text chunks as they are generated.
- We use `yield` to yield the text chunks as they are generated by the LLM model instead of using `return` to return a single response.
- The output dictionary now contains a single key `text` that contains the text chunk generated by the LLM model instead of a single key `summary`.

That's it! Now you have a streaming version of the summarization endpoint and you can display the output as it is generated by the LLM model.

```python
python -c "
import requests, json;
[print(json.loads(c)['text'], end='') 
 for c in requests.post(
    'http://127.0.0.1:8000/video/summarize_stream', 
    data={'body': json.dumps({'video': {'url': 'https://www.youtube.com/watch?v=VhJFyyukAzA'}})}, 
    stream=True).iter_content(chunk_size=None)]
"
```

## App Template

The best way to start building a new application is to use [Aana App Template](https://github.com/mobiusml/aana_app_template). It is a GitHub template repository that you can use to create a new repository with the same directory structure and files as the template. It will help you get started with the Aana SDK and provide you with a basic structure for your application and its dependencies.

Let's create a new project using the Aana App Template:

### Create a New Repository

Go to the [Aana App Template](https://github.com/new?template_name=aana_app_template). Choose a name for your repository and fill in the other details. Click on the "Create repository" button. GitHub will create a new repository with the same directory structure and files as the template. There is a GitHub Actions workflow that will run after the repository is created to modify the project name to match the repository name. It is pretty fast but make sure it's finished before you proceed.

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone <repository-url>
cd <repository-name>
```

### Create the Application

For our example project we need to adjust a few things.

#### Deployments

Define the deployments in the [`aana_summarize_video/configs/deployments.py`](https://github.com/mobiusml/aana_summarize_video/blob/main/aana_summarize_video/configs/deployments.py). Just like in the example above.

#### Endpoints

The endpoints are defined in the [`aana_summarize_video/endpoints`](https://github.com/mobiusml/aana_summarize_video/tree/main/aana_summarize_video/endpoints) directory. It is a good practice to define each endpoint in a separate file. In our case, we will define 3 endpoints in 3 separate files: `transcribe_video.py`, `summarize_video.py`, and `summarize_video_stream.py`.

We also need to register the endpoints. The list of endpoints is defined in the [`aana_summarize_video/configs/endpoints.py`](https://github.com/mobiusml/aana_summarize_video/blob/main/aana_summarize_video/configs/endpoints.py) file.

### Run the Application

Now you can run the application:

```bash
aana deploy aana_summarize_video.app:aana_app
```

Or if you want to be on the safe side, you can run the application with `poetry run` and CUDA_VISIBLE_DEVICES set to the GPU index you want to use:

```bash
CUDA_VISIBLE_DEVICES=0 poetry run aana deploy aana_summarize_video.app:aana_app
```

Once the application is running, you can send a request to transcribe and summarize a video as described in the previous sections.


## Conclusion

In this tutorial, we have walked you through the process of creating a new project with Aana SDK. We have reviewed the video transcription application and extended it to include the LLM model for summarization and a new endpoint. We have also demonstrated how to stream the output from the LLM model. You can use this tutorial as a reference to build your own applications with Aana SDK. 

The full code for the application is available in the [projects](https://github.com/mobiusml/aana_sdk/tree/main/aana/projects/summarize_transcript) directory.
