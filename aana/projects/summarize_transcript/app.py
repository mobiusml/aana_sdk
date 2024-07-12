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


class SummarizeVideoStreamEndpointOutput(TypedDict):
    """Summarize video endpoint output."""

    text: Annotated[str, Field(description="The text chunk.")]


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


class SummarizeVideoEndpoint(Endpoint):
    """Summarize video endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.asr_handle = await AanaDeploymentHandle.create("asr_deployment")
        self.llm_handle = await AanaDeploymentHandle.create("llm_deployment")
        await super().initialize()

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


class SummarizeVideoStreamEndpoint(Endpoint):
    """Summarize video endpoint with streaming output."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.asr_handle = await AanaDeploymentHandle.create("asr_deployment")
        self.llm_handle = await AanaDeploymentHandle.create("llm_deployment")
        await super().initialize()

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
    {
        "name": "summarize_video_stream",
        "path": "/video/summarize_stream",
        "summary": "Summarize a video with streaming output",
        "endpoint_cls": SummarizeVideoStreamEndpoint,
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
