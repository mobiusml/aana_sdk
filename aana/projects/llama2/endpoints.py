from collections.abc import AsyncGenerator
from typing import Annotated, TypedDict

from pydantic import Field

from aana.api.api_generation import Endpoint
from aana.core.models.chat import ChatDialog, Prompt
from aana.core.models.sampling import SamplingParams
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle


class LlmGenerateEndpointOutput(TypedDict):
    """LLM generate endpoint output."""

    completion: Annotated[str, Field(description="The generated text.")]


class LlmGenerateEndpoint(Endpoint):
    """LLM generate endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.llm_handle = await AanaDeploymentHandle.create("llm_deployment")
        await super().initialize()

    async def run(
        self, prompt: Prompt, sampling_params: SamplingParams
    ) -> LlmGenerateEndpointOutput:
        """Run the LLM generate endpoint."""
        llm_output = await self.llm_handle.generate(
            prompt=prompt, sampling_params=sampling_params
        )
        return LlmGenerateEndpointOutput(completion=llm_output["text"])


class LlmGenerateStreamEndpoint(Endpoint):
    """LLM generate stream endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.llm_handle = await AanaDeploymentHandle.create("llm_deployment")
        await super().initialize()

    async def run(
        self, prompt: Prompt, sampling_params: SamplingParams
    ) -> AsyncGenerator[LlmGenerateEndpointOutput, None]:
        """Run the LLM generate stream endpoint."""
        async for item in self.llm_handle.generate_stream(
            prompt=prompt, sampling_params=sampling_params
        ):
            yield LlmGenerateEndpointOutput(completion=item["text"])


class LlmChatEndpointOutput(TypedDict):
    """LLM chat endpoint output."""

    message: Annotated[str, Field(description="The response message.")]


class LlmChatEndpoint(Endpoint):
    """LLM chat endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.llm_handle = await AanaDeploymentHandle.create("llm_deployment")
        await super().initialize()

    async def run(
        self, dialog: ChatDialog, sampling_params: SamplingParams
    ) -> LlmChatEndpointOutput:
        """Run the LLM chat endpoint."""
        llm_output = await self.llm_handle.chat(
            dialog=dialog, sampling_params=sampling_params
        )
        return LlmChatEndpointOutput(message=llm_output["message"])


class LlmChatStreamEndpointOutput(TypedDict):
    """LLM chat stream endpoint output."""

    completion: Annotated[str, Field(description="Chunk of generated text.")]


class LlmChatStreamEndpoint(Endpoint):
    """LLM chat stream endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.llm_handle = await AanaDeploymentHandle.create("llm_deployment")
        await super().initialize()

    async def run(
        self, dialog: ChatDialog, sampling_params: SamplingParams
    ) -> AsyncGenerator[LlmChatStreamEndpointOutput, None]:
        """Run the LLM chat stream endpoint."""
        async for item in self.llm_handle.chat_stream(
            dialog=dialog, sampling_params=sampling_params
        ):
            yield LlmChatStreamEndpointOutput(completion=item["text"])
