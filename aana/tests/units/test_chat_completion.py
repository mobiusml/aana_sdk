# ruff: noqa: S101, S113
from collections.abc import AsyncGenerator

import pytest
import requests
from openai import NotFoundError, OpenAI
from ray import serve

from aana.core.models.chat import ChatDialog, ChatMessage
from aana.core.models.sampling import SamplingParams
from aana.deployments.base_text_generation_deployment import (
    BaseTextGenerationDeployment,
    ChatOutput,
    LLMOutput,
)


@serve.deployment
class LowercaseLLM(BaseTextGenerationDeployment):
    """Ray deployment that returns the lowercase version of a text structured as an LLM."""

    async def generate_stream(
        self, prompt: str, sampling_params: SamplingParams | None = None
    ) -> AsyncGenerator[LLMOutput, None]:
        """Generate text stream.

        Args:
            prompt (str): The prompt.
            sampling_params (SamplingParams): The sampling parameters.

        Yields:
            LLMOutput: The generated text.
        """
        for char in prompt:
            yield LLMOutput(text=char.lower())

    async def chat(
        self, dialog: ChatDialog, sampling_params: SamplingParams | None = None
    ) -> ChatOutput:
        """Dummy chat method."""
        text = dialog.messages[-1].content
        return ChatOutput(message=ChatMessage(content=text.lower(), role="assistant"))

    async def chat_stream(
        self, dialog: ChatDialog, sampling_params: SamplingParams | None = None
    ) -> AsyncGenerator[LLMOutput, None]:
        """Dummy chat stream method."""
        text = dialog.messages[-1].content
        for char in text:
            yield LLMOutput(text=char.lower())


deployments = [
    {
        "name": "lowercase_deployment",
        "instance": LowercaseLLM,
    }
]


def test_chat_completion(app_setup):
    """Test the chat completion endpoint for OpenAI compatible API."""
    aana_app = app_setup(deployments, [])

    port = aana_app.port
    route_prefix = ""

    # Check that the server is ready
    response = requests.get(f"http://localhost:{port}{route_prefix}/api/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}

    messages = [
        {"role": "user", "content": "Hello World!"},
    ]
    expected_output = messages[0]["content"].lower()

    client = OpenAI(
        api_key="token",
        base_url=f"http://localhost:{port}",
    )

    # Test chat completion endpoint
    completion = client.chat.completions.create(
        messages=messages,
        model="lowercase_deployment",
    )
    assert completion.choices[0].message.content == expected_output

    # Test chat completion endpoint with stream
    stream = client.chat.completions.create(
        messages=messages,
        model="lowercase_deployment",
        stream=True,
    )
    generated_text = ""
    for chunk in stream:
        generated_text += chunk.choices[0].delta.content or ""
    assert generated_text == expected_output

    # Test chat completion endpoint with non-existent model
    with pytest.raises(NotFoundError) as exc_info:
        completion = client.chat.completions.create(
            messages=messages,
            model="non_existent_model",
        )
    assert (
        exc_info.value.body["message"]
        == "The model `non_existent_model` does not exist."
    )
