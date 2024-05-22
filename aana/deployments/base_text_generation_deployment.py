import contextlib
from collections.abc import AsyncGenerator

from typing_extensions import TypedDict

with contextlib.suppress(ImportError):
    pass

from aana.core.chat.chat_template import apply_chat_template
from aana.core.models.chat import ChatDialog, ChatMessage
from aana.core.models.sampling import SamplingParams
from aana.deployments.base_deployment import BaseDeployment, test_cache


class LLMOutput(TypedDict):
    """The output of the LLM model.

    Attributes:
        text (str): the generated text
    """

    text: str


class LLMBatchOutput(TypedDict):
    """The output of the LLM model for a batch of inputs.

    Attributes:
        texts (List[str]): the list of generated texts
    """

    texts: list[str]


class ChatOutput(TypedDict):
    """The output of the chat model.

    Attributes:
        dialog (ChatDialog): the dialog with the responses from the model
    """

    message: ChatMessage


class BaseTextGenerationDeployment(BaseDeployment):
    """Base class for text generation deployments."""

    @test_cache
    async def generate_stream(
        self, prompt: str, sampling_params: SamplingParams | None = None
    ) -> AsyncGenerator[LLMOutput, None]:
        """Generate completion for the given prompt and stream the results.

        Args:
            prompt (str): the prompt
            sampling_params (SamplingParams | None): the sampling parameters

        Yields:
            LLMOutput: the dictionary with the key "text" and the generated text as the value
        """
        raise NotImplementedError

    @test_cache
    async def generate(
        self, prompt: str, sampling_params: SamplingParams | None = None
    ) -> LLMOutput:
        """Generate completion for the given prompt.

        Args:
            prompt (str): the prompt
            sampling_params (SamplingParams | None): the sampling parameters

        Returns:
            LLMOutput: the dictionary with the key "text" and the generated text as the value
        """
        generated_text = ""
        async for chunk in self.generate_stream(prompt, sampling_params):
            generated_text += chunk["text"]
        return LLMOutput(text=generated_text)

    @test_cache
    async def generate_batch(
        self, prompts: list[str], sampling_params: SamplingParams | None = None
    ) -> LLMBatchOutput:
        """Generate completion for the batch of prompts.

        Args:
            prompts (List[str]): the prompts
            sampling_params (SamplingParams | None): the sampling parameters

        Returns:
            LLMBatchOutput: the dictionary with the key "texts"
                            and the list of generated texts as the value
        """
        texts = []
        for prompt in prompts:
            text = await self.generate(prompt, sampling_params)
            texts.append(text["text"])

        return LLMBatchOutput(texts=texts)

    @test_cache
    async def chat(
        self, dialog: ChatDialog, sampling_params: SamplingParams | None = None
    ) -> ChatOutput:
        """Chat with the model.

        Args:
            dialog (ChatDialog): the dialog
            sampling_params (SamplingParams | None): the sampling parameters

        Returns:
            ChatOutput: the dictionary with the key "message"
                        and the response message with a role "assistant"
                        and the generated text as the content
        """
        prompt = apply_chat_template(self.tokenizer, dialog, self.chat_template_name)
        response = await self.generate(prompt, sampling_params)
        response_message = ChatMessage(content=response["text"], role="assistant")
        return ChatOutput(message=response_message)

    @test_cache
    async def chat_stream(
        self, dialog: ChatDialog, sampling_params: SamplingParams | None = None
    ) -> AsyncGenerator[LLMOutput, None]:
        """Chat with the model and stream the responses.

        Args:
            dialog (ChatDialog): the dialog
            sampling_params (SamplingParams | None): the sampling parameters

        Yields:
            LLMOutput: the dictionary with the key "text" and the generated text as the value
        """
        prompt = apply_chat_template(self.tokenizer, dialog, self.chat_template_name)
        async for chunk in self.generate_stream(prompt, sampling_params):
            yield chunk
