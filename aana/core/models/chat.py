from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from aana.core.models.base import pydantic_protected_fields

__all__ = [
    "Role",
    "Prompt",
    "Question",
    "ChatMessage",
    "ChatDialog",
    "ChatCompletionRequest",
    "ChatCompletionChoice",
    "ChatCompletion",
]

Role = Literal["system", "user", "assistant"]
"""
The role of a participant in a conversation.

- "system": Used for instructions or context provided to the model.
- "user": Represents messages from the user.
- "assistant": Represents LLM responses.
"""

Prompt = Annotated[str, Field(description="The prompt for the LLM.")]
"""
The prompt for the LLM.
"""

Question = Annotated[str, Field(description="The question.")]
"""
The question.
"""


class ChatMessage(BaseModel):
    """A chat message.

    Attributes:
        content (str): the text of the message
        role (Role): the role of the message
    """

    content: str
    role: Role
    model_config = ConfigDict(
        json_schema_extra={
            "description": "A chat message.",
            "examples": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                },
                {
                    "role": "assistant",
                    "content": "I am doing well, thank you.",
                },
            ],
        }
    )


class ChatDialog(BaseModel):
    """A chat dialog.

    Attributes:
        messages (list[ChatMessage]): the messages in the dialog
    """

    messages: list[ChatMessage]
    model_config = ConfigDict(
        json_schema_extra={
            "description": "A chat dialog.",
            "examples": [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {
                            "role": "user",
                            "content": "Hello, how are you?",
                        },
                        {
                            "role": "assistant",
                            "content": "I am doing well, thank you.",
                        },
                    ]
                }
            ],
        }
    )

    @classmethod
    def from_list(cls, messages: list[dict[str, str]]) -> "ChatDialog":
        """Create a ChatDialog from a list of dictionaries.

        Args:
            messages (list[dict[str, str]]): the list of messages

        Returns:
            ChatDialog: the chat dialog
        """
        return ChatDialog(messages=[ChatMessage(**message) for message in messages])

    @classmethod
    def from_prompt(cls, prompt: str) -> "ChatDialog":
        """Create a ChatDialog from a prompt.

        Args:
            prompt (str): the prompt

        Returns:
            ChatDialog: the chat dialog
        """
        return ChatDialog(messages=[ChatMessage(content=prompt, role="user")])


class ChatCompletionRequest(BaseModel):
    """A chat completion request for OpenAI compatible API.

    Attributes:
        model (str): the model name (name of the LLM deployment)
        messages (list[ChatMessage]): a list of messages comprising the conversation so far
        temperature (float): float that controls the randomness of the sampling
        top_p (float): float that controls the cumulative probability of the top tokens to consider
        max_tokens (int): the maximum number of tokens to generate
        repetition_penalty (float): float that penalizes new tokens based on whether they appear in the prompt and the generated text so far
        stream (bool): if set, partial message deltas will be sent
    """

    model: str = Field(..., description="The model name (name of the LLM deployment).")
    messages: list[ChatMessage] = Field(
        ..., description="A list of messages comprising the conversation so far."
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Float that controls the randomness of the sampling. "
            "Lower values make the model more deterministic, "
            "while higher values make the model more random. "
            "Zero means greedy sampling."
        ),
    )
    top_p: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description=(
            "Float that controls the cumulative probability of the top tokens to consider. "
            "Must be in (0, 1]. Set to 1 to consider all tokens."
        ),
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="The maximum number of tokens to generate."
    )
    repetition_penalty: float = Field(
        default=1.0,
        description=(
            "Float that penalizes new tokens based on whether they appear in the "
            "prompt and the generated text so far. Values > 1 encourage the model "
            "to use new tokens, while values < 1 encourage the model to repeat tokens. "
            "Default is 1.0 (no penalty)."
        ),
    )

    stream: bool | None = Field(
        default=False,
        description=(
            "If set, partial message deltas will be sent, like in ChatGPT. "
            "Tokens will be sent as data-only server-sent events as they become available, "
            "with the stream terminated by a data: [DONE] message."
        ),
    )

    model_config = ConfigDict(protected_namespaces=(*pydantic_protected_fields,))


class ChatCompletionChoice(BaseModel):
    """A chat completion choice for OpenAI compatible API.

    Attributes:
        index (int): the index of the choice in the list of choices
        message (ChatMessage): a chat completion message generated by the model
    """

    index: int = Field(
        ..., description="The index of the choice in the list of choices."
    )
    message: ChatMessage = Field(
        ..., description="A chat completion message generated by the model."
    )


class ChatCompletion(BaseModel):
    """A chat completion for OpenAI compatible API.

    Attributes:
        id (str): a unique identifier for the chat completion
        model (str): the model used for the chat completion
        created (int): the Unix timestamp (in seconds) of when the chat completion was created
        choices (list[ChatCompletionChoice]): a list of chat completion choices
        object (Literal["chat.completion"]): the object type, which is always `chat.completion`
    """

    id: str = Field(..., description="A unique identifier for the chat completion.")
    model: str = Field(..., description="The model used for the chat completion.")
    created: int = Field(
        ...,
        description="The Unix timestamp (in seconds) of when the chat completion was created.",
    )
    choices: list[ChatCompletionChoice] = Field(
        ...,
        description="A list of chat completion choices.",
    )
    object: Literal["chat.completion"] = Field(
        "chat.completion",
        description="The object type, which is always `chat.completion`.",
    )

    model_config = ConfigDict(protected_namespaces=(*pydantic_protected_fields,))
