from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """A chat message.

    Attributes:
        text (str): the text of the message
        role (Role): the role of the message
    """

    content: str
    role: Role

    class Config:
        schema_extra = MappingProxyType(
            {
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

    class Config:
        schema_extra = MappingProxyType(
            {
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