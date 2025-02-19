from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aana.core.models.audio import Audio
from aana.core.models.image_chat import TextContent

Role = Literal["system", "user", "assistant"]


class AudioContent(BaseModel):
    """Image content for a chat message.

    Attributes:
        type (Literal["image"]): the type of the content, always "image"
        image (Image): the image
    """

    type: Literal["audio"] = "audio"
    audio: Audio

    model_config = ConfigDict(arbitrary_types_allowed=True)

Content = Annotated[
    TextContent |  AudioContent,
    Field(description="The content of the message, either text or image."),
]


class AudioChatMessage(BaseModel):
    """A chat message with image support.

    Attributes:
        content (list[Content]): the content of the message
        role (Role): the role of the message
    """

    content: list[Content]
    role: Role
    model_config = ConfigDict(
        json_schema_extra={"description": "A chat message with image support."}
    )


class AudioChatDialog(BaseModel):
    """A chat dialog with image support.

    Attributes:
        messages (list[AudioChatMessage]): the list of messages
    """

    messages: list[AudioChatMessage]
    model_config = ConfigDict(
        json_schema_extra={"description": "A chat dialog with audio support."}
    )

    @classmethod
    def from_list(cls, messages: list[dict[str, Any]]) -> "AudioChatDialog":
        """Create an AudioChatDialog from a list of messages.

        Args:
            messages (list[dict[str, str]]): the list of messages

        Returns:
            AudioeChatDialog: the chat dialog
        
        Example:
        ```
        messages = [
            {
                "content": [
                    { "type": "audio", "audio": Audio(...) },
                    { "type": "text", "text": "..." }
                ],
                "role": "system"
            },
            {
                "content": [
                    { "type": "audio", "audio": Audio(...) },
                    { "type": "text", "text": "..." }
                ],
                "role": "user"
            }
        ]
        dialog = AudioChatDialog.from_list(messages)
        ```
        """
        return AudioChatDialog(
            messages=[AudioChatMessage(**message) for message in messages]
        )

    @classmethod
    def from_prompt(cls, prompt: str, audios: list[Audio]) -> "AudioChatDialog":
        """Create an AudioChatDialog from a prompt and a list of audios.

        Args:
            prompt (str): the prompt
            images (list[Audio]): the list of audios

        Returns:
            AudioChatDialog: the chat dialog
        """
        content: list[Content] = [AudioContent(image=image) for image in audios]
        content.append(TextContent(text=prompt))

        return AudioChatDialog(
            messages=[AudioChatMessage(content=content, role="user")]
        )

    def to_objects(self) -> tuple[list[dict], list[Audio]]:
        """Convert AudioChatDialog to messages and audios.

        Returns:
            tuple[list[dict], list[Image]]: the messages and the audios
        """
        dialog_dict = self.model_dump(
            exclude={"messages": {"__all__": {"content": {"__all__": {"audio"}}}}}
        )
        messages = dialog_dict["messages"]
        # images = []
        # for message in self.messages:
        #     for content in message.content:
        #         if content.type == "image":
        #             images.append(content.image)
        audios = [content.audio for message in self.messages for content in message.content if content.type == "audio"]

        return messages, audios

