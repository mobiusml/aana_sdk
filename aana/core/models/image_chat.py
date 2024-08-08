from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aana.core.models.image import Image

Role = Literal["system", "user", "assistant"]


class TextContent(BaseModel):
    """Text content for a chat message.

    Attributes:
        type (Literal["text"]): the type of the content, always "text"
        text (str): the text of the message
    """

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content for a chat message.

    Attributes:
        type (Literal["image"]): the type of the content, always "image"
        image (Image): the image
    """

    type: Literal["image"] = "image"
    image: Image

    model_config = ConfigDict(arbitrary_types_allowed=True)


Content = Annotated[
    TextContent | ImageContent,
    Field(description="The content of the message, either text or image."),
]


class ImageChatMessage(BaseModel):
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


class ImageChatDialog(BaseModel):
    """A chat dialog with image support.

    Attributes:
        messages (list[ImageChatMessage]): the list of messages
    """

    messages: list[ImageChatMessage]
    model_config = ConfigDict(
        json_schema_extra={"description": "A chat dialog with image support."}
    )

    @classmethod
    def from_list(cls, messages: list[dict[str, Any]]) -> "ImageChatDialog":
        """Create an ImageChatDialog from a list of messages.

        Args:
            messages (list[dict[str, str]]): the list of messages

        Returns:
            ImageChatDialog: the chat dialog
        
        Example:
        ```
        messages = [
            {
                "content": [
                    { "type": "image", "image": Image(...) },
                    { "type": "text", "text": "..." }
                ],
                "role": "system"
            },
            {
                "content": [
                    { "type": "image", "image": Image(...) },
                    { "type": "text", "text": "..." }
                ],
                "role": "user"
            }
        ]
        dialog = ImageChatDialog.from_list(messages)
        ```
        """
        return ImageChatDialog(
            messages=[ImageChatMessage(**message) for message in messages]
        )

    @classmethod
    def from_prompt(cls, prompt: str, images: list[Image]) -> "ImageChatDialog":
        """Create an ImageChatDialog from a prompt and a list of images.

        Args:
            prompt (str): the prompt
            images (list[Image]): the list of images

        Returns:
            ImageChatDialog: the chat dialog
        """
        content: list[Content] = [ImageContent(image=image) for image in images]
        content.append(TextContent(text=prompt))

        return ImageChatDialog(
            messages=[ImageChatMessage(content=content, role="user")]
        )

    def to_objects(self) -> tuple[list[dict], list[Image]]:
        """Convert ImageChatDialog to messages and images.

        Returns:
            tuple[list[dict], list[Image]]: the messages and the images
        """
        dialog_dict = self.model_dump(
            exclude={"messages": {"__all__": {"content": {"__all__": {"image"}}}}}
        )
        messages = dialog_dict["messages"]
        # images = []
        # for message in self.messages:
        #     for content in message.content:
        #         if content.type == "image":
        #             images.append(content.image)
        images = [content.image for message in self.messages for content in message.content if content.type == "image"]

        return messages, images
