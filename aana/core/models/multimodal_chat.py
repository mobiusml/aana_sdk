from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aana.core.models.image import Image

Role = Literal["system", "user", "assistant"]


class FrameVideo(BaseModel):
    """A video represented as a list of images.

    Attributes:
        frames (list[Image]): the frames of the video
    """

    frames: list[Image]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_base64_url(self) -> str:
        """Get the base64 URL of the video.

        Returns:
            str: the base64 URL of the video
        """
        base64_frames = [frame.get_base64() for frame in self.frames]
        return f"data:video/jpeg;base64,{','.join(base64_frames)}"


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


class VideoContent(BaseModel):
    """Video content for a chat message.

    Attributes:
        type (Literal["video"]): the type of the content, always "video"
        video (FrameVideo): the video as a list of frames
    """

    type: Literal["video"] = "video"
    video: FrameVideo

    model_config = ConfigDict(arbitrary_types_allowed=True)


Content = Annotated[
    TextContent | ImageContent | VideoContent,
    Field(description="The content of the message, either text, image, or video."),
]


class MultimodalChatMessage(BaseModel):
    """A chat message with multimodal support.

    Attributes:
        content (list[Content]): the content of the message
        role (Role): the role of the message
    """

    content: list[Content]
    role: Role
    model_config = ConfigDict(
        json_schema_extra={"description": "A chat message with multimodal support."}
    )


class MultimodalChatDialog(BaseModel):
    """A chat dialog with multimodal support.

    Attributes:
        messages (list[MultimodalChatMessage]): the list of messages
    """

    messages: list[MultimodalChatMessage]
    model_config = ConfigDict(
        json_schema_extra={"description": "A chat dialog with multimodal support."}
    )

    @classmethod
    def from_list(cls, messages: list[dict[str, Any]]) -> "MultimodalChatDialog":
        """Create a MultimodalChatDialog from a list of messages.

        Args:
            messages (list[dict[str, str]]): the list of messages

        Returns:
            MultimodalChatDialog: the chat dialog

        Example:
        ```
        messages = [
            {
                "content": [
                    { "type": "text", "text": "..." }
                ],
                "role": "system"
            },
            {
                "content": [
                    { "type": "image", "image": Image(...) },
                    { "type": "video", "video": FrameVideo(...) },
                    { "type": "text", "text": "..." }
                ],
                "role": "user"
            }
        ]
        dialog = MultimodalChatDialog.from_list(messages)
        ```
        """
        return MultimodalChatDialog(
            messages=[MultimodalChatMessage(**message) for message in messages]
        )

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        images: list[Image] | None = None,
        videos: list[FrameVideo] | None = None,
    ) -> "MultimodalChatDialog":
        """Create an ImageChatDialog from a prompt and a list of images and videos.

        Args:
            prompt (str): the prompt
            images (list[Image] | None): the list of images
            videos (list[FrameVideo] | None): the list of videos

        Returns:
            ImageChatDialog: the chat dialog
        """
        content: list[Content] = []
        if images:
            content.extend([ImageContent(image=image) for image in images])
        if videos:
            content.extend([VideoContent(video=video) for video in videos])
        content.append(TextContent(text=prompt))

        return MultimodalChatDialog(
            messages=[MultimodalChatMessage(content=content, role="user")]
        )

    def to_objects(self) -> tuple[list[dict], list[Image], list[FrameVideo]]:
        """Convert ImageChatDialog to messages, images, and videos.

        Returns:
            tuple[list[dict], list[Image], list[FrameVideo]]: the messages, images, and videos
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
        images = [
            content.image
            for message in self.messages
            for content in message.content
            if content.type == "image"
        ]
        videos = [
            content.video
            for message in self.messages
            for content in message.content
            if content.type == "video"
        ]

        return messages, images, videos
