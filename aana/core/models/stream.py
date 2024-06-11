import uuid
from typing import Annotated

from pydantic import (
    AfterValidator,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
)

from aana.core.models.media import MediaId


class StreamInput(BaseModel):
    """A video stream input.

    The 'url' must be provided.

    Attributes:
        media_id (MediaId): the ID of the video stream. If not provided, it will be generated automatically.
        url (AnyUrl): the URL of the video stream
        channel_number (int): the desired channel of stream to be processed
        extract_fps (float): the number of frames to extract per second
    """

    url: Annotated[
        AnyUrl,
        Field(description="The URL of the video stream."),
        AfterValidator(lambda x: str(x)),
    ]
    channel_number: int = Field(
        default=0,
        ge=0,
        description=("the desired channel of stream"),
    )

    extract_fps: float = Field(
        default=3.0,
        gt=0.0,
        description=(
            "The number of frames to extract per second. "
            "Can be smaller than 1. For example, 0.5 means 1 frame every 2 seconds."
        ),
    )

    media_id: MediaId = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The ID of the video. If not provided, it will be generated automatically.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "description": ("A video Stream. \n" "The 'url' must be provided. \n")
        },
        validate_assignment=True,
        file_upload=False,
    )
