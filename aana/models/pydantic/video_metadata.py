
from pydantic import BaseModel, ConfigDict, Field


class VideoMetadata(BaseModel):
    """Metadata of a video.

    Attributes:
        title (str): the title of the video
        description (str): the description of the video
    """

    title: str = Field(None, description="The title of the video.")
    description: str = Field(None, description="The description of the video.")
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Metadata of a video.",
        }
    )
