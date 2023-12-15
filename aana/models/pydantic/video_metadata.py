from types import MappingProxyType

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """Metadata of a video.

    Attributes:
        title (str): the title of the video
        description (str): the description of the video
    """

    title: str = Field(None, description="The title of the video.")
    description: str = Field(None, description="The description of the video.")

    class Config:
        schema_extra = MappingProxyType(
            {
                "description": "Metadata of a video.",
            }
        )
