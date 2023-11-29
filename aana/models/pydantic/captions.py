from types import MappingProxyType

from aana.models.pydantic.base import BaseListModel
from pydantic import BaseModel


class Caption(BaseModel):
    """A model for a caption."""

    __root__: str

    def __str__(self):
        """Convert to string."""
        return self.__root__

    class Config:
        schema_extra = MappingProxyType({"description": "A caption."})


class CaptionsList(BaseListModel):
    """A model for a list of captions."""

    __root__: list[Caption]

    class Config:
        schema_extra = MappingProxyType({"description": "A list of captions."})


class VideoCaptionsList(BaseListModel):
    """A model for a list of captions for a list of videos."""

    __root__: list[CaptionsList]

    class Config:
        schema_extra = MappingProxyType(
            {
                "description": (
                    "A list of a list of captions. "
                    "For a list of videos and a list of captions for each video "
                    "and for each video we have a list of captions"
                )
            }
        )
