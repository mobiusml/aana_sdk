from typing import List
from pydantic import BaseModel

from aana.models.pydantic.base import BaseListModel


class Caption(BaseModel):
    """A model for a caption."""

    __root__: str

    def __str__(self):
        return self.__root__

    class Config:
        schema_extra = {"description": "A caption."}


class CaptionsList(BaseListModel):
    """A model for a list of captions."""

    __root__: List[Caption]

    class Config:
        schema_extra = {"description": "A list of captions."}


class VideoCaptionsList(BaseListModel):
    """A model for a list of captions for a list of videos."""

    __root__: List[CaptionsList]

    class Config:
        schema_extra = {
            "description": (
                "A list of a list of captions. "
                "For a list of videos and a list of captions for each video "
                "and for each video we have a list of captions"
            )
        }
