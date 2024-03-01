
from pydantic import ConfigDict

from aana.models.pydantic.base import BaseListModel, BaseStringModel


class Caption(BaseStringModel):
    """A model for a caption."""

    model_config = ConfigDict(json_schema_extra={"description": "A caption."})


class CaptionsList(BaseListModel):
    """A model for a list of captions."""

    root: list[Caption]
    model_config = ConfigDict(json_schema_extra={"description": "A list of captions."})


class VideoCaptionsList(BaseListModel):
    """A model for a list of captions for a list of videos."""

    root: list[CaptionsList]
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A list of a list of captions. "
                "For a list of videos and a list of captions for each video "
                "and for each video we have a list of captions"
            )
        }
    )
