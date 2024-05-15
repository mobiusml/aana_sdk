from typing_extensions import TypedDict

from aana.models.core.image import Image


class FramesDict(TypedDict):
    """Represents a set of frames with ids, timestamps and total duration."""

    frames: list[Image]
    timestamps: list[float]
    duration: float
    frame_ids: list[int]

