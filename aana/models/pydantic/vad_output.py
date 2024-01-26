from types import MappingProxyType  # for immutable dictionary
from aana.models.pydantic.base import BaseListModel
from pydantic import BaseModel, Field
from aana.models.pydantic.time_interval import TimeInterval


class VadSegment(BaseModel):
    """Pydantic schema for Segment from Voice Activity Detection model.

    Attributes:
        time_interval (TimeInterval): The start and end time of the segment
        segments (list[tuple[float, float]]): vad segments within a merged vad segment
    """

    time_interval: TimeInterval = Field(description="Time interval of the segment")
    segments: list[tuple[float, float]] = Field(description="VAD Segment for ASR")

    def to_dict(self):
        """Generate dict from VADSegment."""
        return {
            "start": self.time_interval.start,
            "end": self.time_interval.end,
            "segments": self.segments,
        }

    class Config:
        schema_extra = MappingProxyType(
            {
                "description": "VAD Segment for ASR",
            }
        )


class VadSegments(BaseListModel):
    """Pydantic schema for the list of ASR segments."""

    __root__: list[VadSegment] = Field(
        description="List of VAD segments", default_factory=list
    )

    class Config:
        schema_extra = MappingProxyType(
            {
                "description": "List of VAD segments",
            }
        )
