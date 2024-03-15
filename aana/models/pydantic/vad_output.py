from pydantic import BaseModel, ConfigDict, Field

from aana.models.pydantic.base import BaseListModel
from aana.models.pydantic.time_interval import TimeInterval


class VadSegment(BaseModel):
    """Pydantic schema for Segment from Voice Activity Detection model.

    Attributes:
        time_interval (TimeInterval): The start and end time of the segment
        segments (list[tuple[float, float]]): smaller voiced segments within a merged vad segment
    """

    time_interval: TimeInterval = Field(description="Time interval of the segment")
    segments: list[tuple[float, float]] = Field(
        description="List of voiced segments within a Segment for ASR"
    )

    def to_whisper_dict(self):
        """Generate dictionary with start, end and segments keys from VADSegment for faster whisper."""
        return {
            "start": self.time_interval.start,
            "end": self.time_interval.end,
            "segments": self.segments,
        }

    model_config = ConfigDict(
        json_schema_extra={
            "description": "VAD Segment for ASR",
        }
    )


class VadSegments(BaseListModel):
    """Pydantic schema for the list of ASR segments."""

    root: list[VadSegment] = Field(
        description="List of VAD segments", default_factory=list
    )

    model_config = ConfigDict(
        json_schema_extra={
            "description": "List of VAD segments",
        }
    )
