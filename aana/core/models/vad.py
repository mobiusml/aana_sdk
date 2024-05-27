from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from aana.core.models.time import TimeInterval


class VadParams(BaseModel):
    """A model for the Voice Activity Detection model parameters.

    Attributes:
        chunk_size (float): The maximum length of each vad output chunk.
        merge_onset (float): Onset to be used for the merging operation.
        merge_offset (float): "Optional offset to be used for the merging operation.
    """

    chunk_size: float = Field(
        default=30, ge=10.0, description="The maximum length of each vad output chunk."
    )

    merge_onset: float = Field(
        default=0.5, ge=0.0, description="Onset to be used for the merging operation."
    )

    merge_offset: float | None = Field(
        default=None,
        description="Optional offset to be used for the merging operation.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "description": "Parameters for the voice activity detection model.",
        }
    )


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

    def to_whisper_dict(self) -> dict:
        """Generate dictionary with start, end and segments keys from VADSegment for faster whisper.

        Returns:
            dict: Dictionary with start, end and segments keys
        """
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


VadSegments = Annotated[
    list[VadSegment], Field(description="List of VAD segments", default_factory=list)
]
