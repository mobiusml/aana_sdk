from pydantic import BaseModel, Field
from types import MappingProxyType

from pathlib import Path


class VadParams(BaseModel):
    """A model for the Voice Activity Detection model parameters.

    Attributes:
        model (Path | str): Path or URL to the VAD model file.
        onset (float): Threshold to decide a positive voice activity.
        offset (float): Threshold to consider as a silence region.
        min_duration_on (float): Minimum voiced duration.
        min_duration_off (float): Minimum duration to consider as silence.
        chunk_size (float): The maximum lenth of each vad output chunk.
        merge_onset (float): Onset to be used for the merging operation.
        merge_offset (float): "Optional offset to be used for the merging operation.
    """

    chunk_size: float = Field(
        default=30, ge=10.0, description="The maximum lenth of each vad output chunk."
    )

    merge_onset: float = Field(
        default=0.5, ge=0.0, description="Onset to be used for the merging operation."
    )

    merge_offset: float | None = Field(
        default=None,
        description="Optional offset to be used for the merging operation.",
    )

    class Config:
        schema_extra = MappingProxyType(
            {"description": "Parameters for the voice activity detection model."}
        )
