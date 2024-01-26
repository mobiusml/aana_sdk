from pydantic import BaseModel, Field
from types import MappingProxyType

from pathlib import Path


class VadParams(BaseModel):
    """A model for the Voice Activity Detection model parameters.

    Attributes:
        model_fp (str): Path to the VAD model file.
        vad_segmentation_url (str): Model source location to download the model if model_fp is None.
        onset (float): Threshold to decide a positive voice activity.
        offset (float): Threshold to consider as a silence region.
        min_duration_on (float): Minimum voiced duration.
        min_duration_off (float): "Minimum duration to consider as silence.
    """

    model_fp: Path | None = Field(
        default=None,  # "/nas/jilt/models/vad/whisper-x_vad/pytorch_model.bin",  # None
        description="Model file path.",
    )

    vad_segmentation_url: str = Field(
        default="https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin",
        description="Model source to download from.",
    )

    onset: float = Field(
        default=0.500,
        ge=0.0,
        description="Threshold to decide a positive voice activity.",
    )

    offset: float = Field(
        default=0.363,
        ge=0.0,
        description="Threshold to consider as a silence region.",
    )

    min_duration_on: float = Field(
        default=0.1, ge=0.0, description="Minimum voiced duration."
    )

    min_duration_off: float = Field(
        default=0.1, ge=0.0, description="Minimum duration to consider as silence."
    )

    # for the merge_chunks function:
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
