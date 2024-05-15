from pydantic import BaseModel, ConfigDict, Field


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
