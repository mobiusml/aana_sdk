from pydantic import BaseModel, ConfigDict, Field


class VideoParams(BaseModel):
    """A pydantic model for video parameters.

    Attributes:
        extract_fps (int): the number of frames to extract per second
        fast_mode_enabled (bool): whether to use fast mode (keyframes only)
    """

    extract_fps: float = Field(
        default=3.0,
        gt=0.0,
        description=(
            "The number of frames to extract per second. "
            "Can be smaller than 1. For example, 0.5 means 1 frame every 2 seconds."
        ),
    )
    fast_mode_enabled: bool = Field(
        default=True,
        description=(
            "Whether to use fast mode (keyframes only). "
            "extract_fps will be ignored if this is set to True."
        ),
    )
    model_config = ConfigDict(
        json_schema_extra={"description": "Video parameters."}, validate_assignment=True
    )
