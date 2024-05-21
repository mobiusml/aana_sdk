from pydantic import BaseModel, ConfigDict, Field


class TimeInterval(BaseModel):
    """Pydantic schema for TimeInterval.

    Attributes:
        start (float): Start time in seconds
        end (float): End time in seconds
    """

    start: float = Field(ge=0.0, description="Start time in seconds")
    end: float = Field(ge=0.0, description="End time in seconds")
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Time interval in seconds",
        }
    )
